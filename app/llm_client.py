"""
llm_client.py
─────────────
Обёртка над llama-cpp-python.

Ключевые изменения:
  • enable_thinking: bool  — режим chain-of-thought для Qwen3 (парсит <think>…</think>)
  • unload_model(name)     — явная выгрузка модели из VRAM + gc.collect()
  • _parse_thinking_output — разбор thinking-блоков без конфликта с GBNF-грамматикой
  • _parse_response        — расширен fallback через re.search для JSON в теле текста
  • Кэш и LRU-вытеснение сохранены без изменений
"""

import gc
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from llama_cpp import Llama, LlamaGrammar
from llama_cpp.llama_grammar import SchemaConverter

import metrics_store

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
#  Конфигурация
# ────────────────────────────────────────────────────────────────────────────
MODELS_DIR: str    = os.getenv("MODELS_DIR", "/models")
N_GPU_LAYERS: int  = int(os.getenv("N_GPU_LAYERS", "-1"))
N_CTX_DEFAULT: int = int(os.getenv("N_CTX_DEFAULT", "4096"))
MAX_CACHED_MODELS  = 4

EMBED_PREFIX: str = "cluster: "

# ────────────────────────────────────────────────────────────────────────────
#  Кэш моделей
# ────────────────────────────────────────────────────────────────────────────
_model_cache: dict[str, Llama] = {}
_cache_order: list[str]        = []
_cache_lock  = threading.Lock()


def _resolve_path(model_name: str) -> str:
    # Базовая санитизация имени: убираем случайные пробелы и слэши по краям
    model_name = model_name.strip().strip("/\\")

    if not model_name:
        raise ValueError("model_name пустое — проверьте константы MAIN_MODEL/THINKING_MODEL/DISTRIBUTION_MODEL")

    if not model_name.endswith(".gguf"):
        raise ValueError(
            f"Имя модели должно заканчиваться на .gguf: '{model_name}'. "
            f"Проверьте константы в loading_workflow.py"
        )

    p = Path(model_name)
    if not p.is_absolute():
        p = Path(MODELS_DIR) / p
    if not p.exists():
        # Показываем что реально лежит в /models для диагностики
        try:
            available = [f for f in Path(MODELS_DIR).iterdir() if f.suffix == ".gguf"]
            avail_str = ", ".join(f.name for f in sorted(available)[:10])
        except Exception:
            avail_str = "(не удалось прочитать директорию)"
        raise FileNotFoundError(
            f"Модель не найдена: {p}\n"
            f"Доступные .gguf в {MODELS_DIR}/: {avail_str}"
        )
    return str(p)


def _get_model(
    model_name: str,
    n_ctx: int = N_CTX_DEFAULT,
    embedding: bool = False,
) -> Llama:
    path      = _resolve_path(model_name)
    cache_key = f"{path}::embed={embedding}::ctx={n_ctx}"

    with _cache_lock:
        if cache_key in _model_cache:
            logger.debug("Кэш-попадание: %s", cache_key)
            return _model_cache[cache_key]

        # LRU-вытеснение
        while len(_cache_order) >= MAX_CACHED_MODELS:
            evict_key = _cache_order.pop(0)
            evicted   = _model_cache.pop(evict_key, None)
            if evicted is not None:
                del evicted     # вызывает llama_free → освобождает VRAM
            logger.info("LRU-выгрузка: %s", evict_key)

        logger.info("Загрузка модели: %s (embed=%s, ctx=%d)", path, embedding, n_ctx)
        model = Llama(
            model_path=path,
            n_gpu_layers=N_GPU_LAYERS,
            n_ctx=n_ctx,
            n_batch=512,
            embedding=embedding,
            verbose=False,
        )
        _model_cache[cache_key] = model
        _cache_order.append(cache_key)
        logger.info("Модель загружена: %s", path)
        return model


# ────────────────────────────────────────────────────────────────────────────
#  Явная выгрузка из VRAM
# ────────────────────────────────────────────────────────────────────────────

def unload_model(model_name: str) -> list[str]:
    """
    Выгружает все варианты модели (разные n_ctx/embed) из кэша и VRAM.

    Args:
        model_name: имя .gguf файла или абсолютный путь.

    Returns:
        Список выгруженных ключей кэша.
    """
    try:
        path = _resolve_path(model_name)
    except FileNotFoundError:
        path = model_name   # файл мог быть удалён вручную

    unloaded: list[str] = []
    with _cache_lock:
        to_remove = [k for k in list(_model_cache) if path in k or model_name in k]
        for key in to_remove:
            model_obj = _model_cache.pop(key, None)
            if model_obj is not None:
                del model_obj   # llama_cpp.__del__ → llama_free
            if key in _cache_order:
                _cache_order.remove(key)
            unloaded.append(key)

    if unloaded:
        gc.collect()
        logger.info("Модель выгружена из VRAM: %s (%d вариантов)", model_name, len(unloaded))
    else:
        logger.warning("Модель не найдена в кэше: %s", model_name)

    return unloaded


# ────────────────────────────────────────────────────────────────────────────
#  Thinking output parser
# ────────────────────────────────────────────────────────────────────────────

def _parse_thinking_output(raw: str) -> tuple[str, str]:
    """
    Разбирает вывод Qwen3 с chain-of-thought.

    Формат Qwen3:
        <think>
        ...цепочка рассуждений...
        </think>
        {итоговый JSON или текст}

    Returns:
        (thinking_text, response_text)
        Если блока <think> нет — ("", raw).
    """
    match = re.search(r'<think>(.*?)</think>', raw, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        response = raw[match.end():].strip()
        return thinking, response
    return "", raw.strip()


# ────────────────────────────────────────────────────────────────────────────
#  Вспомогательные
# ────────────────────────────────────────────────────────────────────────────

def _schema_to_grammar(schema: dict | None) -> LlamaGrammar | None:
    if schema is None:
        return None
    try:
        converter = SchemaConverter(
            prop_order={},
            allow_fetch=False,
            dotall=False,
            raw_pattern=False,
        )
        converter.visit(schema, "")
        grammar_text = converter.format_grammar()
        return LlamaGrammar.from_string(grammar_text, verbose=False)
    except Exception as exc:
        logger.warning("Не удалось создать грамматику: %s", exc)
        return None


def _build_messages(prompt: str, system_prompt: str | None) -> list[dict]:
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": prompt})
    return msgs


def _parse_response(raw: str, schema: dict | None) -> Any:
    """Разбирает JSON из текста с fallback на regex-поиск объекта."""
    if schema is None:
        return {"response": raw.strip()}

    # Прямой парсинг
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # Поиск JSON-объекта в тексте (для моделей без грамматики)
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("Не удалось распарсить JSON ответ, возвращаем как текст.")
    return {"response": raw.strip(), "_parse_error": True}


# ────────────────────────────────────────────────────────────────────────────
#  Основной инференс
# ────────────────────────────────────────────────────────────────────────────

def run_llm(
    model_name: str,
    prompt: str,
    system_prompt: str | None = None,
    response_schema: dict | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    n_ctx: int = N_CTX_DEFAULT,
    enable_thinking: bool = False,
) -> Any:
    """
    Универсальная функция инференса для любой GGUF chat-модели.

    Args:
        model_name:      Имя .gguf файла или абсолютный путь.
        prompt:          Пользовательский запрос.
        system_prompt:   Системный промпт (опционально).
        response_schema: JSON Schema → GBNF-грамматика.
                         ВНИМАНИЕ: грамматика автоматически отключается при
                         enable_thinking=True (конфликт с <think>-тегами).
        max_tokens:      Максимум токенов в ответе.
        temperature:     Температура генерации.
        top_p:           Nucleus sampling.
        n_ctx:           Размер контекстного окна.
        enable_thinking: Режим chain-of-thought (Qwen3).
                         True  → модель думает, _thinking добавляется в ответ.
                         False → добавляет /no_think для отключения у Qwen3.

    Returns:
        dict с полями по схеме + опционально _thinking: str.
    """
    model    = _get_model(model_name, n_ctx=n_ctx, embedding=False)
    messages = _build_messages(prompt, system_prompt)

    # ── Управление thinking-режимом ───────────────────────────────────────
    if not enable_thinking:
        # Qwen3: /no_think в начале user-сообщения отключает CoT
        if messages and messages[-1]["role"] == "user":
            messages[-1] = {
                **messages[-1],
                "content": "/no_think\n" + messages[-1]["content"],
            }

    # ── Грамматика отключается при thinking (конфликт с <think>-тегами) ──
    grammar = None if enable_thinking else _schema_to_grammar(response_schema)

    kwargs: dict = dict(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    if grammar is not None:
        kwargs["grammar"] = grammar

    # ── Инференс ─────────────────────────────────────────────────────────
    t0     = time.perf_counter()
    result = model.create_chat_completion(**kwargs)
    elapsed = time.perf_counter() - t0

    raw     = result["choices"][0]["message"]["content"]
    usage   = result.get("usage", {})
    prompt_t = usage.get("prompt_tokens",     0)
    compl_t  = usage.get("completion_tokens", 0)
    total_t  = usage.get("total_tokens",      0)
    tps     = compl_t / elapsed if elapsed > 0 else 0.0

    metrics_store.record_inference(
        model_name=model_name,
        prompt_tokens=prompt_t,
        completion_tokens=compl_t,
        total_tokens=total_t,
        latency_sec=elapsed,
        tokens_per_sec=tps,
    )
    logger.info(
        "LLM [%s] | thinking=%s | %.2fs | %d tok | %.1f tok/s",
        Path(model_name).name, enable_thinking, elapsed, compl_t, tps,
    )

    # ── Разбор thinking-вывода ────────────────────────────────────────────
    thinking_text, response_text = _parse_thinking_output(raw)

    # Если модель поместила всё в <think> без финального ответа — fallback
    if thinking_text and not response_text:
        logger.warning("Thinking-блок есть, но финальный ответ пуст. Используем весь вывод.")
        response_text = raw

    parsed = _parse_response(response_text or raw, response_schema)

    # Добавляем reasoning trace только когда явно запрошен
    if enable_thinking and thinking_text and isinstance(parsed, dict):
        parsed["_thinking"] = thinking_text

    return parsed


# ────────────────────────────────────────────────────────────────────────────
#  Эмбеддинги
# ────────────────────────────────────────────────────────────────────────────

def embed_text(model_name: str, text: str, n_ctx: int = 2048) -> list[float]:
    """Векторизация одного текста через GGUF embedding-модель."""
    model    = _get_model(model_name, n_ctx=n_ctx, embedding=True)
    prefixed = EMBED_PREFIX + text

    t0     = time.perf_counter()
    result = model.embed(prefixed)
    elapsed = time.perf_counter() - t0

    metrics_store.record_inference(
        model_name=model_name,
        prompt_tokens=len(prefixed.split()),
        completion_tokens=0,
        total_tokens=len(prefixed.split()),
        latency_sec=elapsed,
        tokens_per_sec=0.0,
    )

    return result[0] if isinstance(result[0], list) else result


def embed_batch(
    model_name: str,
    texts: list[str],
    n_ctx: int = 2048,
) -> list[list[float]]:
    """Пакетная векторизация."""
    return [embed_text(model_name, t, n_ctx) for t in texts]


def list_cached_models() -> list[dict]:
    with _cache_lock:
        return [{"cache_key": k, "position": i} for i, k in enumerate(_cache_order)]