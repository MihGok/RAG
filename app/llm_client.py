"""
llm_client.py
─────────────
Обёртка над llama-cpp-python.

Ключевые изменения:
  • Имя / путь модели передаётся в каждом запросе — не захардкожено.
  • Кэш загруженных моделей по пути (LRU-подобный, до MAX_CACHED_MODELS).
  • Автоматическая запись метрик инференса (скорость, токены, задержка)
    в модуль metrics_store — подхватывается endpoint /metrics/inference.
"""

import os
import time
import json
import logging
import threading
from pathlib import Path
from typing import Any

from llama_cpp import Llama, LlamaGrammar
from llama_cpp.llama_grammar import SchemaConverter

import metrics_store  # внутренний модуль хранения метрик

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
#  Конфигурация
# ────────────────────────────────────────────────────────────────────────────
MODELS_DIR: str     = os.getenv("MODELS_DIR", "/models")
N_GPU_LAYERS: int   = int(os.getenv("N_GPU_LAYERS", "-1"))
N_CTX_DEFAULT: int  = int(os.getenv("N_CTX_DEFAULT", "4096"))
MAX_CACHED_MODELS   = 4   # максимум одновременно загруженных моделей в RAM/VRAM

EMBED_PREFIX: str = "cluster: "

# ────────────────────────────────────────────────────────────────────────────
#  Кэш моделей
# ────────────────────────────────────────────────────────────────────────────
_model_cache: dict[str, Llama] = {}        # path -> Llama instance
_cache_order: list[str] = []               # FIFO для вытеснения
_cache_lock  = threading.Lock()


def _resolve_path(model_name: str) -> str:
    """
    Если передан только filename (без /), добавляет MODELS_DIR.
    Иначе возвращает как есть (абсолютный путь).
    """
    p = Path(model_name)
    if not p.is_absolute():
        p = Path(MODELS_DIR) / p
    if not p.exists():
        raise FileNotFoundError(
            f"Модель не найдена: {p}. "
            f"Положите .gguf файл в {MODELS_DIR}/ или укажите абсолютный путь."
        )
    return str(p)


def _get_model(
    model_name: str,
    n_ctx: int = N_CTX_DEFAULT,
    embedding: bool = False,
) -> Llama:
    """
    Вернуть Llama-инстанс из кэша или загрузить новый.
    При переполнении кэша выгружает самую старую модель.
    """
    path = _resolve_path(model_name)
    cache_key = f"{path}::embed={embedding}::ctx={n_ctx}"

    with _cache_lock:
        if cache_key in _model_cache:
            logger.debug("Кэш-попадание: %s", cache_key)
            return _model_cache[cache_key]

        # Вытеснение если кэш полон
        while len(_cache_order) >= MAX_CACHED_MODELS:
            evict_key = _cache_order.pop(0)
            _model_cache.pop(evict_key, None)
            logger.info("Выгрузка модели из кэша: %s", evict_key)

        logger.info("Загрузка модели: %s (embedding=%s, ctx=%d)", path, embedding, n_ctx)
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
#  Вспомогательные функции
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
    if schema is None:
        return {"response": raw.strip()}
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        logger.warning("Не удалось распарсить JSON ответ, возвращаем как текст.")
        return {"response": raw.strip(), "_parse_error": True}


# ────────────────────────────────────────────────────────────────────────────
#  Публичный API
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
) -> Any:
    """
    Универсальная функция инференса для любой chat-модели в формате GGUF.

    Args:
        model_name:      Имя файла (.gguf) или абсолютный путь к модели.
        prompt:          Пользовательский запрос.
        system_prompt:   Системный промпт (опционально).
        response_schema: JSON Schema → GBNF-грамматика для структурированного вывода.
        max_tokens:      Максимум токенов в ответе.
        temperature:     Температура генерации.
        top_p:           Nucleus sampling.
        n_ctx:           Размер контекстного окна.

    Returns:
        dict с полями по схеме, или {"response": str} если схема не задана.
    """
    model = _get_model(model_name, n_ctx=n_ctx, embedding=False)
    grammar = _schema_to_grammar(response_schema)
    messages = _build_messages(prompt, system_prompt)

    kwargs: dict = dict(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    if grammar is not None:
        kwargs["grammar"] = grammar

    t0 = time.perf_counter()
    result = model.create_chat_completion(**kwargs)
    elapsed = time.perf_counter() - t0

    raw      = result["choices"][0]["message"]["content"]
    usage    = result.get("usage", {})
    prompt_t = usage.get("prompt_tokens", 0)
    compl_t  = usage.get("completion_tokens", 0)
    total_t  = usage.get("total_tokens", 0)
    tps      = compl_t / elapsed if elapsed > 0 else 0.0

    # Запись метрик инференса
    metrics_store.record_inference(
        model_name=model_name,
        prompt_tokens=prompt_t,
        completion_tokens=compl_t,
        total_tokens=total_t,
        latency_sec=elapsed,
        tokens_per_sec=tps,
    )
    logger.info(
        "LLM [%s] | %.2fs | %d tok | %.1f tok/s",
        Path(model_name).name, elapsed, compl_t, tps,
    )

    return _parse_response(raw, response_schema)


def embed_text(model_name: str, text: str, n_ctx: int = 2048) -> list[float]:
    """
    Получить вектор текста через embedding-модель GGUF.
    Автоматически добавляет префикс "cluster: ".

    Args:
        model_name: Имя .gguf файла (напр. "Qwen3-Embedding-0.6B-f16.gguf").
        text:       Исходный текст.
        n_ctx:      Контекстное окно.

    Returns:
        list[float] — вектор.
    """
    model = _get_model(model_name, n_ctx=n_ctx, embedding=True)
    prefixed = EMBED_PREFIX + text

    t0 = time.perf_counter()
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

    if isinstance(result[0], list):
        return result[0]
    return result


def embed_batch(
    model_name: str,
    texts: list[str],
    n_ctx: int = 2048,
) -> list[list[float]]:
    """Пакетная векторизация. Каждый текст получает префикс 'cluster: '."""
    return [embed_text(model_name, t, n_ctx) for t in texts]


def list_cached_models() -> list[dict]:
    """Вернуть список загруженных в кэш моделей."""
    with _cache_lock:
        return [
            {"cache_key": k, "position": i}
            for i, k in enumerate(_cache_order)
        ]
