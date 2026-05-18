"""
main.py
───────
FastAPI-приложение ML-endpoint.

Новые возможности:
  • enable_thinking: bool в /task — режим CoT для Qwen3 без отдельного endpoint
  • DELETE /models/{model_name} — явная выгрузка модели из VRAM
  • Остальные endpoint без изменений
"""

import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import FastAPI, HTTPException, Path as FPath
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from transcriber   import transcribe_from_url
from llm_client    import run_llm, embed_text, embed_batch, list_cached_models, unload_model
from gemini_client import run_gemini_async
from schemas       import get_schema, list_schemas as _list_schemas
import metrics_store
from quality_metrics import (
    compute_summarization_metrics,
    compute_tagging_metrics,
    compute_tagging_metrics_batch,
    compute_merging_metrics,
    compute_merging_metrics_batch,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("main")

app = FastAPI(
    title="LLM Endpoint",
    description=(
        "Мультимодальный RAG endpoint: LLM (с thinking-режимом), "
        "эмбеддинги, транскрипция, Gemini, метрики качества, управление VRAM."
    ),
    version="2.1.0",
)

# Единый GPU-пул: 1 воркер во избежание конкуренции за VRAM
_executor = ThreadPoolExecutor(max_workers=1)


async def _run_sync(fn, *args, **kwargs) -> Any:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, lambda: fn(*args, **kwargs))


def _resolve_schema(schema_name: str | None) -> dict | None:
    if schema_name is None:
        return None
    try:
        return get_schema(schema_name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


# ════════════════════════════════════════════════════════════════════════════
#  Pydantic-модели
# ════════════════════════════════════════════════════════════════════════════

class TaskRequest(BaseModel):
    """Универсальный запрос: LLM / embed / video."""

    task_type: str = Field(
        ...,
        description=(
            "'video'  — транскрипция видео (url)\n"
            "'llm'    — инференс GGUF chat-модели (text + model_name)\n"
            "'embed'  — векторизация (text или texts + model_name)"
        ),
    )

    # ── video ────────────────────────────────────────────────────────────
    url: str | None = Field(None, description="URL видео (task_type='video').")

    # ── llm / embed ──────────────────────────────────────────────────────
    model_name: str | None = Field(None, description="Имя .gguf файла или путь.")
    text:  str | None       = Field(None, description="Входной текст.")
    texts: list[str] | None = Field(None, description="Список текстов (batch-embed).")

    # ── только llm ───────────────────────────────────────────────────────
    system_prompt: str | None = Field(None, description="Системный промпт.")
    schema_name:   str | None = Field(None, description="Имя JSON-схемы из /schemas.")
    max_tokens:  int   = Field(1024, ge=1,   le=32768)
    temperature: float = Field(0.7,  ge=0.0, le=2.0)
    top_p:       float = Field(0.9,  ge=0.0, le=1.0)
    n_ctx:       int   = Field(4096, ge=512, le=65536, description="Контекстное окно.")

    # ── thinking mode ─────────────────────────────────────────────────────
    enable_thinking: bool = Field(
        False,
        description=(
            "Включить chain-of-thought reasoning (Qwen3 и совместимые модели). "
            "При True: модель генерирует <think>…</think>, "
            "результат содержит поле _thinking с цепочкой рассуждений. "
            "При True грамматика GBNF отключается (конфликт с thinking-тегами), "
            "JSON извлекается regex-поиском из финального ответа. "
            "При False: добавляет /no_think в Qwen3."
        ),
    )


class GeminiRequest(BaseModel):
    prompt: str
    system_prompt: str | None = None
    schema_name:   str | None = None
    max_tokens:  int   = Field(8192, ge=1, le=65536)
    temperature: float = Field(0.7,  ge=0.0, le=2.0)
    top_p:       float = Field(0.95, ge=0.0, le=1.0)
    chat_history: list[dict] | None = None


class SummarizationMetricsRequest(BaseModel):
    prediction:    str
    reference:     str
    lang:          str  = "ru"
    use_bertscore: bool = True


class TaggingMetricsRequest(BaseModel):
    predicted_tags: list[str]
    reference_tags: list[str]


class TaggingMetricsBatchRequest(BaseModel):
    predicted_list: list[list[str]]
    reference_list: list[list[str]]


class MergingMetricsRequest(BaseModel):
    merged_topics: list[str]
    source_topics: list[str]


class MergingMetricsBatchRequest(BaseModel):
    merged_list: list[list[str]]
    source_list: list[list[str]]


# ════════════════════════════════════════════════════════════════════════════
#  /task — основной инференс
# ════════════════════════════════════════════════════════════════════════════

@app.post("/task", summary="Инференс: LLM (с thinking) / эмбеддинги / транскрипция")
async def handle_task(req: TaskRequest) -> JSONResponse:
    """
    Маршрутизация по task_type:

    | task_type | Обязательные поля          | Заметки                              |
    |-----------|----------------------------|--------------------------------------|
    | video     | url                        |                                      |
    | llm       | model_name, text           | enable_thinking управляет CoT        |
    | embed     | model_name, text / texts   |                                      |
    """

    if req.task_type == "video":
        if not req.url:
            raise HTTPException(422, "Для task_type='video' обязательно поле 'url'.")
        logger.info("Task: video | %s", req.url)
        result = await _run_sync(transcribe_from_url, req.url)
        return JSONResponse({"task_type": "video", "result": result})

    if req.task_type == "llm":
        if not req.model_name:
            raise HTTPException(422, "Для task_type='llm' обязательно поле 'model_name'.")
        if not req.text:
            raise HTTPException(422, "Для task_type='llm' обязательно поле 'text'.")
        schema = _resolve_schema(req.schema_name)
        logger.info(
            "Task: llm | model=%s schema=%s thinking=%s",
            req.model_name, req.schema_name, req.enable_thinking,
        )
        result = await _run_sync(
            run_llm,
            req.model_name,
            req.text,
            req.system_prompt,
            schema,
            req.max_tokens,
            req.temperature,
            req.top_p,
            req.n_ctx,
            req.enable_thinking,
        )
        return JSONResponse({
            "task_type": "llm",
            "model":     req.model_name,
            "thinking":  req.enable_thinking,
            "result":    result,
        })

    if req.task_type == "embed":
        if not req.model_name:
            raise HTTPException(422, "Для task_type='embed' обязательно поле 'model_name'.")
        if not req.text and not req.texts:
            raise HTTPException(422, "Для task_type='embed' нужно 'text' или 'texts'.")
        if req.texts:
            logger.info("Task: embed batch | model=%s count=%d", req.model_name, len(req.texts))
            vectors = await _run_sync(embed_batch, req.model_name, req.texts, req.n_ctx)
            return JSONResponse({
                "task_type":  "embed",
                "model":      req.model_name,
                "count":      len(vectors),
                "embeddings": vectors,
            })
        else:
            logger.info("Task: embed | model=%s", req.model_name)
            vector = await _run_sync(embed_text, req.model_name, req.text, req.n_ctx)
            return JSONResponse({
                "task_type":  "embed",
                "model":      req.model_name,
                "embedding":  vector,
                "dimensions": len(vector),
            })

    raise HTTPException(400, f"Неизвестный task_type: '{req.task_type}'.")


# ════════════════════════════════════════════════════════════════════════════
#  /models — управление VRAM
# ════════════════════════════════════════════════════════════════════════════

@app.get("/models/cached", summary="Модели в кэше VRAM")
async def get_cached_models() -> JSONResponse:
    return JSONResponse({"cached_models": list_cached_models()})


@app.delete(
    "/models/{model_name:path}",
    summary="Выгрузить модель из VRAM",
)
async def unload_model_endpoint(model_name: str = FPath(...)) -> JSONResponse:
    """
    Принудительно выгружает модель из VRAM и кэша.
    Поддерживает частичное имя файла или абсолютный путь.

    Полезно между этапами пайплайна когда разные этапы используют разные модели
    и держать их всех в VRAM одновременно нет смысла (особенно 9B+ моделей).

    Returns:
        {"unloaded": [...keys], "count": N}
    """
    logger.info("Запрос на выгрузку модели: %s", model_name)
    unloaded = await _run_sync(unload_model, model_name)
    return JSONResponse({
        "unloaded": unloaded,
        "count":    len(unloaded),
        "model":    model_name,
    })


# ════════════════════════════════════════════════════════════════════════════
#  /gemini
# ════════════════════════════════════════════════════════════════════════════

@app.post("/gemini", summary="Запрос к Gemini 2.5 Pro")
async def handle_gemini(req: GeminiRequest) -> JSONResponse:
    schema = _resolve_schema(req.schema_name)
    try:
        result = await run_gemini_async(
            prompt=req.prompt,
            system_prompt=req.system_prompt,
            response_schema=schema,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            chat_history=req.chat_history,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Gemini ошибка: %s", exc)
        raise HTTPException(status_code=502, detail=f"Ошибка Gemini API: {exc}")

    return JSONResponse({"model": "gemini-2.5-pro", "schema_used": req.schema_name, "result": result})


# ════════════════════════════════════════════════════════════════════════════
#  /metrics/inference
# ════════════════════════════════════════════════════════════════════════════

@app.get("/metrics/inference")
async def get_inference_metrics() -> JSONResponse:
    return JSONResponse(metrics_store.get_all_stats())


@app.get("/metrics/inference/{model_name:path}")
async def get_model_metrics(model_name: str = FPath(...)) -> JSONResponse:
    stats = metrics_store.get_model_stats(model_name)
    if stats is None:
        raise HTTPException(404, f"Модель '{model_name}' не найдена.")
    history = metrics_store.get_recent_history(model_name, limit=20)
    return JSONResponse({"model": model_name, "stats": stats, "recent_history": history})


@app.delete("/metrics/inference")
async def reset_inference_metrics(model_name: str | None = None) -> JSONResponse:
    metrics_store.reset_stats(model_name)
    return JSONResponse({"status": "reset", "model": model_name or "all"})


# ════════════════════════════════════════════════════════════════════════════
#  /metrics/quality
# ════════════════════════════════════════════════════════════════════════════

@app.post("/metrics/quality/summarization")
async def metrics_summarization(req: SummarizationMetricsRequest) -> JSONResponse:
    try:
        result = await _run_sync(
            compute_summarization_metrics,
            req.prediction, req.reference, req.lang, req.use_bertscore,
        )
    except Exception as exc:
        raise HTTPException(500, f"Ошибка метрик: {exc}")
    return JSONResponse({"task": "summarization", "metrics": result})


@app.post("/metrics/quality/tagging")
async def metrics_tagging(req: TaggingMetricsRequest) -> JSONResponse:
    return JSONResponse({"task": "tagging",
                         "metrics": compute_tagging_metrics(req.predicted_tags, req.reference_tags)})


@app.post("/metrics/quality/tagging/batch")
async def metrics_tagging_batch(req: TaggingMetricsBatchRequest) -> JSONResponse:
    try:
        result = await _run_sync(compute_tagging_metrics_batch, req.predicted_list, req.reference_list)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return JSONResponse({"task": "tagging_batch", "metrics": result})


@app.post("/metrics/quality/merging")
async def metrics_merging(req: MergingMetricsRequest) -> JSONResponse:
    return JSONResponse({"task": "merging",
                         "metrics": compute_merging_metrics(req.merged_topics, req.source_topics)})


@app.post("/metrics/quality/merging/batch")
async def metrics_merging_batch(req: MergingMetricsBatchRequest) -> JSONResponse:
    try:
        result = await _run_sync(compute_merging_metrics_batch, req.merged_list, req.source_list)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return JSONResponse({"task": "merging_batch", "metrics": result})


# ════════════════════════════════════════════════════════════════════════════
#  Вспомогательные
# ════════════════════════════════════════════════════════════════════════════

@app.get("/schemas")
async def get_schemas() -> JSONResponse:
    return JSONResponse({"schemas": _list_schemas()})


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "llm-endpoint", "version": "2.1.0"})


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.exception("Необработанная ошибка: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": "Внутренняя ошибка сервера", "detail": str(exc)},
    )