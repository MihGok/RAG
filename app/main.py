"""
main.py
───────
FastAPI-приложение с полным набором endpoints.

Endpoints:
──────────
  POST /task                        — LLM/embed/video инференс
  POST /gemini                      — Gemini 2.5 Pro

  GET  /metrics/inference           — скорость и токены по всем моделям
  GET  /metrics/inference/{model}   — детальная статистика одной модели
  DELETE /metrics/inference         — сброс статистики
  POST /metrics/quality/summarization  — ROUGE, BLEU, chrF, BERTScore
  POST /metrics/quality/tagging        — Jaccard, Precision, Recall, F1
  POST /metrics/quality/merging        — Coverage, Containment, Symmetric

  GET  /schemas                     — список JSON-схем
  GET  /models/cached               — загруженные в кэш модели
  GET  /health                      — статус сервиса
"""

import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import FastAPI, HTTPException, Path as FPath
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── Внутренние модули ────────────────────────────────────────────────────────
from transcriber  import transcribe_from_url
from llm_client   import run_llm, embed_text, embed_batch, list_cached_models
from gemini_client import run_gemini_async
from schemas      import get_schema, list_schemas as _list_schemas
import metrics_store
from quality_metrics import (
    compute_summarization_metrics,
    compute_tagging_metrics,
    compute_tagging_metrics_batch,
    compute_merging_metrics,
    compute_merging_metrics_batch,
)

# ────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("main")

app = FastAPI(
    title="LLM Endpoint",
    description="Мультимодальный RAG endpoint: LLM, эмбеддинги, транскрипция, Gemini, метрики.",
    version="2.0.0",
)

# Единый пул для GPU-задач — 1 воркер во избежание конкуренции за GPU
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
    """Универсальный запрос для LLM / embed / video."""

    task_type: str = Field(
        ...,
        description=(
            "'video'  — транскрипция видео (требуется url)\n"
            "'llm'    — инференс любой GGUF chat-модели (требуется text + model_name)\n"
            "'embed'  — векторизация (требуется text или texts + model_name)"
        ),
        examples=["llm"],
    )

    # ── video ────────────────────────────────────────────────────────────
    url: str | None = Field(None, description="URL видео (для task_type='video').")

    # ── llm / embed ──────────────────────────────────────────────────────
    model_name: str | None = Field(
        None,
        description=(
            "Имя .gguf файла модели (напр. 'T-lite-it-1.0-Q5_K_M.gguf') "
            "или абсолютный путь. Файл должен лежать в папке ./models/."
        ),
        examples=["T-lite-it-1.0-Q5_K_M.gguf"],
    )
    text:  str | None       = Field(None, description="Входной текст.")
    texts: list[str] | None = Field(None, description="Список текстов для batch-embed.")

    # ── только llm ───────────────────────────────────────────────────────
    system_prompt: str | None = Field(None, description="Системный промпт.")
    schema_name:   str | None = Field(
        None,
        description="Имя JSON-схемы из /schemas (опционально).",
        examples=["summary"],
    )
    max_tokens:  int   = Field(1024, ge=1,   le=8192)
    temperature: float = Field(0.7,  ge=0.0, le=2.0)
    top_p:       float = Field(0.9,  ge=0.0, le=1.0)
    n_ctx:       int   = Field(4096, ge=512, le=32768, description="Контекстное окно.")


class GeminiRequest(BaseModel):
    """Запрос к Gemini 2.5 Pro."""

    prompt: str = Field(..., description="Текст запроса.")
    system_prompt: str | None = Field(None, description="Системный промпт.")
    schema_name:   str | None = Field(
        None,
        description="Имя JSON-схемы из /schemas для структурированного вывода.",
        examples=["lesson_merge"],
    )
    max_tokens:  int   = Field(8192, ge=1,   le=65536)
    temperature: float = Field(0.7,  ge=0.0, le=2.0)
    top_p:       float = Field(0.95, ge=0.0, le=1.0)
    chat_history: list[dict] | None = Field(
        None,
        description=(
            "История диалога для multi-turn. Формат: "
            '[{"role": "user", "parts": ["текст"]}, '
            ' {"role": "model", "parts": ["ответ"]}]'
        ),
    )


# ── Метрики качества ─────────────────────────────────────────────────────────

class SummarizationMetricsRequest(BaseModel):
    prediction: str = Field(..., description="Сгенерированное резюме.")
    reference:  str = Field(..., description="Эталонное резюме.")
    lang:       str = Field("ru", description="Язык текста (ISO 639-1).")
    use_bertscore: bool = Field(True, description="Включить BERTScore (медленнее).")


class TaggingMetricsRequest(BaseModel):
    predicted_tags: list[str] = Field(..., description="Теги от модели.")
    reference_tags: list[str] = Field(..., description="Эталонные теги.")


class TaggingMetricsBatchRequest(BaseModel):
    predicted_list: list[list[str]] = Field(..., description="Список тегов от модели (batch).")
    reference_list: list[list[str]] = Field(..., description="Список эталонных тегов (batch).")


class MergingMetricsRequest(BaseModel):
    merged_topics: list[str] = Field(
        ..., description="Темы/теги объединённого урока."
    )
    source_topics: list[str] = Field(
        ..., description="Темы/теги исходных уроков (union)."
    )


class MergingMetricsBatchRequest(BaseModel):
    merged_list: list[list[str]] = Field(..., description="Batch: темы merged уроков.")
    source_list: list[list[str]] = Field(..., description="Batch: темы source уроков.")


# ════════════════════════════════════════════════════════════════════════════
#  /task — основной инференс
# ════════════════════════════════════════════════════════════════════════════

@app.post("/task", summary="Инференс: LLM / эмбеддинги / транскрипция видео")
async def handle_task(req: TaskRequest) -> JSONResponse:
    """
    Универсальный endpoint. Маршрутизация по `task_type`:

    | task_type | Что делает                        | Обязательные поля        |
    |-----------|-----------------------------------|--------------------------|
    | video     | Скачивает + транскрибирует видео   | url                      |
    | llm       | Инференс GGUF chat-модели          | model_name, text         |
    | embed     | Векторизация через GGUF embed-мод. | model_name, text / texts |
    """

    # ── video ─────────────────────────────────────────────────────────────
    if req.task_type == "video":
        if not req.url:
            raise HTTPException(422, "Для task_type='video' обязательно поле 'url'.")
        logger.info("Task: video | %s", req.url)
        result = await _run_sync(transcribe_from_url, req.url)
        return JSONResponse({"task_type": "video", "result": result})

    # ── llm ───────────────────────────────────────────────────────────────
    if req.task_type == "llm":
        if not req.model_name:
            raise HTTPException(422, "Для task_type='llm' обязательно поле 'model_name'.")
        if not req.text:
            raise HTTPException(422, "Для task_type='llm' обязательно поле 'text'.")
        schema = _resolve_schema(req.schema_name)
        logger.info("Task: llm | model=%s schema=%s", req.model_name, req.schema_name)
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
        )
        return JSONResponse({"task_type": "llm", "model": req.model_name, "result": result})

    # ── embed ─────────────────────────────────────────────────────────────
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
                "prefix":     "cluster",
            })
        else:
            logger.info("Task: embed | model=%s", req.model_name)
            vector = await _run_sync(embed_text, req.model_name, req.text, req.n_ctx)
            return JSONResponse({
                "task_type":  "embed",
                "model":      req.model_name,
                "embedding":  vector,
                "dimensions": len(vector),
                "prefix":     "cluster",
            })

    raise HTTPException(400, f"Неизвестный task_type: '{req.task_type}'.")


# ════════════════════════════════════════════════════════════════════════════
#  /gemini — Gemini 2.5 Pro
# ════════════════════════════════════════════════════════════════════════════

@app.post("/gemini", summary="Запрос к Gemini 2.5 Pro (для сложных задач)")
async def handle_gemini(req: GeminiRequest) -> JSONResponse:
    """
    Отправляет запрос к Gemini 2.5 Pro.

    Поддерживает структурированный JSON-вывод через `schema_name`.
    Для RAG-задач рекомендуются схемы: `tagging`, `summary`, `lesson_merge`.

    Для multi-turn диалога передайте историю в `chat_history`.
    """
    schema = _resolve_schema(req.schema_name)
    logger.info("Gemini | schema=%s tokens=%d", req.schema_name, req.max_tokens)

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

    return JSONResponse({
        "model":       "gemini-2.5-pro",
        "schema_used": req.schema_name,
        "result":      result,
    })


# ════════════════════════════════════════════════════════════════════════════
#  /metrics/inference — скорость и токены
# ════════════════════════════════════════════════════════════════════════════

@app.get("/metrics/inference", summary="Статистика инференса всех моделей")
async def get_inference_metrics() -> JSONResponse:
    """
    Возвращает агрегированную статистику по всем моделям:
    - количество вызовов
    - суммарные / средние токены
    - avg / P50 / P90 / P99 задержка
    - средняя скорость (tokens/sec)
    """
    return JSONResponse(metrics_store.get_all_stats())


@app.get(
    "/metrics/inference/{model_name:path}",
    summary="Детальная статистика одной модели",
)
async def get_model_metrics(model_name: str = FPath(...)) -> JSONResponse:
    """
    Статистика по конкретной модели + история последних 20 вызовов.
    `model_name` — имя файла .gguf (напр. `T-lite-it-1.0-Q5_K_M.gguf`).
    """
    stats = metrics_store.get_model_stats(model_name)
    if stats is None:
        raise HTTPException(404, f"Модель '{model_name}' не найдена в статистике.")
    history = metrics_store.get_recent_history(model_name, limit=20)
    return JSONResponse({"model": model_name, "stats": stats, "recent_history": history})


@app.delete("/metrics/inference", summary="Сбросить статистику инференса")
async def reset_inference_metrics(model_name: str | None = None) -> JSONResponse:
    """Сбросить статистику. Если model_name не передан — сбрасывает всё."""
    metrics_store.reset_stats(model_name)
    return JSONResponse({"status": "reset", "model": model_name or "all"})


# ════════════════════════════════════════════════════════════════════════════
#  /metrics/quality/summarization
# ════════════════════════════════════════════════════════════════════════════

@app.post(
    "/metrics/quality/summarization",
    summary="Метрики качества суммаризации (ROUGE / BLEU / BERTScore)",
)
async def metrics_summarization(req: SummarizationMetricsRequest) -> JSONResponse:
    """
    Вычисляет метрики сравнения сгенерированного резюме с эталоном:

    - **ROUGE-1 / ROUGE-2 / ROUGE-L** — precision, recall, F1 по n-граммам
    - **BLEU** (sacrebleu corpus BLEU, 0–100)
    - **chrF** (character n-gram F-score, 0–100)
    - **BERTScore** — precision, recall, F1 на эмбеддингах BERT (если use_bertscore=true)

    Используется для оценки задачи суммаризации в RAG-пайплайне.
    """
    try:
        result = await _run_sync(
            compute_summarization_metrics,
            req.prediction,
            req.reference,
            req.lang,
            req.use_bertscore,
        )
    except Exception as exc:
        logger.exception("summarization metrics error: %s", exc)
        raise HTTPException(500, f"Ошибка вычисления метрик: {exc}")

    return JSONResponse({"task": "summarization", "metrics": result})


# ════════════════════════════════════════════════════════════════════════════
#  /metrics/quality/tagging
# ════════════════════════════════════════════════════════════════════════════

@app.post(
    "/metrics/quality/tagging",
    summary="Метрики качества тегирования (Jaccard / Precision / Recall / F1)",
)
async def metrics_tagging(req: TaggingMetricsRequest) -> JSONResponse:
    """
    Метрики для оценки тегирования (множественная классификация):

    - **Jaccard** — схожесть множеств |P∩R| / |P∪R|
    - **Precision** — |P∩R| / |P| (доля верных тегов среди предсказанных)
    - **Recall** — |P∩R| / |R| (доля эталонных тегов, которые нашли)
    - **F1** — гармоническое среднее precision и recall
    - **exact_match** — полное совпадение множеств

    Используется для RAG-задачи автоматического тегирования уроков.
    """
    result = compute_tagging_metrics(req.predicted_tags, req.reference_tags)
    return JSONResponse({"task": "tagging", "metrics": result})


@app.post(
    "/metrics/quality/tagging/batch",
    summary="Batch-метрики тегирования (macro-average)",
)
async def metrics_tagging_batch(req: TaggingMetricsBatchRequest) -> JSONResponse:
    """Пакетная оценка тегирования с macro-усреднением и результатами по каждому примеру."""
    try:
        result = await _run_sync(
            compute_tagging_metrics_batch,
            req.predicted_list,
            req.reference_list,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return JSONResponse({"task": "tagging_batch", "metrics": result})


# ════════════════════════════════════════════════════════════════════════════
#  /metrics/quality/merging
# ════════════════════════════════════════════════════════════════════════════

@app.post(
    "/metrics/quality/merging",
    summary="Метрики качества объединения уроков",
)
async def metrics_merging(req: MergingMetricsRequest) -> JSONResponse:
    """
    Метрики для оценки объединения уроков с одинаковыми названиями:

    - **Coverage** (recall) — доля тем источников, вошедших в объединённый урок
    - **Precision** — доля тем объединённого урока, присутствующих в источниках
    - **F1** — гармоническое среднее coverage и precision
    - **Jaccard** — |M∩S| / |M∪S|
    - **Containment M→S** — |M∩S| / |M| (насколько merged ⊆ source)
    - **Containment S→M** — |M∩S| / |S| (насколько source ⊆ merged)
    - **Symmetric Containment** — среднее двух containment
    - **merged_unique_count** — новые темы, добавленные при объединении
    - **source_uncovered_count** — темы источников, потерянные при слиянии

    Используется для RAG-задачи объединения дублирующихся уроков.
    """
    result = compute_merging_metrics(req.merged_topics, req.source_topics)
    return JSONResponse({"task": "merging", "metrics": result})


@app.post(
    "/metrics/quality/merging/batch",
    summary="Batch-метрики объединения уроков (macro-average)",
)
async def metrics_merging_batch(req: MergingMetricsBatchRequest) -> JSONResponse:
    """Пакетная оценка объединения с macro-усреднением."""
    try:
        result = await _run_sync(
            compute_merging_metrics_batch,
            req.merged_list,
            req.source_list,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return JSONResponse({"task": "merging_batch", "metrics": result})


# ════════════════════════════════════════════════════════════════════════════
#  Вспомогательные endpoints
# ════════════════════════════════════════════════════════════════════════════

@app.get("/schemas", summary="Список доступных JSON-схем")
async def get_schemas() -> JSONResponse:
    """
    Возвращает все доступные схемы структурированного вывода.
    Передайте `schema_name` в /task или /gemini для использования.
    """
    return JSONResponse({"schemas": _list_schemas()})


@app.get("/models/cached", summary="Загруженные в кэш GGUF-модели")
async def get_cached_models() -> JSONResponse:
    """Список моделей, загруженных в VRAM прямо сейчас."""
    return JSONResponse({"cached_models": list_cached_models()})


@app.get("/health", summary="Статус сервиса")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "llm-endpoint", "version": "2.0.0"})


# ════════════════════════════════════════════════════════════════════════════
#  Глобальный обработчик ошибок
# ════════════════════════════════════════════════════════════════════════════

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.exception("Необработанная ошибка: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": "Внутренняя ошибка сервера", "detail": str(exc)},
    )
