"""
metrics_store.py
────────────────
Потокобезопасное in-memory хранилище метрик инференса.

Собирает статистику по каждой модели:
  - количество вызовов
  - суммарные / средние токены
  - суммарное / среднее время
  - средняя скорость (tokens/sec)
  - гистограмма латентностей (P50 / P90 / P99)
  - история последних N вызовов

Данные хранятся в памяти процесса.
При перезапуске контейнера сбрасываются.
Если нужна персистентность — подключите PostgreSQL/Mongo.
"""

import time
import threading
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import DefaultDict

# ────────────────────────────────────────────────────────────────────────────
#  Структуры данных
# ────────────────────────────────────────────────────────────────────────────

HISTORY_WINDOW = 200   # хранить последних N запросов на модель


@dataclass
class InferenceRecord:
    """Одна запись о вызове модели."""
    timestamp: float
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_sec: float
    tokens_per_sec: float


@dataclass
class ModelStats:
    """Агрегированная статистика по одной модели."""
    total_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_latency_sec: float = 0.0
    latencies: deque = field(default_factory=lambda: deque(maxlen=HISTORY_WINDOW))
    # Для хранения последних N полных записей
    history: deque = field(default_factory=lambda: deque(maxlen=HISTORY_WINDOW))

    # ── Вычисляемые свойства ──────────────────────────────────────────────

    @property
    def avg_latency_sec(self) -> float:
        return self.total_latency_sec / self.total_calls if self.total_calls else 0.0

    @property
    def avg_tokens_per_sec(self) -> float:
        tps_vals = [r.tokens_per_sec for r in self.history if r.tokens_per_sec > 0]
        return statistics.mean(tps_vals) if tps_vals else 0.0

    @property
    def p50_latency(self) -> float:
        return statistics.median(self.latencies) if self.latencies else 0.0

    @property
    def p90_latency(self) -> float:
        if not self.latencies:
            return 0.0
        s = sorted(self.latencies)
        idx = int(len(s) * 0.90)
        return s[min(idx, len(s) - 1)]

    @property
    def p99_latency(self) -> float:
        if not self.latencies:
            return 0.0
        s = sorted(self.latencies)
        idx = int(len(s) * 0.99)
        return s[min(idx, len(s) - 1)]

    def to_dict(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "avg_latency_sec": round(self.avg_latency_sec, 3),
            "p50_latency_sec": round(self.p50_latency, 3),
            "p90_latency_sec": round(self.p90_latency, 3),
            "p99_latency_sec": round(self.p99_latency, 3),
            "avg_tokens_per_sec": round(self.avg_tokens_per_sec, 1),
        }


# ────────────────────────────────────────────────────────────────────────────
#  Хранилище (singleton)
# ────────────────────────────────────────────────────────────────────────────

_lock = threading.Lock()
_stats: DefaultDict[str, ModelStats] = defaultdict(ModelStats)
_started_at: float = time.time()


# ────────────────────────────────────────────────────────────────────────────
#  Публичный API
# ────────────────────────────────────────────────────────────────────────────

def record_inference(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    latency_sec: float,
    tokens_per_sec: float,
) -> None:
    """Записать результат одного инференса. Вызывается из llm_client."""
    rec = InferenceRecord(
        timestamp=time.time(),
        model_name=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_sec=latency_sec,
        tokens_per_sec=tokens_per_sec,
    )
    with _lock:
        s = _stats[model_name]
        s.total_calls              += 1
        s.total_prompt_tokens      += prompt_tokens
        s.total_completion_tokens  += completion_tokens
        s.total_tokens             += total_tokens
        s.total_latency_sec        += latency_sec
        s.latencies.append(latency_sec)
        s.history.append(rec)


def get_all_stats() -> dict:
    """Вернуть агрегированную статистику по всем моделям."""
    with _lock:
        return {
            "uptime_sec": round(time.time() - _started_at, 1),
            "models": {
                name: s.to_dict()
                for name, s in _stats.items()
            },
        }


def get_model_stats(model_name: str) -> dict | None:
    """Вернуть статистику по конкретной модели."""
    with _lock:
        s = _stats.get(model_name)
        return s.to_dict() if s else None


def get_recent_history(model_name: str, limit: int = 20) -> list[dict]:
    """Вернуть последние N записей инференса для модели."""
    with _lock:
        s = _stats.get(model_name)
        if not s:
            return []
        records = list(s.history)[-limit:]
        return [
            {
                "timestamp": r.timestamp,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "latency_sec": round(r.latency_sec, 3),
                "tokens_per_sec": round(r.tokens_per_sec, 1),
            }
            for r in records
        ]


def reset_stats(model_name: str | None = None) -> None:
    """Сбросить статистику (всех моделей или одной)."""
    global _stats, _started_at
    with _lock:
        if model_name:
            _stats.pop(model_name, None)
        else:
            _stats = defaultdict(ModelStats)
            _started_at = time.time()
