"""
quality_metrics.py
──────────────────
Вычисление метрик качества для трёх RAG-задач:

  1. Суммаризация  → ROUGE-1/2/L, BLEU (sacrebleu), BERTScore
  2. Тегирование   → Jaccard, Precision, Recall, F1 (на множествах тегов)
  3. Объединение уроков → Overlap, Containment, Symmetric Containment,
                          Pairwise F1 (схожесть множества тем/заголовков)

Все функции принимают prediction(s) и reference(s) в виде строк или списков.
Возвращают dict с числовыми метриками (от 0.0 до 1.0 или 0–100 для BLEU).

BERTScore считается на GPU если доступен.
"""

import logging
from typing import Union

logger = logging.getLogger(__name__)

# ── lazily-imported тяжёлые библиотеки ──────────────────────────────────────
_rouge_scorer = None
_bertscore_fn = None


def _get_rouge():
    global _rouge_scorer
    if _rouge_scorer is None:
        from rouge_score import rouge_scorer as rs
        _rouge_scorer = rs.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=False,
        )
    return _rouge_scorer


def _get_bertscore():
    """BERTScore импортируется лениво — требует torch."""
    global _bertscore_fn
    if _bertscore_fn is None:
        from bert_score import score as bs_score
        _bertscore_fn = bs_score
    return _bertscore_fn


# ════════════════════════════════════════════════════════════════════════════
#  1. СУММАРИЗАЦИЯ
# ════════════════════════════════════════════════════════════════════════════

def compute_summarization_metrics(
    prediction: str,
    reference: str,
    lang: str = "ru",
    use_bertscore: bool = True,
) -> dict:
    """
    Метрики для оценки качества суммаризации.

    Args:
        prediction:     Сгенерированное резюме.
        reference:      Эталонное резюме.
        lang:           Язык текста (для BERTScore). "ru" | "en" | "multilingual"
        use_bertscore:  Включить BERTScore (тяжелее, но точнее).

    Returns:
        {
          "rouge1":     {"precision": f, "recall": f, "f1": f},
          "rouge2":     {"precision": f, "recall": f, "f1": f},
          "rougeL":     {"precision": f, "recall": f, "f1": f},
          "bleu":       float,   # sacrebleu corpus BLEU (0–100)
          "chrf":       float,   # chrF score (0–100)
          "bertscore":  {"precision": f, "recall": f, "f1": f},  # если запрошен
        }
    """
    result: dict = {}

    # ── ROUGE ────────────────────────────────────────────────────────────
    scorer = _get_rouge()
    scores = scorer.score(reference, prediction)
    for key in ("rouge1", "rouge2", "rougeL"):
        s = scores[key]
        result[key] = {
            "precision": round(s.precision, 4),
            "recall":    round(s.recall,    4),
            "f1":        round(s.fmeasure,  4),
        }

    # ── BLEU (sacrebleu) ─────────────────────────────────────────────────
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu([prediction], [[reference]])
        result["bleu"] = round(bleu.score, 2)

        chrf = sacrebleu.corpus_chrf([prediction], [[reference]])
        result["chrf"] = round(chrf.score, 2)
    except Exception as exc:
        logger.warning("sacrebleu ошибка: %s", exc)
        result["bleu"] = None
        result["chrf"] = None

    # ── BERTScore ────────────────────────────────────────────────────────
    if use_bertscore:
        try:
            bs_fn = _get_bertscore()
            # model_type для русского: "setu4993/LaBSE" или "bert-base-multilingual-cased"
            bs_model = (
                "bert-base-multilingual-cased"
                if lang in ("ru", "multilingual")
                else "bert-base-uncased"
            )
            P, R, F1 = bs_fn(
                [prediction], [reference],
                model_type=bs_model,
                lang=lang,
                verbose=False,
            )
            result["bertscore"] = {
                "precision": round(P.mean().item(), 4),
                "recall":    round(R.mean().item(), 4),
                "f1":        round(F1.mean().item(), 4),
            }
        except Exception as exc:
            logger.warning("BERTScore ошибка: %s", exc)
            result["bertscore"] = None

    return result


# ════════════════════════════════════════════════════════════════════════════
#  2. ТЕГИРОВАНИЕ
# ════════════════════════════════════════════════════════════════════════════

def _normalize_tags(tags: list[str]) -> set[str]:
    """Нормализация тегов: нижний регистр, trim."""
    return {t.strip().lower() for t in tags if t.strip()}


def compute_tagging_metrics(
    predicted_tags: list[str],
    reference_tags: list[str],
) -> dict:
    """
    Метрики для оценки качества тегирования (множественная классификация).

    Args:
        predicted_tags:  Теги, сгенерированные моделью.
        reference_tags:  Эталонные теги.

    Returns:
        {
          "jaccard":          float,   # |P ∩ R| / |P ∪ R|
          "precision":        float,   # |P ∩ R| / |P|
          "recall":           float,   # |P ∩ R| / |R|
          "f1":               float,   # гармоническое среднее P и R
          "exact_match":      bool,    # P == R
          "predicted_count":  int,
          "reference_count":  int,
          "overlap_count":    int,
        }
    """
    P = _normalize_tags(predicted_tags)
    R = _normalize_tags(reference_tags)

    intersection = P & R
    union        = P | R

    jaccard   = len(intersection) / len(union)        if union        else 0.0
    precision = len(intersection) / len(P)            if P            else 0.0
    recall    = len(intersection) / len(R)            if R            else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "jaccard":         round(jaccard,   4),
        "precision":       round(precision, 4),
        "recall":          round(recall,    4),
        "f1":              round(f1,        4),
        "exact_match":     P == R,
        "predicted_count": len(P),
        "reference_count": len(R),
        "overlap_count":   len(intersection),
    }


def compute_tagging_metrics_batch(
    predicted_list: list[list[str]],
    reference_list: list[list[str]],
) -> dict:
    """
    Пакетные метрики тегирования (macro-average по всем примерам).
    """
    if len(predicted_list) != len(reference_list):
        raise ValueError("predicted_list и reference_list должны быть одинаковой длины.")

    all_metrics = [
        compute_tagging_metrics(p, r)
        for p, r in zip(predicted_list, reference_list)
    ]

    keys = ["jaccard", "precision", "recall", "f1"]
    macro = {
        k: round(sum(m[k] for m in all_metrics) / len(all_metrics), 4)
        for k in keys
    }
    macro["exact_match_rate"] = round(
        sum(1 for m in all_metrics if m["exact_match"]) / len(all_metrics), 4
    )
    macro["n_samples"] = len(all_metrics)
    macro["per_sample"] = all_metrics
    return macro


# ════════════════════════════════════════════════════════════════════════════
#  3. ОБЪЕДИНЕНИЕ УРОКОВ (Lesson Merging)
# ════════════════════════════════════════════════════════════════════════════

def compute_merging_metrics(
    merged_topics: list[str],
    source_topics: list[str],
) -> dict:
    """
    Метрики для оценки качества объединения уроков с одинаковыми названиями.

    Проверяет, насколько полно и точно объединённый урок покрывает темы
    исходных уроков.

    Args:
        merged_topics:  Список тем/ключевых слов объединённого урока.
        source_topics:  Список тем/ключевых слов исходных уроков (union).

    Returns:
        {
          "coverage":               float,   # доля тем источников, вошедших в merged
          "precision":              float,   # доля тем merged, которые есть в источниках
          "f1":                     float,
          "jaccard":                float,
          "containment_merged_in_source": float,  # |M ∩ S| / |M|
          "containment_source_in_merged": float,  # |M ∩ S| / |S|
          "symmetric_containment":  float,   # среднее двух containment
          "merged_unique_count":    int,     # новые темы, которых не было в источниках
        }
    """
    M = _normalize_tags(merged_topics)
    S = _normalize_tags(source_topics)

    intersection = M & S
    union        = M | S

    coverage  = len(intersection) / len(S) if S else 0.0   # recall
    precision = len(intersection) / len(M) if M else 0.0
    f1 = (
        2 * precision * coverage / (precision + coverage)
        if (precision + coverage) > 0
        else 0.0
    )
    jaccard = len(intersection) / len(union) if union else 0.0

    cont_m_in_s = len(intersection) / len(M) if M else 0.0
    cont_s_in_m = len(intersection) / len(S) if S else 0.0
    sym_cont    = (cont_m_in_s + cont_s_in_m) / 2

    return {
        "coverage":                      round(coverage,   4),
        "precision":                     round(precision,  4),
        "f1":                            round(f1,         4),
        "jaccard":                       round(jaccard,    4),
        "containment_merged_in_source":  round(cont_m_in_s, 4),
        "containment_source_in_merged":  round(cont_s_in_m, 4),
        "symmetric_containment":         round(sym_cont,   4),
        "merged_topic_count":            len(M),
        "source_topic_count":            len(S),
        "overlap_count":                 len(intersection),
        "merged_unique_count":           len(M - S),   # темы только в merged
        "source_uncovered_count":        len(S - M),   # темы источников, потерянные при merge
    }


def compute_merging_metrics_batch(
    merged_list: list[list[str]],
    source_list: list[list[str]],
) -> dict:
    """Пакетные метрики объединения (macro-average)."""
    if len(merged_list) != len(source_list):
        raise ValueError("merged_list и source_list должны быть одинаковой длины.")

    all_metrics = [
        compute_merging_metrics(m, s)
        for m, s in zip(merged_list, source_list)
    ]

    keys = ["coverage", "precision", "f1", "jaccard", "symmetric_containment"]
    macro = {
        k: round(sum(m[k] for m in all_metrics) / len(all_metrics), 4)
        for k in keys
    }
    macro["n_samples"] = len(all_metrics)
    macro["per_sample"] = all_metrics
    return macro
