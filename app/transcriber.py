"""
transcriber.py
──────────────
Скачивание видео по ссылке и транскрибация с помощью faster-whisper.

Функции:
    download_audio(url, download_dir) -> Path
    transcribe(audio_path)            -> dict
    transcribe_from_url(url)          -> dict   (скачать + транскрибировать)

Модель загружается один раз (синглтон).
"""

import os
import logging
import tempfile
from pathlib import Path

import yt_dlp
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL", "medium")
DOWNLOAD_DIR: Path      = Path(os.getenv("DOWNLOAD_DIR", "/tmp/downloads"))
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

_whisper_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        logger.info("Загрузка Whisper '%s' на GPU...", WHISPER_MODEL_SIZE)
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device="cuda",
            compute_type="float16",
        )
        logger.info("Whisper загружен.")
    return _whisper_model


def download_audio(url: str, download_dir: Path | None = None) -> Path:
    """Скачать аудиодорожку из видео по URL, вернуть Path к .mp3."""
    out_dir = download_dir or DOWNLOAD_DIR

    with tempfile.NamedTemporaryFile(dir=out_dir, suffix=".tmp", delete=False) as f:
        template = f.name.replace(".tmp", "")

    ydl_opts = {
        "format":      "bestaudio/best",
        "outtmpl":     f"{template}.%(ext)s",
        "quiet":       True,
        "no_warnings": True,
        "postprocessors": [{
            "key":              "FFmpegExtractAudio",
            "preferredcodec":   "mp3",
            "preferredquality": "128",
        }],
    }

    logger.info("Скачивание аудио: %s", url)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    audio_path = Path(f"{template}.mp3")
    if not audio_path.exists():
        raise FileNotFoundError(f"Аудиофайл не найден: {audio_path}")

    logger.info("Аудио сохранено: %s (%.1f MB)", audio_path, audio_path.stat().st_size / 1e6)
    return audio_path


def transcribe(audio_path: Path | str) -> dict:
    """
    Транскрибировать аудиофайл.

    Returns:
        {text, language, duration_seconds, segments}
    """
    model = _get_model()
    audio_path = Path(audio_path)

    logger.info("Транскрибация: %s", audio_path.name)
    segments_iter, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        language=None,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        word_timestamps=False,
    )

    segments, full_text_parts = [], []
    for seg in segments_iter:
        segments.append({
            "start": round(seg.start, 2),
            "end":   round(seg.end,   2),
            "text":  seg.text.strip(),
        })
        full_text_parts.append(seg.text.strip())

    result = {
        "text":             " ".join(full_text_parts),
        "language":         info.language,
        "duration_seconds": round(info.duration, 2),
        "segments":         segments,
    }
    logger.info(
        "Транскрибация завершена. Язык: %s, длина: %.1fs",
        info.language, info.duration,
    )
    return result


def transcribe_from_url(url: str) -> dict:
    """Скачать видео по URL и транскрибировать. Временный файл удаляется."""
    audio_path = download_audio(url)
    try:
        result = transcribe(audio_path)
        result["source_url"] = url
        return result
    finally:
        if audio_path.exists():
            audio_path.unlink()
