import time
from typing import Dict, Any, List, Optional

import requests

from config import AppConfig  # BUG FIX: was missing


class Client:
    """
    Клиент для взаимодействия с ML Backend.

    POST /task с task_type='video' для транскрибации.
    Формат ответа:
        {
            "task_type": "video",
            "result": {
                "transcript": "...",
                "segments": [...]
            }
        }
    """

    ML_BACKEND_URL = AppConfig.ML_SERVER_URL.rstrip("/")
    TASK_ENDPOINT = f"{ML_BACKEND_URL}/task"
    HEALTH_ENDPOINT = f"{ML_BACKEND_URL}/health"

    MAX_TRANSCRIBE_WORKERS = 1

    CONNECT_TIMEOUT_SECONDS = 5
    REQUEST_TIMEOUT_SECONDS = 1800  # до 30 минут на длинное видео

    RETRY_COUNT = 3
    RETRY_BACKOFF_BASE = 2.0

    @classmethod
    def _get_session(cls) -> requests.Session:
        return requests.Session()

    @classmethod
    def _parse_transcribe_response(cls, data: Any) -> Dict[str, Any]:
        """
        Приводит ответ backend'а к единому виду:
        {"text": "...", "segments": [...]}
        """
        if not isinstance(data, dict):
            return {"text": "", "segments": []}

        result = data.get("result")
        if isinstance(result, dict):
            transcript = result.get("transcript") or result.get("text") or ""
            segments = result.get("segments") or []
        else:
            transcript = data.get("transcript") or data.get("text") or ""
            segments = data.get("segments") or []

        if not isinstance(transcript, str):
            transcript = str(transcript) if transcript is not None else ""

        if not isinstance(segments, list):
            segments = []

        return {
            "text": transcript.strip(),
            "segments": segments,
        }

    @classmethod
    def transcribe(cls, video_url: str, step_id: int = None) -> Dict[str, Any]:
        """
        Транскрибирует одно видео через POST /task.

        Returns:
            {"text": "полная транскрипция", "segments": [...]}
        """
        payload = {
            "task_type": "video",
            "url": video_url,
        }

        session = cls._get_session()
        last_error: Optional[Exception] = None

        try:
            for attempt in range(1, cls.RETRY_COUNT + 1):
                try:
                    response = session.post(
                        cls.TASK_ENDPOINT,
                        json=payload,
                        timeout=(cls.CONNECT_TIMEOUT_SECONDS, cls.REQUEST_TIMEOUT_SECONDS),
                    )
                    response.raise_for_status()

                    data = response.json()
                    parsed = cls._parse_transcribe_response(data)

                    if not parsed["text"]:
                        raise ValueError("Empty transcript in backend response")

                    return parsed

                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                    last_error = e
                    if attempt >= cls.RETRY_COUNT:
                        break
                    sleep_for = cls.RETRY_BACKOFF_BASE ** (attempt - 1)
                    print(
                        f"   [Transcribe Retry] Step {step_id}: "
                        f"попытка {attempt}/{cls.RETRY_COUNT} ({e}), "
                        f"жду {sleep_for:.1f}s"
                    )
                    time.sleep(sleep_for)

                except (requests.exceptions.HTTPError, ValueError, KeyError, TypeError) as e:
                    last_error = e
                    if attempt >= cls.RETRY_COUNT:
                        break
                    sleep_for = cls.RETRY_BACKOFF_BASE ** (attempt - 1)
                    print(
                        f"   [Transcribe Retry] Step {step_id}: "
                        f"попытка {attempt}/{cls.RETRY_COUNT} ({e}), "
                        f"жду {sleep_for:.1f}s"
                    )
                    time.sleep(sleep_for)

                except Exception as e:
                    last_error = e
                    break

        finally:
            try:
                session.close()
            except Exception:
                pass

        print(f"   [Transcribe Error] Step {step_id}: {last_error}")
        return {"text": "", "segments": []}

    @classmethod
    def transcribe_batch(cls, videos: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        Последовательная транскрибация списка видео.

        Args:
            videos: [{"step_id": 123, "video_url": "..."}, ...]

        Returns:
            {step_id: {"text": "...", "segments": [...]}, ...}
        """
        if not videos:
            return {}

        print(f"   [Transcribe Batch] {len(videos)} видео последовательно...")
        results: Dict[int, Dict[str, Any]] = {}

        for idx, item in enumerate(videos, start=1):
            step_id = item.get("step_id")
            video_url = item.get("video_url")

            if not video_url:
                print(f"      ❌ Step {step_id}: пустой video_url")
                results[step_id] = {"text": "", "segments": []}
                continue

            try:
                result = cls.transcribe(video_url, step_id)
                results[step_id] = result

                if result.get("text"):
                    print(f"      ✅ Step {step_id}: {len(result['text'])} символов")
                else:
                    print(f"      ⚠️  Step {step_id}: пустая транскрипция")

            except Exception as e:
                print(f"      ❌ Step {step_id}: {e}")
                results[step_id] = {"text": "", "segments": []}

            if idx % 10 == 0 or idx == len(videos):
                print(f"      [Progress] {idx}/{len(videos)}")

        return results

    @classmethod
    def health(cls) -> bool:
        """Проверка доступности backend."""
        session = cls._get_session()
        try:
            response = session.get(cls.HEALTH_ENDPOINT, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
        finally:
            try:
                session.close()
            except Exception:
                pass