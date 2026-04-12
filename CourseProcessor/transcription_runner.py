import os
import re
import json
import time
import asyncio
import httpx
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


DEFAULT_TIMEOUT_SECONDS = 1800
DEFAULT_RETRY_COUNT = 3
DEFAULT_BACKOFF_BASE = 2


@dataclass
class TranscriptionResult:
    step_file: str
    step_id: Any
    video_url: str
    success: bool
    transcript: str = ""
    error: str = ""
    duration_sec: float = 0.0
    backend_url: str = ""


# ─────────────────────────────────────────────
# SCAN
# ─────────────────────────────────────────────

def _pick_best_url(urls: List[Dict[str, Any]]) -> Optional[str]:
    numeric = []
    fallback = []

    for entry in urls:
        q = entry.get("quality")
        u = entry.get("url") or entry.get("src") or entry.get("link")

        if not u:
            continue

        if isinstance(q, str):
            m = re.search(r"(\d+)", q)
            if m:
                numeric.append((int(m.group(1)), u))
                continue

        fallback.append(u)

    if numeric:
        numeric.sort(key=lambda x: x[0])
        for q, u in numeric:
            if q == 360:
                return u
        return numeric[0][1]

    return fallback[-1] if fallback else None


def scan_video_steps(course_dirs: List[str]) -> List[Dict[str, Any]]:
    pending = []

    for course_dir in course_dirs:
        for root, _, files in os.walk(course_dir):
            for fname in files:
                if not fname.startswith("step_"):
                    continue

                path = os.path.join(root, fname)

                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except:
                    continue

                block = data.get("block") or {}
                if isinstance(block, list):
                    block = block[0] if block else {}

                if (block.get("name") or "").lower() != "video":
                    continue

                urls = (block.get("video") or block).get("urls")
                video_url = _pick_best_url(urls or [])

                if not video_url:
                    continue

                if data.get("transcript"):
                    continue

                pending.append({
                    "step_file": path,
                    "step_id": data.get("id"),
                    "video_url": video_url
                })

    print(f"[SCAN] {len(pending)} задач")
    return pending


# ─────────────────────────────────────────────
# BACKEND CALL
# ─────────────────────────────────────────────

async def call_backend(client: httpx.AsyncClient, backend: str, video_url: str):
    resp = await client.post(
        f"{backend}/task",
        json={"task_type": "video", "url": video_url},
    )
    resp.raise_for_status()

    data = resp.json()
    result = data.get("result", {})

    return result.get("transcript"), result.get("segments", [])


# ─────────────────────────────────────────────
# WORKER (САМЫЙ ВАЖНЫЙ)
# ─────────────────────────────────────────────

async def process_task(task, backend, semaphore):
    async with semaphore:
        start = time.time()

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(DEFAULT_TIMEOUT_SECONDS)
        ) as client:

            for attempt in range(DEFAULT_RETRY_COUNT):
                try:
                    transcript, segments = await call_backend(
                        client, backend, task["video_url"]
                    )

                    if not transcript:
                        raise ValueError("empty transcript")

                    _write(task["step_file"], transcript, segments)

                    return TranscriptionResult(
                        step_file=task["step_file"],
                        step_id=task["step_id"],
                        video_url=task["video_url"],
                        success=True,
                        transcript=transcript,
                        duration_sec=time.time() - start,
                        backend_url=backend,
                    )

                except Exception as e:
                    if attempt == DEFAULT_RETRY_COUNT - 1:
                        return TranscriptionResult(
                            step_file=task["step_file"],
                            step_id=task["step_id"],
                            video_url=task["video_url"],
                            success=False,
                            error=str(e),
                            duration_sec=time.time() - start,
                            backend_url=backend,
                        )

                    await asyncio.sleep(DEFAULT_BACKOFF_BASE ** attempt)


# ─────────────────────────────────────────────
# WRITE
# ─────────────────────────────────────────────

def _write(path, transcript, segments):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["transcript"] = transcript
    data["_segments"] = segments

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    os.replace(tmp, path)


async def run_transcription(course_dirs, backends):

    tasks = scan_video_steps(course_dirs)

    if not tasks:
        return []

    # 1 семафор на backend → 1 GPU задача
    semaphores = {
        b: asyncio.Semaphore(1)
        for b in backends
    }

    coroutines = []

    for i, task in enumerate(tasks):
        backend = backends[i % len(backends)]
        coroutines.append(process_task(task, backend, semaphores[backend]))

    results = await asyncio.gather(*coroutines)

    ok = sum(r.success for r in results)
    print(f"[DONE] {ok}/{len(results)}")

    return results