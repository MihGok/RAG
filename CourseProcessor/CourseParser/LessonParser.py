import os
import re
import json
from typing import List, Dict, Any, Iterator

from CourseProcessor.client_api import Client
from CourseProcessor.CourseParser.StepParser import StepAnalyzer


class LessonAnalyzer:
    STEP_FILENAME_PREFIX = "step_"
    STEP_FILENAME_SUFFIX = ".json"

    def __init__(self, lesson_dir: str, knowledge_base_dir: str, course_id: str = "unknown"):
        """
        Args:
            lesson_dir:          Путь к папке урока (содержит step_*.json файлы).
            knowledge_base_dir:  Куда сохранять результаты парсинга.
            course_id:           ID курса (для метаданных).
        """
        self.lesson_dir = lesson_dir
        self.knowledge_base_dir = knowledge_base_dir
        self.course_id = course_id

    def iter_step_files(self) -> Iterator[str]:
        if not os.path.isdir(self.lesson_dir):
            return
        for fname in sorted(os.listdir(self.lesson_dir)):
            if fname.startswith(self.STEP_FILENAME_PREFIX) and fname.endswith(self.STEP_FILENAME_SUFFIX):
                yield os.path.join(self.lesson_dir, fname)

    def _clean_lesson_title(self, dir_name: str) -> str:
        match = re.search(r'^Lesson_\d+_(.+)$', dir_name, re.IGNORECASE)
        clean_name = match.group(1).strip() if match else dir_name.replace('_', ' ').strip()
        return re.sub(r'[<>:"/\\|?*]', '', clean_name).strip()

    def _save_lesson_content(self, all_parsed_steps: List[Dict], lesson_name: str):
        """Сохраняет весь урок (текст + транскрипцию) в один файл content.txt."""
        lesson_dir = os.path.join(self.knowledge_base_dir, lesson_name)
        os.makedirs(lesson_dir, exist_ok=True)
        filepath = os.path.join(lesson_dir, "content.txt")

        parts = [f"LESSON: {lesson_name}", "=" * 50]

        for step in all_parsed_steps:
            parts.append(f"\nSTEP ID: {step['step_id']}")
            if step.get('update_date'):
                parts.append(f"UPDATED: {step['update_date']}")
            parts.append("-" * 20)

            if step.get("text"):
                parts.append(step["text"])

            if step.get("transcript"):
                parts.append("\n[TRANSCRIPT]:")
                parts.append(step["transcript"])

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("\n".join(parts))
            print(f"   [KB] Сохранен текст урока: {filepath}")
        except Exception as e:
            print(f"   [KB Error] Не удалось сохранить {filepath}: {e}")

    def parse(self) -> List[Dict[str, Any]]:
        """
        Парсинг урока из legacy-структуры (step_*.json файлы).
        Используется CourseParser для анализа уже скачанных данных.
        """
        parsed_steps = []
        videos_to_transcribe = []
        step_file_map: Dict[int, str] = {}

        raw_lesson_dir_name = os.path.basename(self.lesson_dir)
        clean_name = self._clean_lesson_title(raw_lesson_dir_name)

        print(f"\n[Lesson] Обработка: {clean_name}")

        # ЭТАП 1: Парсинг всех шагов и сбор видео
        for step_file in self.iter_step_files():
            try:
                with open(step_file, "r", encoding="utf-8") as f:
                    raw_step = json.load(f)
            except Exception as e:
                print(f"   [Error] Не удалось прочитать {step_file}: {e}")
                continue

            parsed = StepAnalyzer.parse_step_dict(raw_step, os.path.basename(step_file))
            if not parsed:
                continue

            transcript_text = raw_step.get("transcript", "")

            if parsed.get("video_url") and not transcript_text:
                videos_to_transcribe.append({
                    "step_id": parsed["step_id"],
                    "video_url": parsed["video_url"],
                })
                step_file_map[parsed["step_id"]] = step_file
            elif transcript_text:
                parsed["transcript"] = transcript_text

            parsed_steps.append(parsed)

        # ЭТАП 2: Транскрибация видео
        if videos_to_transcribe:
            print(f"   [Transcribe] {len(videos_to_transcribe)} видео для транскрибации...")
            transcription_results = Client.transcribe_batch(videos_to_transcribe)

            # ЭТАП 3: Сохранение транскрипций
            for step_id, trans_result in transcription_results.items():
                transcript_text = trans_result.get("text", "")

                if transcript_text:
                    for p in parsed_steps:
                        if p["step_id"] == step_id:
                            p["transcript"] = transcript_text
                            break

                    step_file = step_file_map.get(step_id)
                    if step_file:
                        try:
                            with open(step_file, "r", encoding="utf-8") as f:
                                raw_step = json.load(f)

                            raw_step["transcript"] = transcript_text
                            raw_step["_generated_transcript"] = transcript_text
                            raw_step["_segments"] = trans_result.get("segments", [])

                            with open(step_file, "w", encoding="utf-8") as f:
                                json.dump(raw_step, f, ensure_ascii=False, indent=2)

                        except Exception as e:
                            print(f"   [Save Error] Step {step_id}: {e}")

        # ЭТАП 4: Сохранение
        self._save_lesson_content(parsed_steps, clean_name)

        return parsed_steps