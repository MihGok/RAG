import os
import re
import json
import shutil
from typing import List, Dict, Any, Iterator

from services.storage_service import StorageService
from services.LLM_Service.llm_service import GeminiService
from services.config import AppConfig
from CourseProcessor.client_api import Client
from CourseProcessor.CourseParser.StepParser import StepAnalyzer

class LessonAnalyzer:
    STEP_FILENAME_PREFIX = "step_"
    STEP_FILENAME_SUFFIX = ".json"

    def __init__(self, lesson_dir: str, knowledge_base_dir: str):
        self.lesson_dir = lesson_dir
        self.knowledge_base_dir = knowledge_base_dir
        self.storage = StorageService()
        self.llm_service = GeminiService()
        self.temp_base_dir = AppConfig.TEMP_DIR
        os.makedirs(self.temp_base_dir, exist_ok=True)

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
        """Сохраняет весь урок (текст шагов + транскрипцию) в один файл content.txt"""
        lesson_dir = os.path.join(self.knowledge_base_dir, lesson_name)
        os.makedirs(lesson_dir, exist_ok=True)
        filepath = os.path.join(lesson_dir, "content.txt")

        parts = [f"LESSON: {lesson_name}", "="*50]

        for step in all_parsed_steps:
            parts.append(f"\nSTEP ID: {step['step_id']}")
            if step.get('update_date'):
                parts.append(f"UPDATED: {step['update_date']}")
            parts.append("-" * 20)
            
            # Основной текст шага (если есть)
            if step.get("text"):
                parts.append(step["text"])
            
            # Транскрипция видео (если есть)
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
        Главный метод парсинга урока.
        УЛУЧШЕНО: Параллельная транскрибация всех видео в уроке.
        """
        parsed_steps = []
        videos_to_transcribe = []
        step_file_map = {}  # {step_id: step_file_path}
        
        raw_lesson_dir_name = os.path.basename(self.lesson_dir)
        clean_name = self._clean_lesson_title(raw_lesson_dir_name)
        
        print(f"\n[Lesson] Обработка: {clean_name}")

        # ЭТАП 1: Парсинг всех шагов и сбор видео для транскрибации
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
            
            # Проверяем наличие транскрипции
            transcript_text = raw_step.get("transcript", "")
            
            if parsed.get("video_url") and not transcript_text:
                # Нужна транскрибация
                videos_to_transcribe.append({
                    "step_id": parsed["step_id"],
                    "video_url": parsed["video_url"]
                })
                step_file_map[parsed["step_id"]] = step_file
            elif transcript_text:
                # Транскрипция уже есть
                parsed["transcript"] = transcript_text
            
            parsed_steps.append(parsed)

        # ЭТАП 2: Параллельная транскрибация всех видео
        if videos_to_transcribe:
            print(f"   [Transcribe] Найдено {len(videos_to_transcribe)} видео для транскрибации...")
            
            transcription_results = Client.transcribe_batch(videos_to_transcribe)
            
            # ЭТАП 3: Сохранение результатов транскрибации в файлы
            for step_id, trans_result in transcription_results.items():
                transcript_text = trans_result.get("text", "")
                
                if transcript_text:
                    # Обновляем parsed_steps
                    for parsed in parsed_steps:
                        if parsed["step_id"] == step_id:
                            parsed["transcript"] = transcript_text
                            break
                    
                    # Сохраняем в JSON файл
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

        # ЭТАП 4: Сохранение контента урока
        self._save_lesson_content(parsed_steps, clean_name)
        
        return parsed_steps