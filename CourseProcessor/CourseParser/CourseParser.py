import os
import re
import json
from typing import Any, Dict, List, Iterator
from .SectionParser import SectionAnalyzer


class CourseAnalyzer:
    def __init__(self, course_dir: str, search_query: str):
        self.course_dir = os.path.abspath(course_dir)
        self.search_query = search_query

        self.knowledge_base_dir = os.path.abspath(
            os.path.join("knowledge_base", search_query)
        )
        os.makedirs(self.knowledge_base_dir, exist_ok=True)
        self.course_id = self._extract_course_id(os.path.basename(self.course_dir))

    @staticmethod
    def _extract_course_id(dir_name: str) -> str:
        """
        Из "Course_12345_Название курса" извлекает "12345".
        Если парсинг не удается — возвращает "unknown", чтобы не ломаться.
        """
        match = re.match(r'^Course_(\d+)', dir_name, re.IGNORECASE)
        return match.group(1) if match else "unknown"

    def iter_section_dirs(self) -> Iterator[str]:
        if not os.path.isdir(self.course_dir):
            return
        for name in sorted(os.listdir(self.course_dir)):
            full = os.path.join(self.course_dir, name)
            if os.path.isdir(full) and name.lower().startswith("section_"):
                yield full

    def parse(self) -> List[Dict[str, Any]]:
        all_steps = []
        course_jsonl = os.path.join(self.course_dir, "course_steps_texts.jsonl")

        if os.path.exists(course_jsonl):
            try:
                os.remove(course_jsonl)
            except OSError:
                pass

        print(f"[CourseParser] Knowledge Base dir: {self.knowledge_base_dir}")
        print(f"[CourseParser] course_id: {self.course_id}")

        for section_dir in self.iter_section_dirs():
            sp = SectionAnalyzer(section_dir, self.knowledge_base_dir, self.course_id)
            parsed = sp.parse()

            with open(course_jsonl, "a", encoding="utf-8") as cf:
                for rec in parsed:
                    rec_copy = dict(rec)
                    rec_copy["section_dir"] = os.path.basename(section_dir)
                    all_steps.append(rec_copy)
                    cf.write(json.dumps(rec_copy, ensure_ascii=False) + "\n")

        return all_steps