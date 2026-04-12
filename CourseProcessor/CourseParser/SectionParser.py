import os
import json
from typing import Any, Dict, List, Iterator
from .LessonParser import LessonAnalyzer


class SectionAnalyzer:
    def __init__(self, section_dir: str, knowledge_base_dir: str, course_id: str):
        self.section_dir = section_dir
        self.knowledge_base_dir = knowledge_base_dir 
        self.course_id = course_id

    def iter_lesson_dirs(self) -> Iterator[str]:
        if not os.path.isdir(self.section_dir):
            return
        for name in sorted(os.listdir(self.section_dir)):
            full = os.path.join(self.section_dir, name)
            if os.path.isdir(full) and name.lower().startswith("lesson_"):
                yield full

    def parse(self) -> List[Dict[str, Any]]:
        all_steps = []
        section_jsonl = os.path.join(self.section_dir, "section_steps_texts.jsonl")

        if os.path.exists(section_jsonl):
            try:
                os.remove(section_jsonl)
            except OSError:
                pass

        for lesson_dir in self.iter_lesson_dirs():
            lp = LessonAnalyzer(lesson_dir, self.knowledge_base_dir, self.course_id)
            parsed = lp.parse()
            all_steps.extend(parsed)

            with open(section_jsonl, "a", encoding="utf-8") as sf:
                for rec in parsed:
                    rec_copy = dict(rec)
                    rec_copy["lesson_dir"] = os.path.basename(lesson_dir)
                    sf.write(json.dumps(rec_copy, ensure_ascii=False) + "\n")

        return all_steps
    