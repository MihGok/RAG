
from typing import List, Dict

class LLMPrompts:
    @staticmethod
    def build_course_ranking_prompt(topic: str, courses: List[Dict]) -> str:
        """Генерирует промпт для ранжирования курсов."""
        course_list = "\n".join([f"{i+1}. {c['title']} (ID: {c['id']})" for i, c in enumerate(courses)])
        prompt = (
            f"Тема: {topic}\n\n"
            f"Вот список курсов по данной теме:\n{course_list}\n\n"
            "Пожалуйста, оцени каждый курс по шкале от 1 до 10, где 10 - лучший курс для изучения данной темы. "
            "Учитывай качество контента, структуру курса, отзывы студентов и актуальность материала. "
            "Ответь в формате JSON, где ключ - ID курса, а значение - его оценка."
        )
        return prompt