import os


class AppConfig:
    """Общие настройки приложения"""
    
    # ML Backend
    ML_SERVER_URL = os.getenv("ML_SERVER_URL", "http://localhost:8000")
    
    # Gemini
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    
    # Stepik
    STEPIK_CLIENT_ID = os.getenv("STEPIK_CLIENT_ID")
    STEPIK_CLIENT_SECRET = os.getenv("STEPIK_CLIENT_SECRET")
    
    
    # Whisper
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
    WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")

    # Qdrant
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    
    @classmethod
    def validate(cls) -> bool:
        """Проверяет наличие критических настроек"""
        errors = []
        
        if not cls.GEMINI_API_KEY:
            errors.append("GEMINI_API_KEY не установлен")
        
        if not cls.STEPIK_CLIENT_ID or not cls.STEPIK_CLIENT_SECRET:
            errors.append("STEPIK_CLIENT_ID или STEPIK_CLIENT_SECRET не установлены")

        
        if errors:
            print("[Config] ОШИБКИ КОНФИГУРАЦИИ:")
            for err in errors:
                print(f"  ❌ {err}")
            return False
        
        print("[Config] Все критические настройки в порядке")
        return True