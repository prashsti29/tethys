import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

class Settings(BaseModel):
    # Database Settings
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "voice_assistant")
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5433"))

    # Redis Settings
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))

    # Ollama Settings
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2:1b")

    # Model & Voice Pipeline Settings
    WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "base")
    WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cpu")
    PIPER_MODEL_PATH: str = os.getenv(
        "PIPER_MODEL_PATH",
        str(BASE_DIR / "models" / "piper" / "en_US-lessac-medium.onnx")
    )

    # Latency Instrumentation
    LATENCY_LOGGING_ENABLED: bool = os.getenv("LATENCY_LOGGING_ENABLED", "true").lower() == "true"
    LATENCY_AGGREGATE_INTERVAL: int = int(os.getenv("LATENCY_AGGREGATE_INTERVAL", "10"))  # log P50/P95 every N turns

    # VAD & Turn Detection
    VAD_SILENCE_TIMEOUT_MS: int = int(os.getenv("VAD_SILENCE_TIMEOUT_MS", "1200"))  # higher for medical conversations
    VAD_ADAPTIVE_ENABLED: bool = os.getenv("VAD_ADAPTIVE_ENABLED", "true").lower() == "true"

    # Barge-In & Backchannel
    BACKCHANNEL_DETECTION_ENABLED: bool = os.getenv("BACKCHANNEL_DETECTION_ENABLED", "true").lower() == "true"
    BACKCHANNEL_MAX_DURATION_MS: int = int(os.getenv("BACKCHANNEL_MAX_DURATION_MS", "600"))

    # Safety Gate
    SAFETY_GATE_ENABLED: bool = os.getenv("SAFETY_GATE_ENABLED", "true").lower() == "true"
    SAFETY_GATE_HOLDBACK_TOKENS: int = int(os.getenv("SAFETY_GATE_HOLDBACK_TOKENS", "20"))

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @property
    def REDIS_URL(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/0"

settings = Settings()
