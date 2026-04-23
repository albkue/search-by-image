"""ML Search Service Configuration."""
import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings for ML Search Service."""
    
    # Service
    SERVICE_NAME: str = "ml-search-service"
    SERVICE_PORT: int = 8001
    
    # Main API
    MAIN_API_URL: str = os.getenv("MAIN_API_URL", "http://localhost:8000")
    
    # Model paths
    YOLO_MODEL: str = os.getenv("YOLO_MODEL", "yolov8n.pt")  # nano for speed, use 's' or 'm' for accuracy
    CLIP_MODEL: str = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
    
    # Confidence thresholds
    YOLO_CONFIDENCE_THRESHOLD: float = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.5"))
    OCR_CONFIDENCE_THRESHOLD: float = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.6"))
    
    # FAISS
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
    EMBEDDING_DIMENSION: int = 512
    
    # GPU settings
    USE_GPU: bool = os.getenv("USE_GPU", "false").lower() == "true"
    
    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
