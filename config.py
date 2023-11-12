import os
class Config:
    SECRET_KEY = "justsomerandomstringstrictlyfordevelopment"
    CORS_ORIGINS = ["http://localhost:3000"]
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_MODEL_NAME = 'yolov8n'
    DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, 'models', DEFAULT_MODEL_NAME)