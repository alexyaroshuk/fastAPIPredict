import os
class Config:
    SECRET_KEY = "justsomerandomstringstrictlyfordevelopment"
    CORS_ORIGINS = ["https://yolo-predict-tester-git-dev-alexyaroshuk.vercel.app"]
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_MODEL_NAME = 'ds1_2_best.pt'
    DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, 'models', DEFAULT_MODEL_NAME)