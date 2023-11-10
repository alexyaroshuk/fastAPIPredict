from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from starlette.middleware.sessions import SessionMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import torch
import tempfile
import os
import base64
from config import Config  # Import the Config class
import logging
import shutil

app = FastAPI()

# Define the models directory
MODELS_DIR = 'models'

DISK_DIR = '/disk'

# Ensure the disk directory exists
os.makedirs(DISK_DIR, exist_ok=True)

# Ensure the models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yolo-predict-tester-git-dev-alexyaroshuk.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.add_middleware(SessionMiddleware, secret_key=Config.SECRET_KEY, same_site='none', secure=False)

DEFAULT_MODEL_NAME = Config.DEFAULT_MODEL_NAME
DEFAULT_MODEL_PATH = Config.DEFAULT_MODEL_PATH

loaded_model = None
loaded_model_path = None

def get_request() -> Request:
    return Request(scope={}, receive=None)


def get_model_info(request: Request):
    model_name = request.session.get('model_name', DEFAULT_MODEL_NAME)
    model_path = request.session.get('model_path', DEFAULT_MODEL_PATH)

    # Check if the model file exists
    if not os.path.exists(model_path):
        # If the file does not exist, return the default model name and path
        model_name = DEFAULT_MODEL_NAME
        model_path = DEFAULT_MODEL_PATH

    return model_name, model_path



def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


@app.get("/current_model")
async def current_model(request: Request):
    model_name, _ = get_model_info(request)
    return {"model_used": model_name}


@app.get("/download_model")
async def download_model(request: Request):
    _, model_path = get_model_info(request)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    return FileResponse(model_path, filename=model_path)


@app.post("/upload_model")
async def upload_model(request: Request, model_file: UploadFile = File(...)):
    global loaded_model
    model_data = io.BytesIO(await model_file.read())
    model = torch.load(model_data)
    model_name = model_file.filename

    # Save the model to a file in the disk directory
    try:
        disk_model_path = os.path.join(DISK_DIR, model_name)
        torch.save(model, disk_model_path)
        logging.info(f"Model {model_name} saved to disk directory successfully")
    except Exception as e:
        logging.error(f"Exception when saving to disk directory: {str(e)}")

    # Save the model to a file in the models directory
    try:
        models_model_path = os.path.join(MODELS_DIR, model_name)
        torch.save(model, models_model_path)
        logging.info(f"Model {model_name} saved to models directory successfully")

        # Store the model path in the session instead of the model itself
        request.session['model_path'] = models_model_path
        request.session['model_name'] = model_name

        # Store the model in the global variable
        loaded_model = YOLO(models_model_path)  # Load the model using YOLO
    except RuntimeError as e:  # Correct exception for PyTorch model loading
        logging.error(f"RuntimeError: {str(e)}")
        raise HTTPException(
            status_code=400, detail=f"Error loading model: {str(e)}")
    except Exception as e:
        logging.error(f"Exception when saving to models directory: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Unexpected error: {str(e)}")

    return {"message": f"Model {model_name} loaded successfully", "model_name": model_name}
        
@app.get("/disk_content")
async def disk_content():
    try:
        content = os.listdir(DISK_DIR)
        return {"content": content}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unexpected error: {str(e)}")
        


@app.post("/select_model")
async def select_model(request: Request, model_name: str):
    global loaded_model, loaded_model_path
    model_path = os.path.join(MODELS_DIR, model_name)

    # Check if the model file exists
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")

    # Load the model
    loaded_model = YOLO(model_path)
    loaded_model_path = model_path

    # Store the model path in the session
    request.session['model_path'] = loaded_model_path
    request.session['model_name'] = model_name

    print("Session data after select_model:", request.session)

    print("model selected:", model_name)

    return {"message": f"Model {model_name} selected successfully", "model_name": model_name}




@app.get("/project_structure")
async def project_structure():
    project_structure = {}

    for root, dirs, files in os.walk("."):
        project_structure[root] = {
            "dirs": dirs,
            "files": files
        }

    return project_structure



@app.get("/models")
async def list_models():
    try:
        models = os.listdir(MODELS_DIR)
        return {"models": models}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unexpected error: {str(e)}")


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    global loaded_model, loaded_model_path

    print("Session data before predict:", request.session)

    model_name, model_path = get_model_info(request)

    # If no model has been loaded or if the model has changed, load the model
    if loaded_model is None or loaded_model_path != model_path:
        print(f"Loading model from: {model_path}")  # Debug print
        loaded_model = YOLO(model_path)
        loaded_model_path = model_path
        print(f"Loaded model from: {loaded_model_path}")  # Debug print

    # Use the loaded model for prediction
    model = loaded_model
    
    print("model to use for predict:", loaded_model_path)

    # Rest of the code...

    print("model_path from session:", model_path)

    # Read image file
    image = Image.open(io.BytesIO(await file.read()))

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the image to the temporary directory
        image_path = os.path.join(temp_dir, 'temp.jpg')
        image.save(image_path)

        # Run inference on the image
        results = model(image_path)  # list of Results objects

        # Get the annotated image from the results
        annotated_image = results[0].plot(
            font='Roboto-Regular.ttf', pil=True)

        # Convert the numpy array to a PIL Image
        annotated_image = Image.fromarray(annotated_image)

        # Convert the image to RGB mode
        annotated_image = annotated_image.convert("RGB")

        # Save the annotated image to the temporary directory
        output_path = os.path.join(temp_dir, 'output.jpg')
        annotated_image.save(output_path)

        # Read the output image and return it
        image_base64 = image_to_base64(output_path)

        # Convert each Results object to a dictionary
        results_json = [result.tojson() for result in results]

        print("used model:", model_name)

        return {"image": image_base64, "model_used": model_name, "results": results_json}
