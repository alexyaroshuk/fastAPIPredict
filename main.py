from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends, Form
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
from typing import Optional
import uuid
from fastapi.staticfiles import StaticFiles

# Define the directory for shared images
SHARED_IMAGE_DIR = 'disk/shared_images'
os.makedirs(SHARED_IMAGE_DIR, exist_ok=True)

app = FastAPI()

# Mount the shared images directory as static files
app.mount("/shared_images", StaticFiles(directory=SHARED_IMAGE_DIR), name="shared_images")

# Define the models directory
MODELS_DIR = 'models'

DISK_MODELS_DIR = 'disk/models'
DISK_USERDATA_DIR = 'disk/userdata'

# Define the base directory for user images
USER_IMAGE_BASE_DIR = 'disk/userdata/images'


# Ensure the disk directory exists
os.makedirs(DISK_MODELS_DIR, exist_ok=True)

# Ensure the models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

app.add_middleware(SessionMiddleware, secret_key=Config.SECRET_KEY, max_age=3600, same_site="none", https_only=True)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

DEFAULT_MODEL_NAME = Config.DEFAULT_MODEL_NAME
DEFAULT_MODEL_DIR = os.path.join(DISK_MODELS_DIR, DEFAULT_MODEL_NAME)
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, f"{DEFAULT_MODEL_NAME}.pt")

# Global dictionary to store model info
model_info_dict = {}

loaded_model = None
loaded_model_path = None

async def get_request():
    return Request(scope={}, receive=None)

def get_or_set_session_id(request: Request):
    # If the session ID already exists, return it
    if 'id' in request.session:
        return request.session['id']
    
    # Otherwise, generate a new session ID, store it in the session, and return it
    session_id = str(uuid.uuid4())
    request.session['id'] = session_id
    return session_id

def get_model_info(request: Request = Depends(get_request)):
    model_name = request.session.get('model_name', DEFAULT_MODEL_NAME)
    model_path = request.session.get('model_path', DEFAULT_MODEL_PATH)

    # Check if the model file exists
    if not os.path.exists(model_path):
        print('files do not exist')
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
    print("Session data before current_model:", request.session)
    return {"model_used": model_name}

@app.get("/download_model")
async def download_model(request: Request):
    _, model_path = get_model_info(request)
    print("Session data before download_model:", request.session)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    return FileResponse(model_path, filename=model_path)

def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    try:
        return YOLO(model_path)
    except RuntimeError as e:
        logging.error(f"RuntimeError: {str(e)}")
        raise HTTPException(
            status_code=400, detail=f"Error loading model: {str(e)}")
    except Exception as e:
        logging.error(f"Exception when loading model: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Unexpected error: {str(e)}")

def save_model(model, model_path: str):
    try:
        torch.save(model, model_path)
        logging.info(f"Model saved to {model_path} successfully")
    except Exception as e:
        logging.error(f"Exception when saving model: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/upload_model")
async def upload_model(request: Request, description: Optional[str] = Form(None), model_file: UploadFile = File(...), photo: Optional[UploadFile] = File(None)):
    global loaded_model
    model_data = io.BytesIO(await model_file.read())
    model = torch.load(model_data)
    model_name, _ = os.path.splitext(model_file.filename)  # Remove the .pt extension
    print("desc:", description)

    # Create a separate folder for the model
    model_dir = os.path.join(DISK_MODELS_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Save the model to a file in the model directory
    model_path = os.path.join(model_dir, f"{model_name}.pt")
    save_model(model, model_path)

    # Save the description to a file in the model directory
    if description is not None:
        with open(os.path.join(model_dir, "description.txt"), "w", encoding='utf-8') as f:
            f.write(description)

    # Save the photo to a file in the model directory
    if photo is not None:
        photo_data = io.BytesIO(await photo.read())
        with open(os.path.join(model_dir, "photo.jpg"), "wb") as f:
            f.write(photo_data.read())

    # Store the model path in the session instead of the model itself
    request.session['model_path'] = model_path
    request.session['model_name'] = model_name

    # Get or set the session ID
    session_id = get_or_set_session_id(request)
    model_info_dict[session_id] = {'model_path': model_path, 'model_name': model_name}

    # Load the model using YOLO
    loaded_model = load_model(model_path)

    return {"message": f"Model {model_name} loaded successfully", "model_name": model_name}

@app.get("/models")
async def list_models():
    try:
        model_dirs = os.listdir(DISK_MODELS_DIR)
        print(model_dirs)
        return {"models": model_dirs}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/model_info/{model_name}")
async def model_info(model_name: str):
    # Create the path to the model directory
    model_dir = os.path.join(DISK_MODELS_DIR, model_name)
    model_dir = model_dir.replace("\\", "/")  # Replace backslashes with forward slashes
    print("model dir", model_dir, model_name)

    # Check if the model directory exists
    if not os.path.exists(model_dir):
        raise HTTPException(status_code=404, detail="Model not found")

    # Read the model file, description, and photo
    model_path = os.path.join(model_dir, f"{model_name}.pt")
    description_path = os.path.join(model_dir, "description.txt")
    photo_path = os.path.join(model_dir, "photo.jpg")

    # Check if the model file exists
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")

    # Check if the description exists
    description = None
    if os.path.exists(description_path):
        with open(description_path, "r", encoding='utf-8') as f:
            description = f.read()

    # Check if the photo exists
    photo_url = None
    if os.path.exists(photo_path):
        photo_url = f"/models/{model_name}/photo.jpg"

    return {"model_path": model_path, "description": description, "photo_url": photo_url}

@app.get("/models/{model_name}/photo.jpg")
async def get_model_photo(model_name: str):
    # Construct the path to the photo
    photo_path = os.path.join(DISK_MODELS_DIR, model_name, "photo.jpg")

    # Check if the photo file exists
    if not os.path.exists(photo_path):
        raise HTTPException(status_code=404, detail="Photo not found")

    # Return the photo file
    return FileResponse(photo_path, media_type="image/jpeg")

@app.post("/select_model")
async def select_model(request: Request, model_name: str):
    global loaded_model, loaded_model_path
    model_name = model_name.replace("\\", "/")  # Replace backslashes with forward slashes
    if model_name.startswith('models/'):
        model_name = model_name[len('models/'):]  # Remove 'models/' from the start of model_name

    # Check if the model file exists
    """ if not os.path.exists(model_path):
        print("if block")
        # If the model file doesn't exist, try using model_name as a relative path
        model_path = f"{model_name}.pt" """

    model_path = os.path.join(DISK_MODELS_DIR, model_name, f"{model_name}.pt")
    # Construct the model path
    print("path is" ,model_path)

    model_path = model_path.replace("\\", "/")
    print("path is2" ,model_path)
    # Load the model
    loaded_model = load_model(model_path)
    loaded_model_path = model_path

    # Get or set the session ID
    session_id = get_or_set_session_id(request)

    # Store the model path and name in the session
    request.session['model_path'] = model_path
    request.session['model_name'] = model_name

    # Store the model path and name in the global dictionary
    model_info_dict[session_id] = {'model_path': loaded_model_path, 'model_name': model_name}

    return {"message": f"Model {model_name} selected successfully", "model_name": model_name, "session_id": session_id}

@app.get("/disk_content")
async def disk_content():
    try:
        content = os.listdir(DISK_DIR)
        return {"content": content}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unexpected error: {str(e)}")
        
    return {"message": f"Model {model_name} selected successfully", "model_name": model_name, "session_id": session_id}

@app.get("/project_structure")
async def project_structure():
    project_structure = {}

    for root, dirs, files in os.walk("."):
        project_structure[root] = {
            "dirs": dirs,
            "files": files
        }

    return project_structure


@app.get("/shared_images")
async def list_shared_images():
    try:
        images = os.listdir(SHARED_IMAGE_DIR)
        return {"images": images}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unexpected error: {str(e)}")
        
""" @app.get("/user_images")
async def list_user_images(request: Request):
    # Get the session ID from the request
    session_id = request.session.get('id')

    # If the session ID does not exist, return an error
    if session_id is None:
        raise HTTPException(status_code=400, detail="Session ID not found")

    # Create a unique directory for the user within the base directory
    user_image_dir = os.path.join(USER_IMAGE_BASE_DIR, session_id)

    # Check if the user's directory exists
    if not os.path.exists(user_image_dir):
        return {"images": []}

    try:
        images = os.listdir(user_image_dir)
        return {"images": images}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unexpected error: {str(e)}")
        
@app.get("/user_images/{session_id}/{image_name}")
async def download_user_image(session_id: str, image_name: str):
    user_image_dir = os.path.join(USER_IMAGE_BASE_DIR, session_id)
    image_path = os.path.join(user_image_dir, image_name)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path) """

@app.get("/models")
async def list_models():
    try:
        models = os.listdir(DISK_MODELS_DIR)
        return {"models": models}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    global loaded_model, loaded_model_path

    # Get or set the session ID
    session_id = get_or_set_session_id(request)

    # Retrieve the model name and path from the session
    model_name, model_path = get_model_info(request)

    # If no model has been loaded or if the model has changed, load the model
    if loaded_model is None or loaded_model_path != model_path:
        loaded_model = load_model(model_path)
        loaded_model_path = model_path

    # Use the loaded model for pre334eeeeeeeekolidiction
    model = loaded_model

    # Read image file
    image = Image.open(io.BytesIO(await file.read()))

    # Convert the image to RGB mode
    image = image.convert("RGB")

    # Create a unique directory for the user within the base directory
    user_image_dir = os.path.join(USER_IMAGE_BASE_DIR, session_id)
    os.makedirs(user_image_dir, exist_ok=True)

    # Save the original image to the user's directory
    user_image_path = os.path.join(user_image_dir, file.filename)
    image.save(user_image_path)

    # Run inference on the image
    results = model(user_image_path)  # list of Results objects

    # Get the annotated image from the results
    annotated_image = results[0].plot(
        font='Roboto-Regular.ttf', pil=True)

    # Convert the numpy array to a PIL Image
    annotated_image = Image.fromarray(annotated_image)

    # Convert the image to RGB mode
    annotated_image = annotated_image.convert("RGB")

    # Create a unique directory for the annotated images within the base directory
    annotated_image_dir = os.path.join(USER_IMAGE_BASE_DIR, session_id, 'annotated_images')
    os.makedirs(annotated_image_dir, exist_ok=True)

    # Save the annotated image to the annotated images directory
    annotated_image_path = os.path.join(annotated_image_dir, f"annotated_{file.filename}")
    annotated_image.save(annotated_image_path)

    # Read the output image and return it
    image_base64 = image_to_base64(annotated_image_path)

    # Convert each Results object to a dictionary
    results_json = [result.tojson() for result in results]

    print("used model:", model_name)

    return {"image": image_base64, "model_used": model_name, "results": results_json}
