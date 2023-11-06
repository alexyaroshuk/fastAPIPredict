from fastapi import FastAPI, UploadFile, File, HTTPException, Request
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

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.add_middleware(SessionMiddleware, secret_key=Config.SECRET_KEY)

DEFAULT_MODEL_NAME = Config.DEFAULT_MODEL_NAME
DEFAULT_MODEL_PATH = Config.DEFAULT_MODEL_PATH

loaded_model = None


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
    try:
        model_data = io.BytesIO(await model_file.read())
        model = torch.load(model_data)
        model_name = model_file.filename

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the model to a file in the temporary directory
            model_path = os.path.join(temp_dir, model_name)
            torch.save(model, model_path)

            # Store the model path in the session instead of the model itself
            request.session['model_path'] = model_path
            request.session['model_name'] = model_name

            # Store the model in the global variable
            loaded_model = YOLO(model_path)  # Load the model using YOLO

        return {"message": f"Model {model_name} loaded successfully", "model_name": model_name}
    except RuntimeError as e:  # Correct exception for PyTorch model loading
        logging.error(f"RuntimeError: {str(e)}")
        print(e)
        raise HTTPException(
            status_code=400, detail=f"Error loading model: {str(e)}")
    except Exception as e:
        logging.error(f"Exception: {str(e)}")
        print(e)
        raise HTTPException(
            status_code=500, detail=f"Unexpected error: {str(e)}")


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    global loaded_model
    model_name, model_path = get_model_info(request)

    # If no model has been loaded or if the model has changed, load the model
    if loaded_model is None or model_path != request.session.get('model_path'):
        loaded_model = YOLO(model_path)

    # Use the loaded model for prediction
    model = loaded_model

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
