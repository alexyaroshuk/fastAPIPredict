"""
This module is used for YOLO object detection. It includes functionalities for loading models, 
processing images and videos, and running predictions.
"""

import io
import tempfile
import os
import base64
from typing import Optional
import uuid
import logging
import glob
import gc
import torch
import cv2
from shapely.geometry import Polygon
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from fastapi.staticfiles import StaticFiles
from moviepy.editor import VideoFileClip
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from starlette.middleware.sessions import SessionMiddleware
import json
from collections import defaultdict
from config import Config
import psutil

# Import the Config class


# Define the directory for shared images
SHARED_IMAGE_DIR = 'disk/shared_images'
os.makedirs(SHARED_IMAGE_DIR, exist_ok=True)

SHARED_THUMBNAILS_DIR = 'disk/shared_thumbnails'
os.makedirs(SHARED_THUMBNAILS_DIR, exist_ok=True)

app = FastAPI()

# Mount the shared images directory as static files
app.mount("/shared_images", StaticFiles(directory=SHARED_IMAGE_DIR),
          name="shared_images")
app.mount("/shared_thumbnails",
          StaticFiles(directory=SHARED_THUMBNAILS_DIR), name="shared_thumbnails")
# Define the models directory
MODELS_DIR = 'models'

DISK_DIR = './disk'
DISK_MODELS_DIR = 'disk/models'
DISK_USERDATA_DIR = 'disk/userdata'

# Define the base directory for user images
USER_IMAGE_BASE_DIR = 'disk/userdata/images'


# Ensure the disk directory exists
os.makedirs(DISK_MODELS_DIR, exist_ok=True)

# Ensure the models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

app.add_middleware(SessionMiddleware, secret_key=Config.SECRET_KEY,
                   max_age=3600, same_site="none", https_only=True)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.middleware("http")
async def catch_memory_error(request: Request, call_next):
    try:
        return await call_next(request)
    except MemoryError:
        return JSONResponse(status_code=500, content={"error": "Insufficient memory to complete the operation"})

DEFAULT_MODEL_NAME = Config.DEFAULT_MODEL_NAME
DEFAULT_MODEL_DIR = os.path.join(DISK_MODELS_DIR, DEFAULT_MODEL_NAME)
DEFAULT_MODEL_PATH = os.path.join(
    DEFAULT_MODEL_DIR, f"{DEFAULT_MODEL_NAME}.pt")

# Global dictionary to store model info
model_info_dict = {}

loaded_model = None
loaded_model_path = None

# Generate thumbnails for video files


def generate_video_thumbnails():
    files = os.listdir(SHARED_IMAGE_DIR)
    video_files = [file for file in files if file.endswith('.mp4')]

    for video_file in video_files:
        thumbnail_path = os.path.join(
            SHARED_THUMBNAILS_DIR, f'{video_file}_thumbnail.jpg')

        # Check if thumbnail already exists
        if not os.path.exists(thumbnail_path):
            clip = VideoFileClip(os.path.join(SHARED_IMAGE_DIR, video_file))
            clip.save_frame(thumbnail_path, t=0)  # save frame at 0 seconds

# Get the memory usage in percentage


def get_memory_usage():
    return psutil.virtual_memory().percent


# Call the function when the server starts
generate_video_thumbnails()


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


# Convert the PIL Image to a base64 string
def image_to_base64_for_video(pil_image):
    byte_arr = io.BytesIO()
    pil_image.save(byte_arr, format='JPEG')
    encoded_image = base64.encodebytes(byte_arr.getvalue()).decode('ascii')
    return encoded_image


def image_to_base64_for_image(image_path: str) -> str:
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
    model_name, _ = os.path.splitext(
        model_file.filename)  # Remove the .pt extension
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
    model_info_dict[session_id] = {
        'model_path': model_path, 'model_name': model_name}

    # Load the model using YOLO
    loaded_model = load_model(model_path)

    return {"message": f"Model {model_name} loaded successfully", "model_name": model_name}


@app.get("/model_info/{model_name}")
async def model_info(model_name: str):
    # Create the path to the model directory
    model_dir = os.path.join(DISK_MODELS_DIR, model_name)
    # Replace backslashes with forward slashes
    model_dir = model_dir.replace("\\", "/")
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
    # Replace backslashes with forward slashes
    model_name = model_name.replace("\\", "/")
    if model_name.startswith('models/'):
        # Remove 'models/' from the start of model_name
        model_name = model_name[len('models/'):]

    model_path = os.path.join(DISK_MODELS_DIR, model_name, f"{model_name}.pt")
    # Construct the model path
    print("path is", model_path)

    model_path = model_path.replace("\\", "/")
    print("path is2", model_path)
    # Load the model
    loaded_model = load_model(model_path)
    loaded_model_path = model_path

    # Get or set the session ID
    session_id = get_or_set_session_id(request)

    # Store the model path and name in the session
    request.session['model_path'] = model_path
    request.session['model_name'] = model_name

    # Store the model path and name in the global dictionary
    model_info_dict[session_id] = {
        'model_path': loaded_model_path, 'model_name': model_name}

    return {"message": f"Model {model_name} selected successfully", "model_name": model_name, "session_id": session_id}


""" @app.get("/disk_content")
async def disk_content():
    try:
        content = os.listdir(DISK_DIR)
        return {"content": content}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unexpected error: {str(e)}")

    return {"message": f"Model {model_name} selected successfully", "model_name": model_name, "session_id": session_id} """


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
        files = os.listdir(SHARED_IMAGE_DIR)
        images = []
        for file in files:
            # Skip thumbnail images and already resized images
            if file.endswith('_thumbnail.jpg') or file.startswith('resized_'):
                continue
            if file.endswith('.mp4'):
                # If the file is a video, add the filename of the thumbnail image
                thumbnail_filename = f'{file}_thumbnail.jpg'
                images.append({
                    'filename': file,
                    'is_video': True,
                    'thumbnail': thumbnail_filename
                })
            else:
                # If the file is not a video, just add the filename
                image_path = os.path.join(SHARED_IMAGE_DIR, file)
                with Image.open(image_path) as img:
                    # Resize the image
                    img.thumbnail((800, 800))  # Resize to 800x800 pixels
                    # Save the resized image
                    resized_image_path = os.path.join(
                        SHARED_IMAGE_DIR, f"resized_{file}")
                    if not os.path.exists(resized_image_path):
                        img.save(resized_image_path)
                images.append({
                    'filename': f"resized_{file}",
                    'is_video': False
                })
        return {"images": images}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/models")
async def list_models():
    try:
        models = os.listdir(DISK_MODELS_DIR)
        return {"models": models}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unexpected error: {str(e)}")


@app.post("/predict")
async def predict(request: Request, file: Optional[UploadFile] = File(None), mediaIndex: Optional[int] = Form(None)):
    global loaded_model, loaded_model_path

    # Get or set the session ID
    session_id = get_or_set_session_id(request)

    print(f"Memory usage initial: {get_memory_usage()}%")

    # Retrieve the model name and path from the session
    model_name, model_path = get_model_info(request)

    # If no model has been loaded or if the model has changed, load the model
    if loaded_model is None or loaded_model_path != model_path:
        loaded_model = load_model(model_path)
        loaded_model_path = model_path

    print("index", mediaIndex)
    # If mediaIndex is provided, use the file at that index in the SHARED_IMAGE_DIR directory
    if mediaIndex is not None:
        files = os.listdir(SHARED_IMAGE_DIR)
        if mediaIndex < 0 or mediaIndex >= len(files):
            raise HTTPException(status_code=400, detail="Invalid mediaIndex")
        file_path = os.path.join(SHARED_IMAGE_DIR, files[mediaIndex])
        with open(file_path, "rb") as f:
            contents = f.read()
        file = UploadFile(
            filename=files[mediaIndex], file=io.BytesIO(contents))
        filename = files[mediaIndex]
    elif file is not None:
        filename = file.filename

    # Use the loaded model for prediction
    model = loaded_model

    # Check if the file is a video
    if file.filename.endswith('.mp4'):
        print("file is video")

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Write the video data to the temporary file
            temp_file.write(await file.read())

        # Open the video file
        video = cv2.VideoCapture(temp_file.name)

        # Get the frames per second (fps) of the video
        fps = video.get(cv2.CAP_PROP_FPS)

        frame_count = 0
        results_json = []
        MAX_FRAMES = 4  # Maximum number of frames to process
        all_results = []  # List to store results for all frames
        processed_results = None  # Initialize processed_results
        annotated_images = []  # List to store annotated images for all frames
        total_instances = 0
        total_classes = set()
        total_area_by_type = defaultdict(int)
        total_area = 0

        # Get the total area of a frame (all frames have the same size)
        frame_area = None

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:

            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break

                # Process every 'fps' frames (i.e., every 2 seconds)
                if frame_count % (fps * 2) == 0:

                    # Reduce the resolution of the frame
                    frame = cv2.resize(frame, (640, 480))

                    # Save the frame as an image
                    frame_path = os.path.join(
                        temp_dir, f"frame_{frame_count}.jpg")
                    cv2.imwrite(frame_path, frame)

                    # Calculate the time in seconds
                    time_in_seconds = frame_count // fps

                    # Run inference on the frame
                    results = model(frame_path)  # list of Results objects

                    # Get the annotated image from the results
                    annotated_image = results[0].plot(
                        font='Roboto-Regular.ttf', pil=True)

                    # Convert the numpy array to a PIL Image
                    annotated_image = Image.fromarray(annotated_image)

                    # Calculate the height of the extra space and the font size
                    extra_space_height = int(0.04 * annotated_image.height)
                    font_size = extra_space_height

                    # Create a new image with extra space at the top
                    new_image = Image.new(
                        'RGB', (annotated_image.width, annotated_image.height + extra_space_height), 'black')

                    # Paste the original image onto the new image
                    new_image.paste(annotated_image, (0, extra_space_height))

                    # Create a draw object
                    draw = ImageDraw.Draw(new_image)

                    # Define the text and position
                    text = f"Frame: {frame_count} ({time_in_seconds} seconds)"

                    # Define the font (you might need to specify the path to the font file)
                    # replace with the actual path to your Roboto-Regular.ttf file
                    font_path = 'Roboto-Regular.ttf'

                    font = ImageFont.truetype(font_path, font_size)

                    # Calculate the width of the text
                    text_width, _ = draw.textsize(text, font=font)

                    # Calculate the position of the text to be centered
                    position = ((new_image.width - text_width) // 2, 0)

                    # Draw the text on the new image
                    draw.text(position, text, fill="white", font=font)

                    # Convert the image to RGB mode
                    annotated_image = new_image.convert("RGB")

                    annotated_image_base64 = image_to_base64_for_video(
                        new_image)

                    # Save the annotated image to the user's directory

                    annotated_image_path = os.path.join(
                        temp_dir, f"annotated_frame_{frame_count}.jpg")
                    print(f"Saving annotated image to {annotated_image_path}")
                    annotated_image.save(annotated_image_path)

                    annotated_images.append(annotated_image_base64)

                    # Create a GIF from the annotated frames
                    annotated_frame_paths = glob.glob(
                        os.path.join(temp_dir, 'annotated_*.jpg'))  # Only match annotated frames
                    annotated_frame_paths = sorted(
                        annotated_frame_paths, key=lambda path: int(path.split('_')[-1].split('.')[0]))
                    images = [Image.open(frame_path)
                              for frame_path in annotated_frame_paths]
                    images[0].save('movie.gif', save_all=True,
                                   append_images=images[1:], duration=500, loop=0)

                    # Read the GIF file and convert it to base64
                    with open('movie.gif', 'rb') as f:
                        gif_base64 = base64.b64encode(f.read()).decode()

                    # Get the size of the image
                    image_size = frame.shape[1], frame.shape[0]

                    # Get the total area of a frame (all frames have the same size)
                    if frame_area is None:
                        frame_area = frame.shape[0] * frame.shape[1]

                    # Add results to all_results list
                    all_results.extend(results)

                    # Convert each Results object to a dictionary

                    # Process the results for this frame
                    processed_results = process_results(results, image_size)

                    # Update the summary data
                    total_instances += len(processed_results['instances'])
                    total_classes.update(processed_results['Classes'])
                    for class_name, area_info in processed_results['Area by type'].items():
                        area = float(area_info['area'])
                        total_area_by_type[class_name] += area

                    # Append the results for this frame to results_json
                    results_json.append({

                        'frame_number': frame_count,
                        'time_in_seconds': time_in_seconds,
                        'annotated_image': annotated_image_base64,

                        'detection_results': processed_results,


                    })

                    # Delete the frame to free up memory
                    del frame
                    gc.collect()

                frame_count += 1

            video.release()
            print(f"Number of frames processed: {len(results_json)}")
            del results
            gc.collect()

        # Calculate the total area distribution
        total_area_by_type = {
            k: {'area': round(v, 1)} for k, v in total_area_by_type.items()}

        print(f"Memory usage after video processing: {get_memory_usage()}%")
        # Calculate the average percentage area distribution
        average_percentage_area_by_type = {k: {
            'percentage_area': f"{round((v['area'] / (frame_area * len(results_json))) * 100, 2)}%"} for k, v in total_area_by_type.items()}
        # Return the results
        return {
            'type': 'video',
            'model_used': model_name,
            'frames': results_json,
            'detection_summary': {
                'Total # of instances': total_instances,
                'Total # of classes': len(total_classes),
                'Total area by type': total_area_by_type,
                'Average % area by type': average_percentage_area_by_type,
            },
            'gif': gif_base64,
        }

    else:

        print("file name1", file)
        print("file name", file.filename)

        # Read image file
        image = Image.open(io.BytesIO(await file.read()))
        # Create a temporary file
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the uploaded file to the temporary directory
            temp_image_path = os.path.join(temp_dir, file.filename)

            # Convert the image to RGB mode
            image = image.convert("RGB")

            image.save(temp_image_path)

            # Run inference on the image
            results = model(temp_image_path)  # list of Results objects

            # Get the annotated image from the results
            annotated_image = results[0].plot(
                font='Roboto-Regular.ttf', pil=True)

            # Convert the numpy array to a PIL Image
            annotated_image = Image.fromarray(annotated_image)

            # Convert the image to RGB mode
            annotated_image = annotated_image.convert("RGB")

            # Create a unique directory for the annotated images within the base directory
            annotated_image_dir = os.path.join(
                USER_IMAGE_BASE_DIR, session_id, 'annotated_images')
            os.makedirs(annotated_image_dir, exist_ok=True)

            # Save the annotated image to the annotated images directory
            annotated_image_path = os.path.join(
                annotated_image_dir, f"annotated_{file.filename}")
            annotated_image.save(annotated_image_path)

            # Read the output image and return it
            image_base64 = image_to_base64_for_image(annotated_image_path)

            print("used model:", model_name)

            # Get the size of the image
            image_size = image.size
            print(image_size)

            print(f"Memory usage after img processing: {get_memory_usage()}%")

            # Process the results
            processed_results = process_results(results, image_size)
            del results

            return {'type': 'image', "image": image_base64, "model_used": model_name, "detection_results": processed_results}


def calculate_area(segments, image_size):
    polygon = Polygon(zip(segments['x'], segments['y']))
    area = polygon.area
    total_area = image_size[0] * image_size[1]
    return round(area, 1), f"{round((area / total_area) * 100, 2)}%"


def process_results(results, image_size):
    instance_counter = defaultdict(int)
    total_areas = defaultdict(int)
    total_objects = 0
    unique_classes = set()
    instances = []

    for result in results:
        result_dicts = json.loads(result.tojson())
        for result_dict in result_dicts:
            total_objects += 1
            class_name = result_dict['name']
            unique_classes.add(class_name)
            instance_counter[class_name] += 1
            area, area_percentage = calculate_area(
                result_dict['segments'], image_size)
            total_areas[class_name] += area
            instances.append({
                'class_name': class_name,
                'area': area,
                'area_percentage': area_percentage
            })

    # Append instance number to class name if there are multiple instances
    for instance in instances:
        class_name = instance['class_name']
        if instance_counter[class_name] > 1:
            instance['name'] = f"{class_name}_{instance_counter[class_name]}"
            instance_counter[class_name] -= 1
        else:
            instance['name'] = f"{class_name}_1"
        del instance['class_name']

    total_image_area = image_size[0] * image_size[1]

    return {
        'Total # of instances': total_objects,
        'Total # of classes': len(unique_classes),
        'Classes': list(unique_classes),
        'Area by type': {k: {'area': round(v, 1), 'area_percentage': f"{round((v / total_image_area) * 100, 1)}%"} for k, v in total_areas.items()},
        'instances': instances
    }
