"""
This module is designed for YOLO object detection in a Google Colab or similar environment. 
It provides functionalities for loading models, processing images and videos, running predictions, 
and returning results as a pandas DataFrame. It is not designed for use with FastAPI or a client-server method.
"""

import gc
import json
import base64
import glob
from ultralytics import YOLO
import pandas as pd
import cv2
import os
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon
from collections import defaultdict
from config import Config


loaded_model = None
loaded_model_path = None


def load_model_if_needed():
    global loaded_model, loaded_model_path

    DEFAULT_MODEL_NAME = Config.DEFAULT_MODEL_NAME

    DISK_MODELS_DIR = 'disk/models'
    DEFAULT_MODEL_DIR = os.path.join(DISK_MODELS_DIR, DEFAULT_MODEL_NAME)

    DEFAULT_MODEL_PATH = os.path.join(
        DEFAULT_MODEL_DIR, f"{DEFAULT_MODEL_NAME}.pt")
    # Use the default model name and path
    model_name, model_path = DEFAULT_MODEL_NAME, DEFAULT_MODEL_PATH

    # If no model has been loaded or if the model has changed, load the model
    if loaded_model is None or loaded_model_path != model_path:
        loaded_model = load_model(model_path)
        loaded_model_path = model_path

    return loaded_model, model_name


def predict_image(image_path: str):
    # Use the loaded model for prediction
    model, model_name = load_model_if_needed()

    # Read image file
    image = Image.open(image_path)

    # Run inference on the image
    results = model(image_path)  # list of Results objects

    # Get the annotated image from the results
    annotated_image = results[0].plot(font='Roboto-Regular.ttf', pil=True)

    # Convert the numpy array to a PIL Image
    annotated_image = Image.fromarray(annotated_image)

    # Convert the image to RGB mode
    annotated_image = annotated_image.convert("RGB")

    # Get the size of the image
    image_size = image.size

    # Process the results
    processed_results = process_results(results, image_size)
    del results

    # Convert the processed results to a pandas DataFrame
    df = pd.DataFrame(processed_results['instances'])

    # Save the annotated image to a file
    annotated_image_path = 'annotated_image.jpg'
    annotated_image.save(annotated_image_path)

    # Convert the annotated image to base64
    annotated_image_base64 = image_to_base64_for_image(annotated_image_path)

    return df, {'type': 'image', "image": annotated_image_base64, "model_used": model_name}


def predict_video(video_path: str):
    # Use the loaded model for prediction
    model, model_name = load_model_if_needed()

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get the frames per second (fps) of the video
    fps = video.get(cv2.CAP_PROP_FPS)

    frame_count = 0
    all_results = []  # List to store results for all frames

    # Create directories for storing the frames and annotated images
    frames_dir = 'frames'
    annotated_frames_dir = 'annotated_frames'
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(annotated_frames_dir, exist_ok=True)

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Calculate the time in seconds
        time_in_seconds = frame_count // fps

        # Process every 'fps' frames (i.e., every 2 seconds)
        if frame_count % fps == 0:
            # Reduce the resolution of the frame
            frame = cv2.resize(frame, (640, 480))

            # Save the frame as an image
            frame_path = os.path.join(
                frames_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)

            # Run inference on the frame
            results = model(frame_path)  # list of Results objects

            # Get the annotated image from the results
            annotated_image = results[0].plot(
                font='Roboto-Regular.ttf', pil=True)

            # Convert the numpy array to a PIL Image
            annotated_image = Image.fromarray(annotated_image)

            # Convert the image to RGB mode
            annotated_image = annotated_image.convert("RGB")

            # Create a draw object
            draw = ImageDraw.Draw(annotated_image)

            # Define the text and position
            text = f"Frame: {frame_count} ({time_in_seconds} seconds)"

            # Define the font (you might need to specify the path to the font file)
            # replace with the actual path to your Roboto-Regular.ttf file
            font_path = 'Roboto-Regular.ttf'

            # Calculate the height of the extra space and the font size
            extra_space_height = int(0.04 * annotated_image.height)
            font_size = extra_space_height

            font = ImageFont.truetype(font_path, font_size)

            # Calculate the width of the text
            text_width, _ = draw.textsize(text, font=font)

            # Calculate the position of the text to be centered
            position = (10, 10)  # Top left corner

            # Draw the text on the new image with a black background
            draw.rectangle(
                [position, (position[0] + text_width, position[1] + font_size)], fill='black')
            draw.text(position, text, fill="white", font=font)

            # Save the annotated image to the annotated_frames directory
            annotated_image_path = os.path.join(
                annotated_frames_dir, f"annotated_frame_{frame_count}.jpg")
            annotated_image.save(annotated_image_path)

            # Get the size of the image
            image_size = frame.shape[1], frame.shape[0]

            # Process the results for this frame
            processed_results = process_results(results, image_size)

            # Add results to all_results list
            all_results.extend(processed_results['instances'])

            # Delete the frame to free up memory
            del frame
            gc.collect()

        frame_count += 1

    video.release()

    # Convert the processed results to a pandas DataFrame
    df = pd.DataFrame(all_results)

    # Get a list of all the annotated frame paths
    annotated_frame_paths = glob.glob('annotated_frames/*.jpg')

    # Sort the paths by frame number
    annotated_frame_paths = sorted(
        annotated_frame_paths, key=lambda path: int(path.split('_')[-1].split('.')[0]))

    # Read all the annotated frames into a list
    images = [Image.open(frame_path) for frame_path in annotated_frame_paths]

    # Save the images as a GIF
    images[0].save('movie.gif', save_all=True,
                   append_images=images[1:], duration=500, loop=0)

    return df


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

# Convert the PIL Image to a base64 string


def image_to_base64_for_video(pil_image):
    byte_arr = io.BytesIO()
    pil_image.save(byte_arr, format='JPEG')
    encoded_image = base64.encodebytes(byte_arr.getvalue()).decode('ascii')
    return encoded_image


def image_to_base64_for_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def load_model(model_path: str):
    if not os.path.exists(model_path):
        print("error: model file does not exist")
    try:
        return YOLO(model_path)
    except RuntimeError as e:
        print("error:", e)
    except Exception as e:
        print("error:", e)
