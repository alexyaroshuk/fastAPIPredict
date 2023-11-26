from config import Config
import os
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon
from collections import defaultdict
import json
import io
import base64
from ultralytics import YOLO


def predict_image(image_path: str):
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

    # Use the loaded model for prediction
    model = loaded_model

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

    # Convert the annotated image to base64
    annotated_image_base64 = image_to_base64_for_image(annotated_image)

    return {'type': 'image', "image": annotated_image_base64, "model_used": model_name, "detection_results": processed_results}


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
