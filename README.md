# Project Overview

This project is a FastAPI application that provides a RESTful API for managing and interacting with machine learning models, specifically YOLO models. The application allows users to upload, download, and select models, as well as run predictions on images and videos. The application also provides endpoints for managing and viewing shared images and videos.

## Key Features

- **Model Management**: The application allows users to upload and download machine learning models. The models are stored in a specified directory on the server. Each model is stored in its own directory, which can also contain a description and a photo of the model.

- **Model Selection**: Users can select a model for use in predictions. The selected model is stored in the user's session.

- **Predictions**: The application can run predictions on images and videos using the selected model. The results of the predictions are returned as JSON data. For videos, the application processes every 2 seconds of the video and returns the results for each processed frame.

- **Shared Media**: The application provides endpoints for managing and viewing shared images and videos. The shared media files are stored in a specified directory on the server.

- **Session Management**: The application uses session middleware to manage user sessions. Each user is assigned a unique session ID, which is used to store the user's selected model and other session data.

- **CORS**: The application uses CORS middleware to handle Cross-Origin Resource Sharing.

## Usage

To use the application, start the FastAPI server and send HTTP requests to the API endpoints. The endpoints accept and return data in JSON format. Some endpoints also accept file uploads in multipart/form-data format.

## Dependencies

The application depends on several Python libraries, including FastAPI, Starlette, Pydantic, OpenCV, Torch, YOLO, Shapely, MoviePy, and Pillow.
