# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory in the container to /app
WORKDIR /

# Install the necessary packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev

# Add the current directory contents into the container at /app
ADD . /

# Add the /disk and /model directories
ADD disk /disk

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Run the command to start your application when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]