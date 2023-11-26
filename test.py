from predict import predict_video
from IPython.display import display

video_path = 'disk\shared_images\85e2f96b-c575-46da-91ce-08433a8a388a (1).mp4'
df = predict_video(video_path)
display(df)
