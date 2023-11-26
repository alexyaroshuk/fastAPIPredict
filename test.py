from predict import predict_image
from IPython.display import display

df, info = predict_image('disk/shared_images/02_014.jpg')
display(df)
