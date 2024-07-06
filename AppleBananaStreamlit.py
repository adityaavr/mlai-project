import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps


def import_and_predict(image_data, model):
    size = (150, 150)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)  # Use LANCZOS instead of ANTIALIAS
    image = image.convert('RGB')
    image = np.asarray(image)
    image = (image.astype(np.float32) / 255.0)

    img_reshape = image[np.newaxis, ...]

    prediction = model.predict(img_reshape)

    return prediction


model = tf.keras.models.load_model(
    '/Users/aditya/PycharmProjects/mlai_project/my_model.h5')  # Change to your model directory

st.write("""
         # Apple-Banana Prediction
         """)

st.write("This is a simple image classification web app to predict Apple-Banana. Upload an image of an Apple or a Banana.")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)

    if np.argmax(prediction) == 0:
        st.write("It is an apple!")
    elif np.argmax(prediction) == 1:
        st.write("It is a banana!")
    else:
        st.write("It is unknown!")

    st.text("Probability (0: Apple, 1: Banana, 2: Unknown)")
    st.write(prediction)