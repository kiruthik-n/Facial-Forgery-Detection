import streamlit as st
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('my_model.keras')

# Define class names
class_names = ['Fake', 'Real']

# Streamlit UI
st.title("Image Classification App")
st.write("Upload an image, and the model will classify it as Fake or Real.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the image
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)
    
    # Make prediction
    label = model.predict(img_array)
    predicted_class_index = np.argmax(label)
    predicted_class = class_names[predicted_class_index]
    
    # Display the image
    st.image(img, caption=f"Predicted Class: {predicted_class}", use_column_width=True)
    st.write(f"### Prediction: {predicted_class}")
