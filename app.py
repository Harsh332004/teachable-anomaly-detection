import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model (H5 format)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

# Load labels from labels.txt
@st.cache_data
def load_labels():
    with open("labels.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

model = load_model()
class_names = load_labels()

# Streamlit UI
st.title("Anomaly Detection App")
st.write("Upload an image to predict if it's normal or anomalous.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image = image.resize((224, 224))  # match the model's input size
    img_array = np.array(image) / 255.0  # normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    st.markdown(f"### Prediction: `{class_names[predicted_class]}`")
    st.markdown(f"### Confidence: `{confidence * 100:.2f}%`")
