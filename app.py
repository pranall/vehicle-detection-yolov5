import streamlit as st
from PIL import Image
import torch
import numpy as np
import cv2

st.title("Vehicle Detection — YOLOv5m")
st.write("Upload an image to detect: Ambulance, Bus, Car, Motorcycle, Truck")

@st.cache_resource
def load_model():
    model = torch.hub.load('yolov5_src', 'custom', path='best.pt', source='local', device='cpu')
    return model

model = load_model()

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    results = model(np.array(image))
    results.render()
    annotated = results.ims[0]

    st.image(annotated, caption="Detection Result", use_container_width=True)

    st.subheader("Detections")
    st.dataframe(results.pandas().xyxy[0])