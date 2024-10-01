import streamlit as st
import requests
from PIL import Image
import io

st.title("Brain MRI Segmentation")

uploaded_file = st.file_uploader("Choose a brain MRI image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)
    
    # Make request to FastAPI backend
    files = {"file": uploaded_file.getvalue()}
    response = requests.post("http://localhost:8000/segment/", files=files)
    
    if response.status_code == 200:
        # Display segmentation result
        mask = Image.open(io.BytesIO(response.content))
        st.image(mask, caption='Segmentation Result', use_column_width=True)
    else:
        st.error("Error processing image")