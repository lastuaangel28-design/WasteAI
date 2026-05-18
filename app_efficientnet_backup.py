import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# 1. Page Config
st.set_page_config(page_title="Waste Classification (3 Classes)", layout="centered")

st.title("♻️ Waste Classification: Biodegradable, Recyclable, Residual")
st.write("Upload an image of waste to classify it.")

# 2. Load the Model
@st.cache_resource
def load_model_cached():
    try:
        # This file must be in your GitHub repository
        model = load_model("waste_efficientnet_3class.keras")
        return model
    except:
        st.error("Model file not found. Make sure 'waste_efficientnet_3class.keras' is uploaded to GitHub.")
        return None

model = load_model_cached()

# 3. Image Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Display the image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    st.write("Classifying...")
    
    # --- PREPROCESSING (Must match Training) ---
    # Force convert to RGB to prevent channel errors (matching the training fix)
    img = img.convert("RGB") 
    
    # Resize to 224x224
    img = img.resize((224, 224))
    
    # Convert to array
    img_array = image.img_to_array(img)
    
    # Add batch dimension (Model expects a list of images)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize pixel values to [-1, 1] (Matches Rescaling layer in training)
    img_array = img_array / 127.5 - 1.0 

    # --- PREDICTION ---
    predictions = model.predict(img_array)
    
    # Get the index of the highest probability (0, 1, or 2)
    predicted_index = np.argmax(predictions[0])
    score = predictions[0][predicted_index]
    
    # Map index to Label (Alphabetical order: Biodegradable, Recyclable, Residual)
    class_names = ["Biodegradable", "Recyclable", "Residual"]
    predicted_label = class_names[predicted_index]
    
    confidence = score * 100

    # Display Results
    st.success(f"Prediction: **{predicted_label}**")
    st.write(f"Confidence: {confidence:.2f}%")
    
    # Optional: Detailed breakdown
    with st.expander("See detailed probabilities"):
        for i, name in enumerate(class_names):
            st.write(f"{name}: {predictions[0][i]*100:.2f}%")