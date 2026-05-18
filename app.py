import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import tensorflow as tf
# ADD THESE TWO LINES:
from tensorflow.keras.applications import resnet50
from tensorflow.keras.applications import mobilenet_v2

# --- PAGE CONFIG ---
st.set_page_config(page_title="Model Comparison", layout="wide")

st.title("🤖 Model Comparison: EfficientNet vs. ResNet vs. MobileNetV2")
st.write("Upload an image to see how all three models classify it.")

# --- SIDEBAR ---
st.sidebar.header("Settings")
model_option = st.sidebar.radio(
    "Choose Model to Run:",
    ("EfficientNet", "ResNet", "MobileNetV2", "Compare All")
)

# --- LOAD MODELS ---
@st.cache_resource
def load_efficientnet():
    return load_model("waste_efficientnet_3class.keras")

@st.cache_resource
# ... existing code ...

@st.cache_resource
def load_resnet():
    # FIX: Tell Keras where to find the preprocess_input function
    return load_model(
        "waste_resnet_3class.keras", 
        custom_objects={'preprocess_input': resnet50.preprocess_input}
    )

@st.cache_resource
def load_mobilenet():
    # FIX: Tell Keras where to find the preprocess_input function
    return load_model(
        "waste_mobilenet_3class.keras", 
        custom_objects={'preprocess_input': mobilenet_v2.preprocess_input}
    )

# ... existing code ...

# --- PREPROCESSING FUNCTIONS ---
def preprocess_efficientnet(img_array):
    return img_array / 127.5 - 1.0

def preprocess_resnet(img_array):
    return tf.keras.applications.resnet50.preprocess_input(img_array)

def preprocess_mobilenet(img_array):
    # MobileNetV2 also scales to [-1, 1]
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

# --- IMAGE UPLOAD ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Load Image
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)

    class_names = ["Biodegradable", "Recyclable", "Residual"]

    # 1. EFFICIENTNET
    if model_option == "EfficientNet":
        try:
            model = load_efficientnet()
            processed = preprocess_efficientnet(img_batch)
            pred = model.predict(processed, verbose=0)
            idx = np.argmax(pred[0])
            st.metric(f"EfficientNet Prediction", class_names[idx], f"{pred[0][idx]*100:.2f}%")
        except Exception as e:
            st.error(f"Error: {e}")

    # 2. RESNET
    elif model_option == "ResNet":
        try:
            model = load_resnet()
            processed = preprocess_resnet(img_batch)
            pred = model.predict(processed, verbose=0)
            idx = np.argmax(pred[0])
            st.metric(f"ResNet Prediction", class_names[idx], f"{pred[0][idx]*100:.2f}%")
        except Exception as e:
            st.error(f"Error: {e}")

    # 3. MOBILENET
    elif model_option == "MobileNetV2":
        try:
            model = load_mobilenet()
            processed = preprocess_mobilenet(img_batch)
            pred = model.predict(processed, verbose=0)
            idx = np.argmax(pred[0])
            st.metric(f"MobileNetV2 Prediction", class_names[idx], f"{pred[0][idx]*100:.2f}%")
        except Exception as e:
            st.error(f"Error: {e}")

    # 4. COMPARE ALL
    elif model_option == "Compare All":
        col_eff, col_res, col_mob = st.columns(3)
        
        # EfficientNet
        with col_eff:
            st.subheader("EfficientNet")
            try:
                m_eff = load_efficientnet()
                p_eff = preprocess_efficientnet(img_batch)
                r_eff = m_eff.predict(p_eff, verbose=0)
                i_eff = np.argmax(r_eff[0])
                st.success(f"**{class_names[i_eff]}**")
                st.caption(f"{r_eff[0][i_eff]*100:.1f}%")
            except: st.error("Missing Model")

        # ResNet
        with col_res:
            st.subheader("ResNet")
            try:
                m_res = load_resnet()
                p_res = preprocess_resnet(img_batch)
                r_res = m_res.predict(p_res, verbose=0)
                i_res = np.argmax(r_res[0])
                st.success(f"**{class_names[i_res]}**")
                st.caption(f"{r_res[0][i_res]*100:.1f}%")
            except: st.error("Missing Model")

        # MobileNet
        with col_mob:
            st.subheader("MobileNetV2")
            try:
                m_mob = load_mobilenet()
                p_mob = preprocess_mobilenet(img_batch)
                r_mob = m_mob.predict(p_mob, verbose=0)
                i_mob = np.argmax(r_mob[0])
                st.success(f"**{class_names[i_mob]}**")
                st.caption(f"{r_mob[0][i_mob]*100:.1f}%")
            except: st.error("Missing Model")
