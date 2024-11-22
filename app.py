import streamlit as st
from fastai.vision.all import *
import plotly.express as px

model_path = st.text_input("Model Path", "Objects.pkl")  # User-specified model path

st.title("Transport classification model")
st.title("""They are: 1. Cars 2. Airplanes 3. Ships""")

uploaded_image = st.file_uploader("Upload image", type=["png", "jpeg", "gif", "svg"])

if uploaded_image is not None:
    try:
        st.image(uploaded_image)

        # Convert to PIL image
        image = PILImage.create(uploaded_image)

        # Load the model (handle potential errors)
        model = load_learner(model_path)

        # Make prediction
        pred, pred_id, probs = model.predict(image)
        st.success(f"Prophecy: {pred}")
        st.info(f"Reliability: {probs[pred_id]*100:.1f}%")

        # Create bar chart
        fig = px.bar(x=probs * 100, y=model.dls.vocab)
        st.plotly_chart(fig)
    except Exception as e:
        st.error("An error occurred:")
        # Provide specific error messages here, e.g.,
        if "Could not load file" in str(e):
            st.error("The model file was not found or was loaded incorrectly.")
        elif "Invalid image format" in str(e):
            st.error("The image file is in the wrong format. Only PNG, JPEG, GIF, and SVG formats are accepted.")
        else:
            st.error(f"Unknown error: {e}")
