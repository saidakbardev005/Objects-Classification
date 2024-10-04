import streamlit as st
from fastai.vision.all import *
import plotly.express as px

model_path = st.text_input("Model Path", "Objects.pkl")  # User-specified model path

st.title("Transportni klassifikatsiya qiluvchi model")
st.title("""Ular: 1.Mashinalar 2.Samalyotlar 3.Kemalar""")

uploaded_image = st.file_uploader("Rasm yuklash", type=["png", "jpeg", "gif", "svg"])

if uploaded_image is not None:
    try:
        st.image(uploaded_image)

        # Convert to PIL image
        image = PILImage.create(uploaded_image)

        # Load the model (handle potential errors)
        model = load_learner(model_path)

        # Make prediction
        pred, pred_id, probs = model.predict(image)
        st.success(f"Bashorat: {pred}")
        st.info(f"Ishonchlilik: {probs[pred_id]*100:.1f}%")

        # Create bar chart
        fig = px.bar(x=probs * 100, y=model.dls.vocab)
        st.plotly_chart(fig)
    except Exception as e:
        st.error("Xatolik yuz berdi:")
        # Provide specific error messages here, e.g.,
        if "Could not load file" in str(e):
            st.error("Model fayli topilmadi yoki noto'g'ri yuklandi.")
        elif "Invalid image format" in str(e):
            st.error("Rasm fayli noto'g'ri formatda. Faqat PNG, JPEG, GIF va SVG formatlarini qabul qiladi.")
        else:
            st.error(f"Noma'lum xatolik: {e}")
