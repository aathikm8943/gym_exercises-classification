import streamlit as st
import tempfile

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.pipelines.prediction_pipeline import VideoPredictor

st.set_page_config(page_title="Video Classifier", layout="centered")
st.title("🏋️‍♂️ Exercise Video Classifier")

predictor = VideoPredictor()

st.markdown("Upload a video file (.mp4), and the model will predict the exercise class.")

uploaded_file = st.file_uploader("Upload MP4 Video", type=["mp4"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    st.video(uploaded_file)

    with st.spinner("Predicting... ⏳"):
        result = predictor.predict(video_path)

    if "error" in result:
        st.error(f"Prediction failed: {result['error']}")
    else:
        st.success(f"Predicted Class: **{result['predicted_class']}**")
        st.write(f"Confidence: `{result['confidence'] * 100:.2f}%`")

        st.markdown("### Class Probabilities")
        st.json(result["probabilities"])
        st.bar_chart(result['probabilities'])

    # Clean up temp file
    os.remove(video_path)
