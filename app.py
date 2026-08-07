import tempfile
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="Sheep Counter", page_icon="🐑")

st.title("🐑 Sheep Counter")


@st.cache_resource
def get_model():
        return YOLO("yolov8n.pt")


        model = get_model()

        mode = st.selectbox("Select mode:", ["Still Photo", "Video"])

        if mode == "Still Photo":
                file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
                if file:
                        img = Image.open(file)
                        arr = np.array(img)
                        res = model(arr, classes=[19], verbose=False)
                        count = len(res[0].boxes)
                        st.success(f"Found {count} sheep!")

                        annotated_array = res[0].plot()
                        annotated_rgb = annotated_array[..., ::-1]
                        st.image(Image.fromarray(annotated_rgb), use_container_width=True)

                else:
                        vid = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])
                        if vid:
                                tf = tempfile.NamedTemporaryFile(delete=False)
                                tf.write(vid.read())
                                cap = cv2.VideoCapture(tf.name)
                                box = st.empty()
                                max_c = 0

                                while cap.isOpened():
                                        ret, frame = cap.read()
                                        if not ret:
                                                break
                                                res = model(frame, classes=[19], verbose=False)
                                                c = len(res[0].boxes)
                                                max_c = max(max_c, c)
                                                box.metric("Current frame count", c)

                                                cap.release()
                                                st.success(f"Done! Max sheep in a frame: {max_c}")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     