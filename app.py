import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

st.set_page_config(
    page_title="Sheep Counter App", page_icon="🐑", layout="centered"
    )

    st.title("🐑 Smart Sheep Counter")
    st.write(
        "Take a picture of the flock using your phone camera, and the app will"
            " count the sheep for you."
            )


            @st.cache_resource
            def load_model():
                model = YOLO("yolov8n.pt")
                    return model


                    with st.spinner("Loading AI model..."):
                        model = load_model()

                        picture = st.camera_input("Click the button to take a picture of the flock")

                        if picture is not None:
                            image = Image.open(picture)
                                img_array = np.array(image)

                                    with st.spinner("Analyzing image and counting sheep..."):
                                            results = model(img_array, classes=[19])
                                                    sheep_count = len(results[0].boxes)

                                                        st.success(f"Found in image: **{sheep_count} sheep!** 🎉")

                                                            annotated_img = results[0].plot()
                                                                st.image(
                                                                        cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
                                                                                channels="RGB",
                                                                                        use_container_width=True,
                                                                                            )
                                                                                            