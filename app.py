import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

st.set_page_config(
    page_title="Sheep Counter App", page_icon="🐑", layout="centered"
    )

    st.title("🐑 מונה כבשים חכם לשטח")
    st.write("צלם את העדר באמצעות מצלמת הטלפון והאפליקציה תספור את הכבשים עבורך.")


    @st.cache_resource
    def load_model():
        model = YOLO("yolov8n.pt")
            return model


            with st.spinner("טוען את מודל הבינה המלאכותית..."):
                model = load_model()

                picture = st.camera_input("לחץ על הכפתור כדי לצלם את הכבשים")

                if picture is not None:
                    image = Image.open(picture)
                        img_array = np.array(image)

                            with st.spinner("מנתח את התמונה וספירת כבשים..."):
                                    results = model(img_array, classes=[19])
                                            sheep_count = len(results[0].boxes)

                                                st.success(f"נמצאו בתמונה: **{sheep_count} כבשים!** 🎉")

                                                    annotated_img = results[0].plot()
                                                        st.image(
                                                                cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
                                                                        channels="RGB",
                                                                                use_container_width=True,
                                                                                    )
                                                                                    