import tempfile
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# Set up the web page configuration
st.set_page_config(
    page_title="Sheep Counter App", page_icon="🐑", layout="centered"
    )

    st.title("🐑 Sheep Counter - Stills & Video")
    st.write(
        "Choose whether you want to upload a still photo or a video of the flock."
        )


        # Load the AI model
        @st.cache_resource
        def load_model():
            model = YOLO("yolov8n.pt")  # Lightweight model for detection
                return model


                with st.spinner("Loading AI model..."):
                    model = load_model()

                    # Menu to choose between Still Photo or Video
                    mode = st.selectbox("Select mode:", ["Still Photo", "Video"])

                    if mode == "Still Photo":
                        # Upload or capture a still image
                            uploaded_image = st.file_uploader(
                                    "Choose an image file", type=["jpg", "jpeg", "png"]
                                        )

                                            if uploaded_image is not None:
                                                    image = Image.open(uploaded_image)
                                                            img_array = np.array(image)

                                                                    with st.spinner("Analyzing image..."):
                                                                                results = model(img_array, classes=[19], verbose=False)
                                                                                            sheep_count = len(results[0].boxes)

                                                                                                    st.success(f"Found in image: **{sheep_count} sheep!** 🎉")

                                                                                                            # Display the annotated image with bounding boxes
                                                                                                                    annotated_img = results[0].plot()
                                                                                                                            st.image(
                                                                                                                                        cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
                                                                                                                                                    channels="RGB",
                                                                                                                                                                use_container_width=True,
                                                                                                                                                                        )

                                                                                                                                                                        else:
                                                                                                                                                                            # Upload a video file
                                                                                                                                                                                video_file = st.file_uploader(
                                                                                                                                                                                        "Choose a video file", type=["mp4", "mov", "avi", "webm"]
                                                                                                                                                                                            )

                                                                                                                                                                                                if video_file is not None:
                                                                                                                                                                                                        tfile = tempfile.NamedTemporaryFile(delete=False)
                                                                                                                                                                                                                tfile.write(video_file.read())

                                                                                                                                                                                                                        cap = cv2.VideoCapture(tfile.name)

                                                                                                                                                                                                                                st.write("Processing video frames...")
                                                                                                                                                                                                                                        st_frame = st.empty()
                                                                                                                                                                                                                                                count_placeholder = st.empty()

                                                                                                                                                                                                                                                        max_sheep_count = 0

                                                                                                                                                                                                                                                                while cap.isOpened():
                                                                                                                                                                                                                                                                            ret, frame = cap.read()
                                                                                                                                                                                                                                                                                        if not ret:
                                                                                                                                                                                                                                                                                                        break

                                                                                                                                                                                                                                                                                                                    results = model(frame, classes=[19], verbose=False)
                                                                                                                                                                                                                                                                                                                                current_count = len(results[0].boxes)
                                                                                                                                                                                                                                                                                                                                            max_sheep_count = max(max_sheep_count, current_count)

                                                                                                                                                                                                                                                                                                                                                        count_placeholder.metric(
                                                                                                                                                                                                                                                                                                                                                                        label="Sheep detected in current frame", value=current_count
                                                                                                                                                                                                                                                                                                                                                                                    )

                                                                                                                                                                                                                                                                                                                                                                                                annotated_frame = results[0].plot()
                                                                                                                                                                                                                                                                                                                                                                                                            st_frame.image(
                                                                                                                                                                                                                                                                                                                                                                                                                            cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                                                                                                                                                                                                                                                                                                                                                                                                                                            channels="RGB",
                                                                                                                                                                                                                                                                                                                                                                                                                                                            use_container_width=True,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                        )

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                cap.release()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        st.success(
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    f"Processing complete! Maximum sheep counted in a single frame:"
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                f" **{max_sheep_count}** 🎉"
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        )
                                                                                                                                                                                                                                                                                                        