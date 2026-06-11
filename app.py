import streamlit as st
import os
from trajectory.generate_trajectory import generate_trajectory
from PIL import Image

st.set_page_config(
    page_title="Cricket Ball Trajectory Prediction System",
    page_icon="🏏"
)

st.sidebar.title("🏏 Cricket Ball Tracker")

st.sidebar.markdown("""
### Technology Used

- YOLO11
- OpenCV
- NumPy
- Streamlit
- Python

---

### Future Scope

- Speed Estimation
- Bounce Point Detection
- Length Classification
- LBW Prediction
- Tennis Ball Tracking
- Badminton Shuttle Tracking
""")

st.title("🏏 Cricket Ball Trajectory Prediction System")

st.markdown("---")

banner = Image.open("images/cricket_bg.jpeg")

st.image(
    banner,
    use_container_width=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Model",
        value="YOLO11"
    )

with col2:
    st.metric(
        label="Framework",
        value="OpenCV"
    )

with col3:
    st.metric(
        label="Language",
        value="Python"
    )

st.markdown("---")

with st.container(border=True):
    st.markdown("""
    ### AI-powered Cricket Ball Tracking using YOLO11 and OpenCV

    ####  Developed By
    - Vineet Muley 229310401

    ####  Project Guide
    - Dr. Shinnu 

    ---
    """)

with st.container(border=True):
    st.header(" Institution")

    st.write("""
    Manipal University Jaipur, Jaipur,
    Department of Artificial Intelligence and Machine Learning
    """)

with st.container(border=True):
    st.header(" About the Project")

    st.write("""
    This project aims to detect the cricket ball and generate its trajectory using Computer Vision and Deep Learning techniques.

    The system uses a custom-trained YOLO11 model to detect the ball and applies filtering techniques to produce a smooth trajectory similar to professional cricket broadcasting systems.

    The generated trajectory can be visualized directly on the uploaded video and downloaded for further analysis.
    """)

uploaded_file = st.file_uploader(
    "Choose a video",
    type=["mp4"]
)

if uploaded_file is not None:

    st.success("Video uploaded successfully!")

    if st.button("🚀 Generate Trajectory"):

        os.makedirs("uploads", exist_ok=True)
        os.makedirs("output_videos", exist_ok=True)

        # Save uploaded video
        upload_path = os.path.join(
            "uploads",
            uploaded_file.name
        )

        with open(upload_path, "wb") as f:
            f.write(uploaded_file.getbuffer())


        output_path = os.path.join(
            "output_videos",
            "trajectory_output.mp4"
        )

        with st.spinner("Generating trajectory... Please wait ⏳"):
            generate_trajectory(
                upload_path,
                output_path
            )

        st.success("✅ Trajectory generated successfully!")

        with st.container(border=True):

            st.header("📊 Results")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Video")
                st.video(uploaded_file)

            with col2:
                st.subheader("Trajectory Video")
                import os

                st.write("Exists:", os.path.exists(output_path))

                if os.path.exists(output_path):
                    st.write("Size:", os.path.getsize(output_path), "bytes")

                    with open(output_path, "rb") as video_file:
                        video_bytes = video_file.read()

                    st.video(video_bytes)

        st.markdown("---")
        
        with open(output_path, "rb") as file:

            st.download_button(
                label="⬇ Download Trajectory Video",
                data=file,
                file_name="trajectory_output.mp4",
                mime="video/mp4"
            )
        
st.markdown("---")

st.markdown(
"""
<div style='text-align: center; color: gray;'>

Developed by Vineet Muley, 229310401

Department of Artificial Intelligence and Machine Learning

Manipal University Jaipur

© 2026

</div>
""",
unsafe_allow_html=True
)
