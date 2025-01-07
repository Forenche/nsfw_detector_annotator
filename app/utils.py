import json
import os
import streamlit as st

# File to store feedbacks
FEEDBACK_FILE = "feedbacks.json"

media_dir_root = "uploaded_media"
image_dir = f'{media_dir_root}/images'
video_dir = f'{media_dir_root}/videos'

"""
    We are guaranteed to have images and videos dir already.
    app.py is executed first which creates the dirs if they do not exist already.
    Duplicate downloads are not a problem as the zip is stored in the buffer so it can be re downloaded multiple times.
"""

def save_feedbacks(feedbacks):
    with open(FEEDBACK_FILE, "w") as file:
        json.dump(feedbacks, file, indent=4)

def load_feedbacks():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as file:
            return json.load(file)
    return []

def delete_uploaded_images():
    if os.path.exists(image_dir):
        for root, _, files in os.walk(image_dir):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
        return True
    return False

def delete_uploaded_videos():
    if os.path.exists(video_dir):
        for root, _, files in os.walk(video_dir):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
        return True
    return False

def delete_feedback_json():
    if os.path.exists(FEEDBACK_FILE):
        os.remove(FEEDBACK_FILE)
        return True
    return False

def set_page_configs():
    
    st.set_page_config(page_title='NSFW Detector & Annotator', page_icon=':peach:', layout="wide", initial_sidebar_state="auto", menu_items=None)

    st.markdown(
        """
        <script>
        function isPhone() {
            return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
        }

        // Set layout to 'centered' if the user is on a phone
        if (isPhone()) {
            document.body.classList.add('phone-layout');
        }
        </script>
        <style>
        /* Apply these styles only when light mode is active */
            @media (prefers-color-scheme: light) {
            .stApp {
                background-color: #ECEFF4;  /* Nord6: Snow Storm */
                color: #2E3440;            /* Nord0: Polar Night */
                font-family: 'Inter', sans-serif;
            }

            /* Headers and titles */
            h1, h2, h3, h4, h5, h6 {
                color: #2E3440;            /* Nord0: Polar Night */
                font-weight: 600;
            }

            /* Buttons */
            .stButton>button {
                background-color: #81A1C1; /* Nord9: Frost */
                color: #ECEFF4;            /* Nord6: Snow Storm */
                border-radius: 8px;
                border: none;
                padding: 10px 20px;
                font-weight: 500;
                transition: all 0.3s ease;
            }

            .stButton>button:hover {
                background-color: #5E81AC; /* Nord10: Frost */
                color: #ECEFF4;            /* Nord6: Snow Storm */
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }

            /* File uploader */
            .stFileUploader>div>div>button {
                background-color: #E5E9F0; /* Nord5: Snow Storm */
                color: #2E3440;            /* Nord0: Polar Night */
                border-radius: 8px;
                border: 1px solid #D8DEE9; /* Nord4: Snow Storm */
                padding: 8px 12px;
            }

            /* Sliders */
            .stSlider>div>div>div>div {
                background-color: #81A1C1; /* Nord9: Frost */
                border-radius: 8px;
            }

            /* Checkboxes */
            .stCheckbox>label {
                color: #2E3440;            /* Nord0: Polar Night */
                font-weight: 500;
            }

            /* Images */
            .stImage>img {
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                transition: transform 0.3s ease;
            }

            .stImage>img:hover {
                transform: scale(1.02);
            }

            /* Spacing and layout */
            .stMarkdown {
                margin-bottom: 1.5rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("NSFW Detection Tool for Images and Videos")
    st.header("Upload images or videos to classify, detect, and blur explicit content.")
    st.write("Detects and classifies content under these 5 classes: drawing, hentai, normal, porn, and sexy.")
