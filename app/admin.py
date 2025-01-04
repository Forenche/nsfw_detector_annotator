import streamlit as st
import os
import zipfile
from io import BytesIO

# Directory to save uploaded images
UPLOAD_DIR = "uploaded_images"

# File to store feedbacks
FEEDBACK_FILE = "feedbacks.json"

def download_uploaded_images():
    if os.path.exists(UPLOAD_DIR) and os.listdir(UPLOAD_DIR):
        # Create a ZIP file of all uploaded images
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for root, _, files in os.walk(UPLOAD_DIR):
                for file in files:
                    file_path = os.path.join(root, file)
                    zip_file.write(file_path, os.path.relpath(file_path, UPLOAD_DIR))

        # Add a download button for the ZIP file
        st.download_button(
            label="Download Uploaded Images (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="uploaded_images.zip",
            mime="application/zip"
        )
    else:
        st.warning("No uploaded images found.")

def download_feedback_json():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "rb") as file:
            st.download_button(
                label="Download Feedback Data (JSON)",
                data=file,
                file_name="feedbacks.json",
                mime="application/json"
            )
    else:
        st.warning("No feedback data found.")

def admin_panel():
    st.markdown("---")
    st.header("Admin Panel")

    admin_password = st.text_input("Enter Admin Password", type="password")
    correct_password = st.secrets["admin_password"]  # Access password from secrets

    if admin_password == correct_password:
        st.success("Admin access granted.")
        download_uploaded_images()
        download_feedback_json()
    else:
        if admin_password:  # Only show a warning if the user entered an incorrect password
            st.error("Incorrect password. Access denied.")