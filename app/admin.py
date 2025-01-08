import streamlit as st
import os
import zipfile
from io import BytesIO
from utils import delete_uploaded_images, delete_uploaded_videos, delete_feedback_json, FEEDBACK_FILE, image_dir, video_dir

def download_all_data():
    if not os.path.exists(image_dir) and not os.path.exists(video_dir) and not os.path.exists(FEEDBACK_FILE):
        st.warning("No data found to download.")
        return

    # Create a ZIP file
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add uploaded images to the ZIP
        if os.path.exists(image_dir) and os.listdir(image_dir):
            for root, _, files in os.walk(image_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    zip_file.write(file_path, os.path.relpath(file_path, image_dir))
        else:
            st.warning("No uploaded images found to include in the ZIP.")

        # Add uploaded videos to the ZIP
        if os.path.exists(video_dir) and os.listdir(video_dir):
            for root, _, files in os.walk(video_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    zip_file.write(file_path, os.path.relpath(file_path, video_dir))
        else:
            st.warning("No uploaded videos found to include in the ZIP.")

        # Add feedback JSON to the ZIP
        if os.path.exists(FEEDBACK_FILE):
            zip_file.write(FEEDBACK_FILE, os.path.basename(FEEDBACK_FILE))
        else:
            st.warning("No feedback JSON file found to include in the ZIP.")

    # Add a download button for the ZIP file
    if st.download_button(
        label="Download All Data (ZIP)",
        data=zip_buffer.getvalue(),
        file_name="all_data.zip",
        mime="application/zip"
    ):
        # Delete media and feedback JSON after download
        if os.path.exists(image_dir) and os.listdir(image_dir):
            if delete_uploaded_images():
                st.success("All uploaded images have been deleted.")
        if os.path.exists(video_dir) and os.listdir(video_dir):
            if delete_uploaded_videos():
                st.success("All uploaded videos have been deleted.")
        if os.path.exists(FEEDBACK_FILE):
            if delete_feedback_json():
                st.success("Feedback JSON file has been deleted.")

def admin_panel():
    st.markdown("---")
    st.markdown("#### Admin Panel")

    admin_password = st.text_input("Enter Admin Password", type="password")
    correct_password = st.secrets["admin_password"]

    if admin_password == correct_password:
        print("[INFO] Access granted to admin panel.")
        st.success("Admin access granted.")
        download_all_data()
    else:
        if admin_password:
            print("[INFO] Admin panel access denied.")
            st.error("Incorrect password. Access denied.")