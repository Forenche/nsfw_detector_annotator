import streamlit as st
import os
import zipfile
from io import BytesIO
from utils import delete_uploaded_images, delete_feedback_json, FEEDBACK_FILE, UPLOAD_DIR

def download_all_data():
    """Download all uploaded images and feedback JSON as a single ZIP file."""
    if not os.path.exists(UPLOAD_DIR) and not os.path.exists(FEEDBACK_FILE):
        st.warning("No data found to download.")
        return

    # Create a ZIP file
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add uploaded images to the ZIP
        if os.path.exists(UPLOAD_DIR) and os.listdir(UPLOAD_DIR):
            for root, _, files in os.walk(UPLOAD_DIR):
                for file in files:
                    file_path = os.path.join(root, file)
                    zip_file.write(file_path, os.path.relpath(file_path, UPLOAD_DIR))
        else:
            st.warning("No uploaded images found to include in the ZIP.")

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
        # Delete images and feedback JSON after download
        if os.path.exists(UPLOAD_DIR) and os.listdir(UPLOAD_DIR):
            if delete_uploaded_images():
                st.success("All uploaded images have been deleted.")
        if os.path.exists(FEEDBACK_FILE):
            if delete_feedback_json():
                st.success("Feedback JSON file has been deleted.")

def admin_panel():
    st.markdown("---")
    st.header("Admin Panel")

    admin_password = st.text_input("Enter Admin Password", type="password")
    correct_password = st.secrets["admin_password"]  # Access password from secrets

    if admin_password == correct_password:
        st.success("Admin access granted.")
        download_all_data()
    else:
        if admin_password:  # Only show a warning if the user entered an incorrect password
            st.error("Incorrect password. Access denied.")