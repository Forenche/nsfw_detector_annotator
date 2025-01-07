import json
import os

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