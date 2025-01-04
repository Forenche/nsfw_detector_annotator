import json
import os

# File to store feedbacks
FEEDBACK_FILE = "feedbacks.json"

UPLOAD_DIR = "uploaded_images"

def save_feedbacks(feedbacks):
    with open(FEEDBACK_FILE, "w") as file:
        json.dump(feedbacks, file, indent=4)

def load_feedbacks():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as file:
            return json.load(file)
    return []

def delete_uploaded_images():
    if os.path.exists(UPLOAD_DIR):
        for root, _, files in os.walk(UPLOAD_DIR):
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