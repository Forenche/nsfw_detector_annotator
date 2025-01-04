import json
import os

# File to store feedbacks
FEEDBACK_FILE = "feedbacks.json"

def save_feedbacks(feedbacks):
    with open(FEEDBACK_FILE, "w") as file:
        json.dump(feedbacks, file, indent=4)

def load_feedbacks():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as file:
            return json.load(file)
    return []