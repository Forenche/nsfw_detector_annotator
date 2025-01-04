import streamlit as st
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from PIL import Image, ImageFilter
import json
import os

# File to store feedbacks
FEEDBACK_FILE = "feedbacks.json"

save_files_to_disk = True

# Directory to save uploaded images
UPLOAD_DIR = "uploaded_images"

# Create the directory if it doesn't exist
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

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

# Function to save feedbacks to file
def save_feedbacks(feedbacks):
    with open(FEEDBACK_FILE, "w") as file:
        json.dump(feedbacks, file, indent=4)

# Function to load feedbacks from file
def load_feedbacks():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as file:
            return json.load(file)
    return []

# Load existing feedbacks
feedbacks = load_feedbacks()

@st.cache_resource(ttl=24*3600)  # Cache models to save on resources
def load_models():
    classification_model = YOLO('models/classification_model.pt')
    segmentation_model = YOLO('models/segmentation_model.pt')
    return classification_model, segmentation_model

classification_model, segmentation_model = load_models()

st.title("NSFW Detection Tool for Images")
st.header("Upload images to classify, detect and blur explicit content.")
st.write("Detects and classifies images under these 5 classes:\n drawing, hentai, normal, porn and sexy respectively.")

uploaded_file = st.file_uploader("📁 Choose an image...", type=["bmp", "dng", "jpg", "jpeg", "mpo", "png", "tif", "tiff", "webp", "pfm", "HEIC"])

if uploaded_file is not None:

    # Check if the uploaded file type is supported
    supported_formats = ["bmp", "dng", "jpg", "jpeg", "mpo", "png", "tif", "tiff", "webp", "pfm", "HEIC"]
    if uploaded_file.type.split("/")[-1].lower() not in supported_formats:
        st.warning(
            f"⚠️ Unsupported file type. Please upload one of the following formats: "
            f"{', '.join(supported_formats)}."
        )

    # Save uploaded images to local disk only if enabled, otherwise continue
    if save_files_to_disk:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    image = Image.open(uploaded_file)
    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
        st.image(image, caption="Uploaded Image", width=500)

    with st.spinner("Classifying image..."):
        classification_results = classification_model(image)
        category = classification_results[0].names[classification_results[0].probs.top1]
        st.success(f"**Classification Result:** {category}")
        print(f"Inference information about file: {uploaded_file.name}")

    if category == 'porn' or category == 'hentai':
        with st.spinner("Detecting explicit regions..."):
            segmentation_results = segmentation_model(image,
                                                      agnostic_nms=True, # Testing
                                                      retina_masks=True) # Returns high-resolution segmentation masks

        boxes = segmentation_results[0].boxes.xyxy.cpu().tolist() # Bound boxes
        clss = segmentation_results[0].boxes.cls.cpu().tolist() # Class value of the boxes
        confs = segmentation_results[0].boxes.conf.cpu().tolist() # Confidence score of the classes

        """
            Copy of the image for drawing segmentation masks
            Prevents segmentation mask's color being picked up during the blurring process, results in a clean blur
        """
        image_with_boxes = image.copy()
        image_with_blur = image.copy()

        annotator = Annotator(image_with_boxes, line_width=2, example=segmentation_results[0].names)

        for box, cls, conf in zip(boxes, clss, confs):
            class_name = segmentation_results[0].names[int(cls)]

            label = f"{class_name} ({conf:.2f})"
            annotator.box_label(box, color=colors(int(cls), True), label=label)

        blur_ratio = st.slider("Blur Ratio", min_value=1, max_value=100, value=85)

        # Blur explicit regions
        for box in boxes:
            
            # Crop the object region
            obj = image_with_blur.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3])))

            # Blur it out
            blur_obj = obj.filter(ImageFilter.GaussianBlur(radius=blur_ratio))

            # No alpha channel please, causes weird tints over the image
            if blur_obj.mode == 'RGBA':
                blur_obj = blur_obj.convert('RGB')

            # Overlay explicit regions with blurred out regions
            image_with_blur.paste(blur_obj, (int(box[0]), int(box[1])))

        if st.checkbox("Blur NSFW Regions"):
            left_co, cent_co,last_co = st.columns(3)
            with cent_co:
                st.image(image_with_blur, caption="Image with Blurred NSFW Regions", use_container_width=True)
                
        else:
            left_co, cent_co,last_co = st.columns(3)
            with cent_co:
                st.image(image_with_boxes, caption="Image with Segmentation Masks", use_container_width=True)

st.markdown("---")
st.markdown("ℹ️ **Instructions:** Upload an image to classify and detect explicit content. Use the slider to adjust the blur intensity.")

with st.form(key='feedback_form'):
    name = st.text_input('Name')
    feedback = st.text_area('💬Feedback')
    rating = st.selectbox('Rating', ['Excellent', 'Good', 'Average', 'Poor'])
    submit_button = st.form_submit_button('Submit')
    if submit_button:
        if name and feedback:  # Ensure name and feedback are not empty
            # Create a new feedback entry
            new_feedback = {
                "name": name,
                "feedback": feedback,
                "rating": rating
            }
            
            feedbacks.append(new_feedback)
            save_feedbacks(feedbacks)
            st.success('Thanks for your feedback!')
        else:
            st.warning('Please fill out all fields.')

st.markdown("ℹ️ Please note that I collect uploaded images for further analysis and may be used re-training. However, the user remains anonymous.")

# Admin Panel
st.markdown("---")
st.header("Admin Panel")

admin_password = st.text_input("Enter Admin Password", type="password")
correct_password = st.secrets["admin_password"]

if admin_password == correct_password:
    st.success("Admin access granted.")

    if os.path.exists(UPLOAD_DIR) and os.listdir(UPLOAD_DIR):
        import zipfile
        from io import BytesIO

        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for root, _, files in os.walk(UPLOAD_DIR):
                for file in files:
                    file_path = os.path.join(root, file)
                    zip_file.write(file_path, os.path.relpath(file_path, UPLOAD_DIR))

        st.download_button(
            label="Download Uploaded Images (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="uploaded_images.zip",
            mime="application/zip"
        )
    else:
        st.warning("No uploaded images found.")

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
else:
    if admin_password:
        st.error("Incorrect password. Access denied.")