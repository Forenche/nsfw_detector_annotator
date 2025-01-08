import tempfile
import streamlit as st
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from PIL import Image, ImageFilter
import os
import cv2
from pillow_heif import register_heif_opener
import utils
from utils import save_feedbacks, load_feedbacks
from admin import admin_panel

# Enable support for HEIC images
register_heif_opener()

# Set theme and page layout
utils.set_page_configs()

# Directory to save uploaded images
media_dir_root = "uploaded_media"
image_dir = f'{media_dir_root}/images'
video_dir = f'{media_dir_root}/videos'

# Make sure the dirs exist
os.makedirs(image_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

@st.cache_resource(ttl=24*3600)  # Cache models to save on resources
def load_models():
    classification_model = YOLO('models/classification_model.pt')
    segmentation_model = YOLO('models/segmentation_model.pt')
    return classification_model, segmentation_model

classification_model, segmentation_model = load_models()

# Session state to track image navigation
if "image_index" not in st.session_state:
    st.session_state.image_index = 0

# Initialize saved_image_paths in session state if it doesn't exist
if "saved_image_paths" not in st.session_state:
    st.session_state.saved_image_paths = []

# Init session state for caching results
if "results_cache" not in st.session_state:
    st.session_state.results_cache = {}

# Toggle between image and video mode
on = st.toggle("Video mode")

if on:
    st.write("Model will segment explicit regions in videos.")
    uploaded_file = st.file_uploader("📁 Choose a video...", type=["asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv", "webm"])

    if uploaded_file is not None:
        # Save uploaded video to local disk
        file_path = os.path.join(video_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Open the video file
        cap = cv2.VideoCapture(file_path)
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Create a temporary file to save the processed video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_video_path = temp_file.name

        # Video writer
        video_writer = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Process video frame-by-frame
        progress_bar = st.progress(0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                break

            # Perform segmentation on the frame
            results = segmentation_model.predict(im0, imgsz=416, show=False, agnostic_nms=True, device='cpu')
            boxes = results[0].boxes.xyxy.cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            annotator = Annotator(im0, line_width=2, example=segmentation_model.model.names)

            if boxes is not None:
                for box, cls in zip(boxes, clss):
                    annotator.box_label(box, color=colors(int(cls), True), label=segmentation_model.model.names[int(cls)])

                    # Blur explicit regions
                    obj = im0[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    blur_obj = cv2.blur(obj, (85, 85))  # Fixed blur ratio
                    im0[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = blur_obj

            # Write the processed frame to the output video
            video_writer.write(im0)

            # Update progress bar
            current_frame += 1
            progress_bar.progress(current_frame / frame_count)

        # Release video objects
        cap.release()
        video_writer.release()

        # Display the processed video
        st.success("Video processing complete!")
        print('[DEBUG] Video processing  completed.')
        col1, col2, col3 = st.columns([1, 2, 1])
        # Display the video in the middle column
        with col2:
            st.video(temp_video_path)

        # Clean up temporary file
        os.unlink(temp_video_path)

else:
    st.write("Models will classify and segment explicit regions in images.")
    uploaded_files = st.file_uploader(
        "📁 Choose an image...",
        type=["bmp", "dng", "jpg", "jpeg", "mpo", "png", "tif", "tiff", "webp", "pfm", "HEIC"],
        accept_multiple_files=True
    )

    if uploaded_files:
        current_uploaded_files = {file.name for file in uploaded_files}
            
        # Update saved_image_paths to remove deleted files
        st.session_state.saved_image_paths = [
            path for path in st.session_state.saved_image_paths
            if os.path.basename(path) in current_uploaded_files
        ]

        # Add new files to saved_image_paths
        for uploaded_file in uploaded_files:
            file_path = os.path.join(image_dir, uploaded_file.name)
            if file_path not in st.session_state.saved_image_paths:  # Avoid duplicates
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.saved_image_paths.append(file_path)
        
        # Handle invalid image_index
        if st.session_state.image_index >= len(st.session_state.saved_image_paths):
            st.session_state.image_index = max(0, len(st.session_state.saved_image_paths) - 1)
        
        # Load the current image based on image_index
        if st.session_state.saved_image_paths:
            current_image_path = st.session_state.saved_image_paths[st.session_state.image_index]
            image = Image.open(current_image_path)
            left_co, cent_co, last_co = st.columns(3)
            with cent_co:
                st.image(image, caption=f"Image {st.session_state.image_index + 1} of {len(st.session_state.saved_image_paths)}", width=500)

            col1, col2, col3 = st.columns([1, 10, 1])
            with col1:
                if st.button("Previous") and st.session_state.image_index > 0:
                    st.session_state.image_index -= 1
            with col3:
                if st.button("Next") and st.session_state.image_index < len(st.session_state.saved_image_paths) - 1:
                    st.session_state.image_index += 1

            # Display cached results if present
            if current_image_path in st.session_state.results_cache:
                cached_results = st.session_state.results_cache[current_image_path]
                category = cached_results["category"]
                st.success(f"**Classification Result:** {category}")
                print(f"Using cached results for {current_image_path}")
            else:
                with st.spinner("Classifying image..."):
                    print(f"No cached results found for {current_image_path}")
                    classification_results = classification_model(image)
                    category = classification_results[0].names[classification_results[0].probs.top1]

                    st.success(f"**Classification Result:** {category}")
                    print(f"[INFO] Inference information about file: {current_image_path}")

            _ = """ 
                Do not cache segmentation results, it borks website
                Pass to segmentation model only if images need blur, otherwise skip
            """

            if category == 'porn' or category == 'hentai':
                with st.spinner("Detecting explicit regions..."):
                    segmentation_results = segmentation_model(image, agnostic_nms=True, retina_masks=True)

                boxes = segmentation_results[0].boxes.xyxy.cpu().tolist()
                clss = segmentation_results[0].boxes.cls.cpu().tolist()
                confs = segmentation_results[0].boxes.conf.cpu().tolist()

                _ = """
                    Copy of the image for drawing segmentation masks.
                    Prevents segmentation mask's color from being picked up during the blurring process, results in a clean blur.
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
                    obj = image_with_blur.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
                    blur_obj = obj.filter(ImageFilter.GaussianBlur(radius=blur_ratio))
                    if blur_obj.mode == 'RGBA':
                        blur_obj = blur_obj.convert('RGB')
                    image_with_blur.paste(blur_obj, (int(box[0]), int(box[1])))

            if category == 'porn' or category == 'hentai':
                if st.checkbox("Blur explicit regions", True):
                    left_co, cent_co, last_co = st.columns(3)
                    with cent_co:
                        st.image(image_with_blur, caption="Image with Blurred NSFW Regions", use_container_width=True)
                else:
                    left_co, cent_co, last_co = st.columns(3)
                    with cent_co:
                        st.image(image_with_boxes, caption="Image with Segmentation Masks", use_container_width=True)
        else:
            st.warning("No images to display.")

st.markdown("---")
st.markdown("ℹ️ **Instructions:** Upload an image or video to classify and detect explicit content. Use the slider to adjust the blur intensity.")
st.write("---")

st.markdown("#### Feedback Form")
with st.form(key='feedback_form', clear_on_submit=True):
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
            
            feedbacks = load_feedbacks()
            feedbacks.append(new_feedback)
            save_feedbacks(feedbacks)
            st.success('Thanks for your feedback!')
        else:
            st.warning('Please fill out all fields.')

# Call the admin panel
admin_panel()

# A small button to link to the Github repo
st.write("---")
st.markdown("ℹ️ Please note that I collect uploaded images and videos for further analysis and may be used for re-training. However, the user remains anonymous.")
st.link_button("🐙 View on GitHub", "https://github.com/Forenche/nsfw_detector_annotator")