import streamlit as st
import json
import os
import uuid
from pdf2image import convert_from_path
import sys
import easyocr
from ultralytics import YOLO
import cv2
import logging

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Aadhaar Validator",
    page_icon="🔎",
    layout="centered"
)

# --- 2. Configuration ---
YOLO_MODEL_PATH = 'best.pt' # Assumes it's in the same folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- 3. IMPORTANT: Import Your Script's Function ---
# This line imports the actual function from your other file.
try:
    from standalone_script import extract_fields_with_detection
except ImportError:
    st.error("Fatal Error: standalone_script.py not found.")
    st.stop()
except Exception as e:
    st.error(f"Error importing script: {e}")
    st.stop()


# --- 4. Cached Model Loading (The BIG Fix) ---
# This function will run ONCE and cache the models in memory
@st.cache_resource
def load_models():
    st.write("Loading AI models (this happens once)...")
    try:
        logging.info("Loading EasyOCR reader...")
        reader = easyocr.Reader(['en'], gpu=False)
        logging.info("Loading YOLO detection model...")
        detector = YOLO(YOLO_MODEL_PATH)
        logging.info("Models loaded successfully.")
        return detector, reader
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Load the models
detector, reader = load_models()

if detector is None or reader is None:
    st.error("Models failed to load. The app cannot continue.")
    st.stop()
else:
    st.success("Models loaded successfully!")


# --- 5. Main Webpage UI ---
st.title("Aadhaar Validator 🔎")
st.write("Upload an Aadhaar image or PDF to extract details and validate the number format.")

uploaded_file = st.file_uploader(
    "Choose an image or PDF", 
    type=["jpg", "jpeg", "png", "pdf"]
)

if uploaded_file is not None:
    if st.button("Validate Document"):
        with st.spinner('Processing...'):
            
            # Save the uploaded file temporarily
            ext = os.path.splitext(uploaded_file.name)[1]
            temp_filename = str(uuid.uuid4())
            temp_filepath_raw = os.path.join(UPLOAD_FOLDER, temp_filename + ext)
            
            with open(temp_filepath_raw, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            image_path_to_process = ""

            # --- Handle PDF vs Image ---
            if ext.lower() == '.pdf':
                try:
                    images = convert_from_path(temp_filepath_raw, first_page=1, last_page=1)
                    if images:
                        image_path_to_process = os.path.join(UPLOAD_FOLDER, temp_filename + '.jpg')
                        images[0].save(image_path_to_process, 'JPEG')
                    else:
                        st.error("Could not read PDF file.")
                except Exception as e:
                    st.error(f"PDF processing failed. Do you have Poppler installed? Error: {e}")
            else:
                image_path_to_process = temp_filepath_raw

            # --- 6. Run Extraction Directly (No Subprocess) ---
            if image_path_to_process:
                try:
                    # Call the function directly with the loaded models
                    result, doc_type = extract_fields_with_detection(image_path_to_process, detector, reader)
                    
                    st.success("Extraction Complete!")
                    
                    aadhaar_num = result.get('Aadhaar_Number', 'Not found')
                    is_valid = "✅ VALID" in aadhaar_num
                    is_invalid = "❌ INVALID" in aadhaar_num

                    if is_valid:
                        st.markdown(f"### <span style='color:green;'>Number Format: ✅ Valid</span>", unsafe_allow_html=True)
                    elif is_invalid:
                        st.markdown(f"### <span style='color:red;'>Number Format: ❌ Invalid</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"### Number Format: Unknown")
                    
                    st.warning("""
                        **Disclaimer:** This check only validates the number format. 
                        It **does not** guarantee the card is real or belongs to the person.
                    """)
                    
                    result['Document_Type'] = doc_type
                    st.table(result)
                
                except Exception as e:
                    st.error(f"An error occurred during extraction: {e}")