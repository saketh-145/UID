#!/usr/bin/env python3
"""
General-purpose script for Aadhaar document extraction.
This file is a MODULE, intended to be imported by webapp.py.
It does not run on its own.
"""
import json 
import os
import sys
import cv2
import argparse
import logging
from ultralytics import YOLO
import easyocr
import re
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- (Keep all your helper functions: d, p, inv, verhoeff_checksum, verhoeff_validate) ---
# Dihedral Group D5 Multiplication Table (d)
d = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
)
# Permutation Table (p)
p = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8)
)
# Inverse Table (inv)
inv = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)

def verhoeff_checksum(number):
    c = 0
    for i, digit_char in enumerate(reversed(number)):
        try:
            digit = int(digit_char)
        except ValueError:
            return -1 
        c = d[c][p[i % 8][digit]]
    return inv[c]

def verhoeff_validate(number):
    c = 0
    for i, digit_char in enumerate(reversed(number)):
        try:
            digit = int(digit_char)
        except ValueError:
            return False 
        c = d[c][p[i % 8][digit]]
    return c == 0

# --- (Keep your parsing, preprocessing, and OCR functions) ---
def parse_aadhaar_fields(text, field_name):
    if not text:
        return "Not found"
    text = text.replace("|", "1").replace("!", "1").replace("l", "1").replace("I", "1")
    text_clean = text.strip()
    if field_name == "0":
        text_clean = re.sub(r'[OoQSsZBG]', lambda m: {'o':'0', 'O':'0', 'Q':'0', 'S':'5', 's':'5', 'Z':'2', 'B':'8', 'G':'6'}.get(m.group(0), m.group(0)), text_clean)
        digits = re.sub(r'\D', '', text_clean)
        if len(digits) >= 11:
            aadhaar = digits[:12] if len(digits) >= 12 else digits
            if len(aadhaar) == 12:
                is_valid = verhoeff_validate(aadhaar)
                validation_status = "✅ VALID" if is_valid else "❌ INVALID"
                logger.debug(f"Aadhaar Checksum: {aadhaar} is {validation_status}")
                return f"{aadhaar[0:4]} {aadhaar[4:8]} {aadhaar[8:12]} ({validation_status} - OCR Found 12)"
            elif len(aadhaar) == 11:
                calculated_checksum = verhoeff_checksum(aadhaar)
                if calculated_checksum != -1:
                    full_aadhaar = aadhaar + str(calculated_checksum)
                    validation_status = "✅ VALID" 
                    logger.debug(f"Aadhaar Checksum: {aadhaar} completed to {full_aadhaar}")
                    return f"{full_aadhaar[0:4]} {full_aadhaar[4:8]} {full_aadhaar[8:12]} ({validation_status} - OCR Found 11, Calculated Checksum)"
                else:
                    return f"Not found (11 digits: {aadhaar} - Invalid digits present)"
            else:
                return aadhaar
        return f"Not found (found {len(digits)} digits: {digits})"
    elif field_name == "1":
        text_upper = text_clean.upper().replace(" ", "").replace("-", "/")
        match = re.search(r'(\d{2})\/(\d{2})\/(\d{4})', text_upper)
        if match:
            return f"{match.group(1)}/{match.group(2)}/{match.group(3)}"
        if "YOB" in text_upper or "YEAROFBIRTH" in text_upper:
            match = re.search(r'(?:YOB|YEAROFBIRTH)[\s:\/]*(\d{4})', text_upper)
            if match:
                return f"YoB: {match.group(1)}"
        digits = re.sub(r'\D', '', text_clean)
        if len(digits) >= 4:
            for i in range(len(digits) - 3):
                year = digits[i:i+4]
                if year.startswith('19') or year.startswith('20'):
                    try:
                        year_int = int(year)
                        if 1900 <= year_int <= 2025:
                            return f"YoB: {year}"
                    except ValueError:
                        continue
        return f"Not found (raw: {text_clean})"
    elif field_name == "2":
        text_lower = text_clean.lower().replace(" ", "")
        gender_map = {
            "male": "MALE", "nale": "MALE", "wale": "MALE", "hale": "MALE", "maie": "MALE",
            "female": "FEMALE", "fenale": "FEMALE", "fomale": "FEMALE", "femal": "FEMALE", "femae": "FEMALE", "gemid": "FEMALE" 
        }
        for key, value in gender_map.items():
            if key in text_lower:
                return value
        if "we" in text_lower:
             return "MALE"
        return f"Not found (raw: {text_clean})"
    elif field_name == "3":
        text_clean = re.sub(r'(DOB|YOB|NoB|Year of Birth|Birth|Name)', '', text_clean, flags=re.IGNORECASE)
        text_clean = re.sub(r'[^A-Za-z\s]', '', text_clean)
        words = text_clean.split()
        name_words = [w for w in words if len(w) > 1]
        final_name = " ".join(name_words).upper().strip()
        if final_name:
            return final_name
        return "Not found"
    return text_clean

def preprocess_for_easyocr(image, box, field_label):
    x1, y1, x2, y2 = map(int, box)
    pad = 15
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(image.shape[1], x2 + pad)
    y2 = min(image.shape[0], y2 + pad)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return []
    h, w = crop.shape[:2]
    min_h, min_w = 100, 200
    scale = max(4.0, min_h / h if h < min_h else 1.0, min_w / w if w < min_w else 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized
    preprocessed_images = []
    preprocessed_images.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    preprocessed_images.append(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    preprocessed_images.append(cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR))
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessed_images.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))
    try:
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        _, denoised_otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(cv2.cvtColor(denoised_otsu, cv2.COLOR_GRAY2BGR))
    except cv2.error:
        preprocessed_images.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))
    return preprocessed_images

def extract_text_easyocr(crops_list, field_label, reader):
    if not crops_list:
        return ""
    all_texts = []
    for i, crop in enumerate(crops_list):
        if crop is None or crop.size == 0:
            continue
        results_word = reader.readtext(crop, detail=1, paragraph=False)
        if results_word:
            texts = [r[1] for r in results_word]
            confs = [r[2] for r in results_word]
            if texts:
                concatenated = " ".join(texts)
                avg_conf = sum(confs) / len(confs)
                all_texts.append((concatenated, avg_conf, len(concatenated)))
        if field_label in ["0", "1"]:
            results_num = reader.readtext(crop, detail=1, paragraph=True, allowlist='0123456789/ ')
            for result in results_num:
                if len(result) == 3 and result[2] > 0.1:
                    all_texts.append((result[1], result[2], len(result[1])))
        if field_label == "2":
            try:
                results_gender = reader.readtext(crop, detail=1, paragraph=True, allowlist='AaEeFfIiLlMmNnSsTtRrGg ')
                for result in results_gender:
                    if len(result) == 3 and result[2] > 0.1:
                        all_texts.append((result[1], result[2], len(result[1])))
            except Exception:
                pass 
    if not all_texts:
        return ""
    unique_texts = {}
    for text, conf, length in all_texts:
        clean_text = text.strip()
        if not clean_text: continue
        if clean_text not in unique_texts or conf > unique_texts[clean_text][0]:
            unique_texts[clean_text] = (conf, length)
    all_texts = [(text, conf, length) for text, (conf, length) in unique_texts.items()]
    if field_label in ["0", "1"]:
        number_texts = [(t, c, l) for (t, c, l) in all_texts if any(char.isdigit() for char in t)]
        if number_texts:
            full_date_match = next((t for t, c, l in number_texts if re.match(r'\d{2}/\d{2}/\d{4}', t.replace(" ", ""))), None)
            if full_date_match:
                return full_date_match
            best = max(number_texts, key=lambda item: len(re.sub(r'\D', '', item[0])))
            return best[0]
    elif field_label == "3":
        if all_texts:
            best = max(all_texts, key=lambda x: (x[2], x[1])) 
            return best[0]
    elif field_label == "2":
        if all_texts:
            best = max(all_texts, key=lambda x: x[1])
            return best[0]
    if all_texts:
        best = max(all_texts, key=lambda x: (x[2], x[1]))
        return best[0]
    return ""

def extract_uid_from_text(text_data):
    uid_pattern = r'\b\d{4}\s?\d{4}\s?\d{4}\b'
    matches = re.findall(uid_pattern, text_data)
    if matches:
        return re.sub(r'\s+', '', matches[0])
    return None

def extract_name_from_text(text_data):
    name_patterns = [r'(?:Name|नाम)[:\s]+([A-Za-z\s]+)', r'(?:[T|t]o\s+)([A-Za-z\s]+)', r'[A-Z][A-Z\s]*']
    for pattern in name_patterns:
        matches = re.search(pattern, text_data)
        if matches and len(matches.group(1).strip()) > 2:
            return matches.group(1).strip()
    return None


# ----------------------------------------------------------------------
# 4. MAIN EXTRACTION PIPELINE (This is the function we will import)
# ----------------------------------------------------------------------

def extract_fields_with_detection(image_path, detector, reader):
    """Extract document fields using detection model and robust OCR"""
    extracted_data = {
        'Aadhaar_Number': "Not found",
        'DOB': "Not found",
        'Gender': "Not found",
        'Name': "Not found",
    }
    doc_type = "Aadhaar" 
    
    field_names_map = {0: "Aadhaar_Number", 1: "DOB", 2: "Gender", 3: "Name"}
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not read image: {image_path}")
            return extracted_data, doc_type

        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image
        
        results = detector.predict(source=image_path, save=False, conf=0.25, verbose=False) 
        
        if not results or not results[0].boxes:
            logger.warning("No fields detected by YOLO model.")
        else:
            for box_data in results[0].boxes.data.tolist():
                x1, y1, x2, y2, confidence, class_id = box_data
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                label = int(class_id)
                parsed_field_name = field_names_map.get(label, "Unknown")
                
                logger.debug(f"Detected {parsed_field_name} (Conf: {confidence:.2f}) at [{x1}, {y1}, {x2}, {y2}]")

                crops_list = preprocess_for_easyocr(image, (x1, y1, x2, y2), str(label))
                raw_text = extract_text_easyocr(crops_list, str(label), reader)
                parsed_text = parse_aadhaar_fields(raw_text, str(label))

                if label in field_names_map:
                    if extracted_data[parsed_field_name] == "Not found" or "Not found" not in parsed_text:
                        extracted_data[parsed_field_name] = parsed_text
        
        needs_fallback = any(extracted_data[k].startswith("Not found") or extracted_data[k] in ["YoB: 1954", "YoB: 1931", "DM THT"] 
                             for k in ['Gender', 'DOB', 'Name', 'Aadhaar_Number'])

        if needs_fallback:
            logger.info("Some fields failed YOLO/OCR. Running general full-text OCR fallback...")
            
            full_text_results = reader.readtext(image_gray, detail=0, paragraph=True)
            full_text = ' '.join(full_text_results)
            
            if extracted_data['Aadhaar_Number'].startswith("Not found"):
                uid_match = extract_uid_from_text(full_text)
                if uid_match:
                    extracted_data['Aadhaar_Number'] = parse_aadhaar_fields(uid_match, "0")
            if extracted_data['Name'] in ["Not found", "DM THT"]:
                name_match = extract_name_from_text(full_text)
                if name_match:
                    extracted_data['Name'] = parse_aadhaar_fields(name_match, "3")
            if extracted_data['DOB'] in ["Not found", "YoB: 1954", "YoB: 1931"]: 
                parsed_dob = parse_aadhaar_fields(full_text, "1")
                if parsed_dob not in ["Not found"]:
                    extracted_data['DOB'] = parsed_dob
            if extracted_data['Gender'].startswith("Not found"):
                parsed_gender = parse_aadhaar_fields(full_text, "2")
                if parsed_gender != "Not found":
                    extracted_data['Gender'] = parsed_gender
        
        return extracted_data, doc_type
        
    except Exception as e:
        logger.error(f"Error in field extraction: {str(e)}")
        logger.error(traceback.format_exc())
        return extracted_data, "Error"

# ----------------------------------------------------------------------
# 6. MAIN FUNCTION - MODIFIED (This is just a placeholder now)
# ----------------------------------------------------------------------

def main():
    """
    This script is not meant to be run directly anymore. 
    Use webapp.py
    """
    print("This script is a module and should be run via webapp.py")
    print("It does not load models or run extraction on its own.")

if __name__ == "__main__":
    main()