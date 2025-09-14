import cv2
import numpy as np
import easyocr
from ultralytics import YOLO

reader = easyocr.Reader(['en'])
yolo_model = YOLO("detector/car_plate_detection_yolov11n.pt")

def enhanced_preprocess(crop, target_height=60):

    # Convert to grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # Resize to target height (keep aspect ratio)
    h, w = gray.shape
    scale = target_height / h
    new_width = int(w * scale)
    resized = cv2.resize(gray, (new_width, target_height), interpolation=cv2.INTER_CUBIC)
    
    # Noise reduction & contrast enhancement
    filtered = cv2.bilateralFilter(resized, 11, 17, 17)
    enhanced = cv2.convertScaleAbs(filtered, alpha=1.5, beta=20)
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    cleaned = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
    
    # Sharpening
    kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(cleaned, -1, kernel_sharp)
    
    # Anti-aliasing: upscale then downscale
    h2, w2 = sharpened.shape
    upscaled = cv2.resize(sharpened, (w2*2, h2*2), interpolation=cv2.INTER_LINEAR)
    anti_aliased = cv2.resize(upscaled, (w2, h2), interpolation=cv2.INTER_AREA)
    
    return anti_aliased


def normalize_plate(text):
    """Generic normalization for OCR text"""
    if not text:
        return ""

    text = text.replace(" ", "").replace("-", "").replace(".", "").upper()
    return text



def is_valid_plate(text):
    """Generic validation: letters & digits, 3-10 chars"""
    import re
    return bool(re.fullmatch(r'[A-Z0-9]{3,10}', text))

def recognize_plate(image):
    """Detect and recognize license plates in any country"""
    try:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = yolo_model(image_bgr)

        detected_plates = []
        best_crop = None
        best_confidence = 0

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confidences = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else [1.0] * len(boxes)

            for box, detection_conf in zip(boxes, confidences):
                x1, y1, x2, y2 = map(int, box)
                crop = image_bgr[y1:y2, x1:x2]


                if detection_conf > best_confidence:
                    best_confidence = detection_conf
                    best_crop = crop.copy()


                processing_methods = [
                    enhanced_preprocess(crop),
                    cv2.convertScaleAbs(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), alpha=2.0, beta=30),
                    cv2.adaptiveThreshold(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 
                                          255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                ]

                all_candidates = []

                for processed_crop in processing_methods:
                    ocr_configs = [
                        {"allowlist": 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', "width_ths": 0.4, "height_ths": 0.4},
                        {"allowlist": 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', "width_ths": 0.2, "height_ths": 0.2},
                        {"width_ths": 0.6, "height_ths": 0.6}
                    ]

                    for config in ocr_configs:
                        try:
                            results_ocr = reader.readtext(processed_crop, **config)
                            results_ocr = sorted(results_ocr, key=lambda x: x[0][0][0]) 

                            parts = []
                            total_confidence = 0
                            count = 0

                            for (bbox, text, prob) in results_ocr:
                                if prob > 0.1:
                                    clean_text = normalize_plate(text)
                                    if clean_text:
                                        parts.append(clean_text)
                                        total_confidence += prob
                                        count += 1

                            if parts:
                                combined_text = "".join(parts)
                                avg_confidence = total_confidence / count if count > 0 else 0

       
                                if is_valid_plate(combined_text):
                                    avg_confidence += 0.2

                                all_candidates.append((combined_text, avg_confidence))

                        except Exception:
                            continue


                if all_candidates:
                    unique_candidates = {}
                    for text, conf in all_candidates:
                        if text not in unique_candidates or conf > unique_candidates[text]:
                            unique_candidates[text] = conf

                    best_text = max(unique_candidates.items(), key=lambda x: x[1])[0]
                    if best_text not in detected_plates:
                        detected_plates.append(best_text)


        if detected_plates:
            if len(detected_plates) == 1:
                plate_text = f"License Plate: {detected_plates[0]}"
            else:
                plate_text = f"License Plates: {', '.join(detected_plates)}"
        else:
            plate_text = "No license plate detected"


        if best_crop is not None:
            crop_rgb = cv2.cvtColor(best_crop, cv2.COLOR_BGR2RGB)
            height, width = crop_rgb.shape[:2]
            if height < 100:  
                scale = 100 / height
                new_width = int(width * scale)
                crop_rgb = cv2.resize(crop_rgb, (new_width, 100), interpolation=cv2.INTER_CUBIC)
            return plate_text, crop_rgb
        else:
            placeholder = np.zeros((100, 300, 3), dtype=np.uint8)
            placeholder.fill(128)
            return plate_text, placeholder

    except Exception as e:
        error_placeholder = np.zeros((100, 300, 3), dtype=np.uint8)
        error_placeholder.fill(128)
        return f"Error processing image: {str(e)}", error_placeholder
