from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import os
import re

# ✅ New: PaddleOCR Import
from paddleocr import PaddleOCR

app = Flask(__name__)
CORS(app)

latest_detected_text = ""

# ✅ Initialize PaddleOCR (English + Angle classifier)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Load YOLOv8 model
model = YOLO("best.pt")  # Replace with your trained model path

UPLOAD_FOLDER = "/tmp/uploads"
OUTPUT_FOLDER = "/tmp/outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def extract_significant_text(ocr_text):
    """Extracts meaningful alphanumeric sequences from OCR output."""
    words = re.findall(r'[A-Z0-9]{5,}', ocr_text.upper())  # Ensure all caps
    return " ".join(words)


def detect_license_plate_from_frame(frame):
    global latest_detected_text
    results = model(frame)[0]
    extracted_texts = []
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cropped_plate = frame[y1:y2, x1:x2]
        if cropped_plate is None or cropped_plate.size == 0:
            continue

        # ✅ Convert cropped image to BGR (PaddleOCR needs file path or ndarray)
        result = ocr.ocr(cropped_plate, cls=True)

        for line in result:
            for (_, (text, conf)) in line:
                significant_text = extract_significant_text(text)
                if significant_text:
                    extracted_texts.append(significant_text)
                    cv2.putText(frame, significant_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    latest_detected_text = " ".join(extracted_texts) if extracted_texts else "No valid text found"
    return frame, latest_detected_text


@app.route('/detect', methods=['POST'])
def detect_license_plate():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    
    try:
        file.save(image_path)
        image = cv2.imread(image_path)
        if image is None:
            return jsonify({"error": "Could not read image"}), 500

        processed_image, detected_text = detect_license_plate_from_frame(image)

        # Save processed image
        output_path = os.path.join(OUTPUT_FOLDER, f"output_{file.filename}")
        cv2.imwrite(output_path, processed_image)

        return jsonify({
            "ocr_text": detected_text,
            "image_url": f"http://127.0.0.1:5000/output/{file.filename}"
        })

    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

@app.route('/output/<filename>')
def get_output_image(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, f"output_{filename}"), mimetype='image/jpeg')

def generate_frames():
    global latest_detected_text
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use CAP_DSHOW for faster capture on Windows
    cap.set(cv2.CAP_PROP_FPS, 15)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Resize to improve speed
        frame = cv2.resize(frame, (640, 480))

        processed_frame, detected_text = detect_license_plate_from_frame(frame)
        latest_detected_text = detected_text

        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()

        # Yield for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/latest_text')
def latest_text():
    return jsonify({"detected_text": latest_detected_text})

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
