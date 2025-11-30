from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import torch
from ultralytics import YOLO
import base64
import io
from PIL import Image
import numpy as np
from config import Config


# -----------------------------
# LOAD YOLO MODEL
# -----------------------------
print("Loading YOLOv8 model...")
model = YOLO("models/yolov8n.pt")   # <---- Your downloaded model
print("YOLO model loaded successfully.")


# -----------------------------
# GLOBAL VARIABLES
# -----------------------------
latest_alert = {
    "animal_detected": False,
    "animal_type": None,
    "location": None,
    "confidence": 0.0
}

registered_cameras = {}
alert_subscribers = []


# -----------------------------
# FLASK APP SETUP
# -----------------------------
app = Flask(__name__)
CORS(app)
app.config.from_object(Config)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------
# ROUTES
# -----------------------------

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "Backend running!"})


@app.route('/register/camera', methods=['POST'])
def register_camera():
    data = request.get_json()
    camera_id = data.get("camera_id")
    location = data.get("location", "Unknown")
    secret = data.get("secret")

    if secret != Config.CAMERA_SECRET:
        return jsonify({"status": "error", "message": "Invalid secret"}), 401

    registered_cameras[camera_id] = {"location": location}
    return jsonify({"status": "success", "message": "Camera registered"})


@app.route('/camera/detect', methods=['POST'])
def camera_detect():
    try:
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({"status": "error", "message": "Image missing"}), 400

        camera_id = data.get("camera_id", "Unknown")
        base64_img = data["image"]

        # Decode Base64 image
        image_bytes = base64.b64decode(base64_img)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(img)

        # -----------------------------
        # RUN YOLOV8 DETECTION
        # -----------------------------
        results = model(img_array)

        detected_animals = []
        dangerous_animals = Config.DANGEROUS_ANIMALS

        for box in results[0].boxes:
            cls_id = int(box.cls)
            confidence = float(box.conf)
            label = results[0].names[cls_id]

            detected_animals.append((label, confidence))

        # Determine if a dangerous animal was detected
        for animal, conf in detected_animals:
            if animal.lower() in dangerous_animals:
                # Store latest alert
                global latest_alert
                latest_alert = {
                    "animal_detected": True,
                    "animal_type": animal,
                    "location": registered_cameras.get(camera_id, {}).get("location", "Unknown"),
                    "confidence": conf
                }

                send_alert_to_subscribers(latest_alert)

                return jsonify({
                    "status": "success",
                    "dangerous": True,
                    "animal": animal,
                    "confidence": conf
                })

        # If no dangerous animals detected
        latest_alert["animal_detected"] = False

        return jsonify({
            "status": "success",
            "dangerous": False,
            "animals_detected": detected_animals
        })

    except Exception as e:
        logger.error("Detection error: " + str(e))
        return jsonify({"status": "error", "message": str(e)})


@app.route('/latest-alert', methods=['GET'])
def get_latest_alert():
    return jsonify(latest_alert)


@app.route('/public/subscribe', methods=['POST'])
def subscribe_public_app():
    data = request.get_json()
    device_token = data.get("device_token")
    user_id = data.get("user_id")

    if not device_token:
        return jsonify({"status": "error", "message": "Device token missing"}), 400

    alert_subscribers.append({
        "device_token": device_token,
        "user_id": user_id
    })

    return jsonify({
        "status": "success",
        "message": "Subscribed to alerts!"
    })


# ---------------------------------------
# SEND ALERT TO SUBSCRIBER (LOG ONLY)
# ---------------------------------------
def send_alert_to_subscribers(alert):
    logger.info("🚨 ALERT SENT TO SUBSCRIBERS 🚨")
    logger.info(alert)
    # TODO: Add Firebase push notifications here


# -----------------------------
# START FLASK BACKEND
# -----------------------------
if __name__ == '__main__':
    print("🔥 Starting Animal Detection Server...")
    app.run(host="0.0.0.0", port=5000, debug=True)

