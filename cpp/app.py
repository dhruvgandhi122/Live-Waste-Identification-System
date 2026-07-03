from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os
import time
import base64

# CREATE APP FIRST
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.before_request
def log_request():
    print(f"Request: {request.method} {request.url}")

# LOAD MODEL ONCE
model = YOLO("yolov8n.pt")

STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

# SERVE FRONTEND
@app.route("/")
def home():
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    return send_file(file_path)

# SCAN ENDPOINT
@app.route("/scan", methods=["POST", "OPTIONS"])
def scan():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    print("Request received at /scan")
    if "image" not in request.files:
        print("No image in request")
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img_bytes = file.read()
    np_img = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    results = model(img, conf=0.25)

    boxes = results[0].boxes
    names = model.names

    all_detected = []
    biodegradable = []
    non_biodegradable = []

    if boxes is not None:
        for cls_id in boxes.cls.tolist():
            label = names[int(cls_id)]
            all_detected.append(label)

            if label in ["banana", "apple", "food", "orange", "mango", "fruit_peel", "vegetable_peel", "leftover_food", "bread", "rice", "chapati", "leaves", "grass", "flowers", "wood", "paper", "newspaper", "cardboard", "tissue", "paper_cup", "paper_plate", "cotton", "jute", "cloth", "tea_waste", "coffee_grounds", "egg_shell", "nutshell", "corn_cob", "sugarcane_bagasse", "straw", "hay", "plant_stem", "tree_bark", "sawdust", "compost", "manure", "biowaste", "garden_waste", "food_scraps", "vegetable_scraps", "fruit_scraps", "wooden_stick", "wooden_spoon", "person"]:
                biodegradable.append(label)
            else:
                non_biodegradable.append(label)

    annotated_img = results[0].plot(line_width=2)

    # Encode to Base64 (No file save to prevent auto-reload)
    _, buffer = cv2.imencode('.jpg', annotated_img)
    img_str = base64.b64encode(buffer).decode('utf-8')
    image_url = f"data:image/jpeg;base64,{img_str}"

    return jsonify({
        "image_url": image_url,
        "all_detected": list(set(all_detected)),
        "biodegradable": list(set(biodegradable)),
        "non_biodegradable": list(set(non_biodegradable))
    })

if __name__ == "__main__":
    app.run(port=5502, debug=False)


