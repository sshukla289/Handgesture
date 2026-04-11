import csv
import os
import time
from pathlib import Path
from threading import Lock

import cv2
import joblib
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

from hand_tracking import create_hand_landmarker, detect_hands, draw_hand_landmarks
from train_model import resolve_training_data_dir, train_model

app = Flask(__name__)

_camera_lock = Lock()
_model_lock = Lock()
_model_cache = {"model": None, "label_encoder": None}
_collection_lock = Lock()
_collection_state = {
    "active": False,
    "label": "",
    "file_label": "",
    "rows": [],
    "count": 0,
    "last_message": "Collection idle.",
}


PAGE_HTML = Path("templates/index.html").read_text(encoding="utf-8")


def load_model():
    with _model_lock:
        if _model_cache["model"] is None or _model_cache["label_encoder"] is None:
            model_path = os.path.join("model", "svm_model.pkl")
            encoder_path = os.path.join("model", "label_encoder.pkl")

            if not os.path.exists(model_path) or not os.path.exists(encoder_path):
                return None, None

            _model_cache["model"] = joblib.load(model_path)
            _model_cache["label_encoder"] = joblib.load(encoder_path)

        return _model_cache["model"], _model_cache["label_encoder"]


def get_model_metadata():
    training_data_dir = resolve_training_data_dir(None)
    dataset_title = "Custom browser samples"
    if training_data_dir == "datasets_leapgestrecog":
        dataset_title = "Imported leapGestRecog"

    model_path = os.path.join("model", "svm_model.pkl")
    encoder_path = os.path.join("model", "label_encoder.pkl")
    labels = []

    if os.path.exists(encoder_path):
        try:
            label_encoder = joblib.load(encoder_path)
            labels = [str(label) for label in label_encoder.classes_]
        except Exception:
            labels = []

    sample_count = 0
    csv_count = 0
    if os.path.isdir(training_data_dir):
        csv_files = [file for file in os.listdir(training_data_dir) if file.endswith(".csv")]
        csv_count = len(csv_files)
        for file_name in csv_files:
            file_path = os.path.join(training_data_dir, file_name)
            try:
                with open(file_path, "r", newline="") as csv_file:
                    sample_count += sum(1 for _ in csv_file)
            except OSError:
                continue

    return {
        "model_available": os.path.exists(model_path) and os.path.exists(encoder_path),
        "training_data_dir": training_data_dir,
        "dataset_title": dataset_title,
        "label_count": len(labels),
        "labels": labels,
        "training_sample_count": sample_count,
        "csv_file_count": csv_count,
    }


def safe_label_name(label):
    sanitized = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in label.strip())
    return sanitized.strip("_") or "gesture"


def start_collection(label):
    clean_label = label.strip()
    if not clean_label:
        return False, "Gesture label is required."

    with _collection_lock:
        if _collection_state["active"]:
            return False, f"Collection is already running for '{_collection_state['label']}'. Stop it first."

        _collection_state["active"] = True
        _collection_state["label"] = clean_label
        _collection_state["file_label"] = safe_label_name(clean_label)
        _collection_state["rows"] = []
        _collection_state["count"] = 0
        _collection_state["last_message"] = f"Collecting '{clean_label}'. Keep your gesture visible in the camera."

    return True, _collection_state["last_message"]


def stop_collection():
    with _collection_lock:
        if not _collection_state["active"]:
            return "No active collection session."

        label = _collection_state["label"]
        file_label = _collection_state["file_label"]
        rows = list(_collection_state["rows"])
        count = _collection_state["count"]

        _collection_state["active"] = False
        _collection_state["label"] = ""
        _collection_state["file_label"] = ""
        _collection_state["rows"] = []
        _collection_state["count"] = 0

    if not rows:
        message = f"Collection stopped for '{label}'. No samples were captured."
    else:
        os.makedirs("datasets", exist_ok=True)
        file_path = os.path.join("datasets", f"{file_label}_data.csv")
        mode = "a" if os.path.exists(file_path) else "w"
        with open(file_path, mode, newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(rows)
        message = f"Saved {count} samples for '{label}' to {file_path}."

    with _collection_lock:
        _collection_state["last_message"] = message

    return message


def collection_status_payload():
    with _collection_lock:
        return {
            "active": _collection_state["active"],
            "label": _collection_state["label"],
            "count": _collection_state["count"],
            "last_message": _collection_state.get("last_message", "Collection idle."),
        }


def train_model_for_ui():
    summary = train_model()
    if summary["accuracy"] is None:
        return (
            f"Model training completed with {summary['samples']} samples across "
            f"{summary['classes']} labels. Add more data for an accuracy score."
        )
    return (
        f"Model training completed with {summary['accuracy'] * 100:.2f}% accuracy "
        f"using {summary['samples']} samples across {summary['classes']} labels."
    )


def frame_with_message(message):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    lines = message.split("\n")
    y = 200
    for line in lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y += 36
    success, buffer = cv2.imencode(".jpg", frame)
    if success:
        return buffer.tobytes()
    return b""


def generate_detection_stream():
    with _camera_lock:
        model, label_encoder = load_model()

        hands = create_hand_landmarker(num_hands=2)
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            error_frame = frame_with_message(
                "Unable to access webcam.\n"
                "Close other camera apps.\n"
                "Also allow camera access for desktop apps in Windows settings."
            )
            hands.close()
            while True:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + error_frame + b"\r\n")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                detected_hands = detect_hands(hands, frame, int(time.time() * 1000))

                for idx, detected_hand in enumerate(detected_hands):
                    landmarks = detected_hand["flattened"]

                    with _collection_lock:
                        if _collection_state["active"]:
                            _collection_state["rows"].append([_collection_state["label"]] + landmarks)
                            _collection_state["count"] += 1

                    draw_hand_landmarks(frame, detected_hand["landmarks"])

                    if model is not None and label_encoder is not None and len(landmarks) == model.n_features_in_:
                        input_data = np.array(landmarks, dtype=float).reshape(1, -1)
                        prediction = model.predict(input_data)
                        label = label_encoder.inverse_transform(prediction)[0]
                        y_position = 60 + idx * 60
                        color = (0, 255 - idx * 100, 255)
                        cv2.putText(
                            frame,
                            f"Hand {idx + 1}: {label}",
                            (10, y_position),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            color,
                            3,
                        )

                with _collection_lock:
                    if _collection_state["active"]:
                        cv2.putText(
                            frame,
                            f"Collecting '{_collection_state['label']}' - {_collection_state['count']} samples",
                            (10, frame.shape[0] - 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )

                if model is None or label_encoder is None:
                    cv2.putText(
                        frame,
                        "No trained model loaded. Collect data or train the model.",
                        (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

                success, buffer = cv2.imencode(".jpg", frame)
                if not success:
                    continue

                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
        finally:
            cap.release()
            hands.close()


@app.get("/")
def index():
    return render_template_string(PAGE_HTML)


@app.get("/video_feed")
def video_feed():
    return Response(generate_detection_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.post("/train")
def train():
    try:
        message = train_model_for_ui()
    except Exception as exc:
        return jsonify({"message": f"Training failed: {exc}"}), 400

    with _model_lock:
        _model_cache["model"] = None
        _model_cache["label_encoder"] = None
    return jsonify({"message": message})


@app.get("/collection/status")
def collection_status():
    return jsonify(collection_status_payload())


@app.post("/collection/start")
def collection_start():
    payload = request.get_json(silent=True) or {}
    success, message = start_collection(payload.get("label", ""))
    status_code = 200 if success else 400
    return jsonify({"message": message, **collection_status_payload()}), status_code


@app.post("/collection/stop")
def collection_stop():
    message = stop_collection()
    return jsonify({"message": message, **collection_status_payload()})


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
