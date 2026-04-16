import csv
import os
import time
from collections import deque
from threading import Lock

import cv2
import joblib
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from hand_tracking import create_hand_landmarker, detect_hands, draw_hand_landmarks
from train_model import resolve_training_data_dir, train_model

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = False
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.jinja_env.auto_reload = False

_camera_lock = Lock()
_model_lock = Lock()
_model_cache = {"model": None, "label_encoder": None}
_metadata_lock = Lock()
_metadata_cache = None
_prediction_lock = Lock()
_prediction_state = {
    "predictions": [],
    "history": deque(maxlen=20),
    "last_signature": None,
    "last_history_ts": 0.0,
}
_collection_lock = Lock()
_collection_state = {
    "active": False,
    "label": "",
    "file_label": "",
    "rows": [],
    "count": 0,
    "last_message": "Collection idle.",
}
_training_lock = Lock()
_training_state = {
    "status": "idle",
    "message": "Training idle.",
    "last_started_at": None,
    "last_completed_at": None,
}
_training_source_options = {
    "auto": {
        "title": "Auto Select",
        "description": "Use the project's default training folder.",
        "data_dir": None,
    },
    "custom": {
        "title": "Browser Samples",
        "description": "Use gestures collected from this browser UI.",
        "data_dir": "datasets",
    },
    "imported": {
        "title": "Imported Dataset",
        "description": "Use the converted leapGestRecog landmark files.",
        "data_dir": "datasets_leapgestrecog",
    },
}


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


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


def invalidate_model_metadata_cache():
    global _metadata_cache
    with _metadata_lock:
        _metadata_cache = None


def get_dataset_title(training_data_dir):
    if training_data_dir == "datasets_leapgestrecog":
        return "Imported leapGestRecog"
    return "Custom browser samples"


def summarize_dataset_dir(data_dir, title, key):
    sample_count = 0
    csv_count = 0
    available = False

    if os.path.isdir(data_dir):
        csv_files = [file for file in os.listdir(data_dir) if file.endswith(".csv")]
        csv_count = len(csv_files)
        available = csv_count > 0
        for file_name in csv_files:
            file_path = os.path.join(data_dir, file_name)
            try:
                with open(file_path, "r", newline="") as csv_file:
                    sample_count += sum(1 for _ in csv_file)
            except OSError:
                continue

    return {
        "key": key,
        "title": title,
        "training_data_dir": data_dir,
        "sample_count": sample_count,
        "csv_file_count": csv_count,
        "available": available,
    }


def get_training_sources_metadata():
    auto_dir = resolve_training_data_dir(None)
    sources = []

    for key, source in _training_source_options.items():
        source_dir = auto_dir if source["data_dir"] is None else source["data_dir"]
        summary = summarize_dataset_dir(source_dir, source["title"], key)
        summary["description"] = source["description"]
        if key == "auto":
            summary["resolved_key"] = "imported" if auto_dir == "datasets_leapgestrecog" else "custom"
            summary["resolved_title"] = get_dataset_title(auto_dir)
        sources.append(summary)

    return sources


def resolve_requested_training_dir(source_key):
    normalized = (source_key or "auto").strip().lower()
    if normalized not in _training_source_options:
        raise ValueError("Unsupported training source.")

    option = _training_source_options[normalized]
    training_data_dir = resolve_training_data_dir(option["data_dir"])
    return normalized, training_data_dir


def get_model_metadata():
    global _metadata_cache

    with _metadata_lock:
        if _metadata_cache is not None:
            return dict(_metadata_cache)

    training_data_dir = resolve_training_data_dir(None)
    dataset_title = get_dataset_title(training_data_dir)

    model_path = os.path.join("model", "svm_model.pkl")
    encoder_path = os.path.join("model", "label_encoder.pkl")
    labels = []

    if os.path.exists(encoder_path):
        try:
            label_encoder = joblib.load(encoder_path)
            labels = [str(label) for label in label_encoder.classes_]
        except Exception:
            labels = []

    active_dataset = summarize_dataset_dir(training_data_dir, dataset_title, "active")
    training_sources = get_training_sources_metadata()

    metadata = {
        "model_available": os.path.exists(model_path) and os.path.exists(encoder_path),
        "training_data_dir": training_data_dir,
        "dataset_title": dataset_title,
        "label_count": len(labels),
        "labels": labels,
        "training_sample_count": active_dataset["sample_count"],
        "csv_file_count": active_dataset["csv_file_count"],
        "training_sources": training_sources,
        "default_training_source": "auto",
        "resolved_auto_source": "imported" if training_data_dir == "datasets_leapgestrecog" else "custom",
    }

    with _metadata_lock:
        _metadata_cache = dict(metadata)

    return metadata


def prediction_status_payload():
    with _prediction_lock:
        return {
            "predictions": [
                {
                    "hand": prediction["hand"],
                    "label": prediction["label"],
                    "confidence": prediction["confidence"],
                    "top_scores": list(prediction["top_scores"]),
                }
                for prediction in _prediction_state["predictions"]
            ],
            "history": list(_prediction_state["history"]),
        }


def prediction_output_payload():
    with _prediction_lock:
        if not _prediction_state["predictions"]:
            return {
                "label": None,
                "confidence": None,
                "timestamp": None,
            }

        lead_prediction = _prediction_state["predictions"][0]
        return {
            "label": lead_prediction["label"],
            "confidence": lead_prediction["confidence"],
            "timestamp": _prediction_state["last_history_ts"] or None,
        }


def prediction_logs_payload():
    with _prediction_lock:
        return {
            "logs": list(_prediction_state["history"]),
        }


def update_prediction_state(predictions):
    now = time.time()

    with _prediction_lock:
        _prediction_state["predictions"] = predictions

        if not predictions:
            _prediction_state["last_signature"] = None
            return

        signature = tuple((item["hand"], item["label"]) for item in predictions)
        should_log = (
            signature != _prediction_state["last_signature"]
            or now - _prediction_state["last_history_ts"] >= 1.25
        )

        if should_log:
            lead_prediction = predictions[0]
            summary = " | ".join(
                f"Hand {item['hand']}: {item['label']} ({item['confidence'] * 100:.0f}%)"
                for item in predictions
            )
            _prediction_state["history"].appendleft(
                {
                    "timestamp": now,
                    "label": lead_prediction["label"],
                    "confidence": lead_prediction["confidence"],
                    "summary": summary,
                }
            )
            _prediction_state["last_signature"] = signature
            _prediction_state["last_history_ts"] = now


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
    invalidate_model_metadata_cache()

    return message


def collection_status_payload():
    with _collection_lock:
        return {
            "active": _collection_state["active"],
            "label": _collection_state["label"],
            "count": _collection_state["count"],
            "last_message": _collection_state.get("last_message", "Collection idle."),
        }


def set_training_state(status, message):
    with _training_lock:
        _training_state["status"] = status
        _training_state["message"] = message
        if status == "training":
            _training_state["last_started_at"] = time.time()
        if status in {"completed", "failed"}:
            _training_state["last_completed_at"] = time.time()


def training_status_payload():
    with _training_lock:
        return dict(_training_state)


def stats_payload():
    metadata = get_model_metadata()
    collection = collection_status_payload()
    training = training_status_payload()

    return {
        "samples": metadata["training_sample_count"],
        "labels": metadata["label_count"],
        "model_status": "trained" if metadata["model_available"] else "not_trained",
        "training_status": training["status"],
        "training_message": training["message"],
        "current_gesture": collection["label"],
        "current_gesture_samples": collection["count"],
        "collection_active": collection["active"],
    }


def train_model_for_ui(source_key="auto"):
    _, training_data_dir = resolve_requested_training_dir(source_key)
    summary = train_model(data_dir=training_data_dir)
    dataset_title = get_dataset_title(training_data_dir)

    if summary["accuracy"] is None:
        return (
            f"Model training completed from {dataset_title} with {summary['samples']} samples "
            f"across {summary['classes']} labels. Add more data for an accuracy score."
        )
    return (
        f"Model training completed from {dataset_title} with {summary['accuracy'] * 100:.2f}% "
        f"accuracy using {summary['samples']} samples across {summary['classes']} labels."
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


def get_asset_version():
    asset_paths = [
        os.path.join("templates", "index.html"),
        os.path.join("static", "styles.css"),
        os.path.join("static", "app.js"),
    ]
    timestamps = []

    for asset_path in asset_paths:
        try:
            timestamps.append(int(os.path.getmtime(asset_path)))
        except OSError:
            continue

    return str(max(timestamps, default=int(time.time())))


def mjpeg_chunk(frame_bytes):
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"


def generate_detection_stream():
    yield mjpeg_chunk(frame_with_message("Starting camera feed...\nLoading model and webcam."))

    with _camera_lock:
        cap = None

        try:
            model, label_encoder = load_model()
            hands = create_hand_landmarker(num_hands=2)
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                update_prediction_state([])
                error_frame = frame_with_message(
                    "Unable to access webcam.\n"
                    "Close other camera apps.\n"
                    "Also allow camera access for desktop apps in Windows settings."
                )
                while True:
                    yield mjpeg_chunk(error_frame)
                    time.sleep(0.25)
                return

            while True:
                ret, frame = cap.read()
                if not ret:
                    update_prediction_state([])
                    retry_frame = frame_with_message("Waiting for webcam frame...\nPlease keep the camera available.")
                    yield mjpeg_chunk(retry_frame)
                    time.sleep(0.1)
                    continue

                frame = cv2.flip(frame, 1)
                detected_hands = detect_hands(hands, frame, int(time.time() * 1000))
                frame_predictions = []

                for idx, detected_hand in enumerate(detected_hands):
                    landmarks = detected_hand["flattened"]

                    with _collection_lock:
                        if _collection_state["active"]:
                            _collection_state["rows"].append([_collection_state["label"]] + landmarks)
                            _collection_state["count"] += 1

                    draw_hand_landmarks(frame, detected_hand["landmarks"])

                    if model is not None and label_encoder is not None and len(landmarks) == model.n_features_in_:
                        input_data = np.array(landmarks, dtype=float).reshape(1, -1)
                        probabilities = model.predict_proba(input_data)[0]
                        top_indices = np.argsort(probabilities)[::-1][:3]
                        best_index = int(top_indices[0])
                        label = str(label_encoder.classes_[best_index])
                        confidence = float(probabilities[best_index])
                        frame_predictions.append(
                            {
                                "hand": idx + 1,
                                "label": label,
                                "confidence": confidence,
                                "top_scores": [
                                    {
                                        "label": str(label_encoder.classes_[class_index]),
                                        "confidence": float(probabilities[class_index]),
                                    }
                                    for class_index in top_indices
                                ],
                            }
                        )
                        y_position = 60 + idx * 60
                        color = (0, 255 - idx * 100, 255)
                        cv2.putText(
                            frame,
                            f"Hand {idx + 1}: {label} ({confidence * 100:.0f}%)",
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

                update_prediction_state(frame_predictions)

                success, buffer = cv2.imencode(".jpg", frame)
                if not success:
                    continue

                yield mjpeg_chunk(buffer.tobytes())
        except Exception:
            update_prediction_state([])
            failure_frame = frame_with_message(
                "Camera startup failed.\n"
                "Restart the app and check webcam access."
            )
            while True:
                yield mjpeg_chunk(failure_frame)
                time.sleep(0.25)
        finally:
            if cap is not None:
                cap.release()
            update_prediction_state([])


@app.get("/")
def index():
    return render_template("index.html", asset_version=get_asset_version())


@app.get("/video_feed")
def video_feed():
    return Response(generate_detection_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/bootstrap")
def bootstrap():
    return jsonify(
        {
            "stats": stats_payload(),
            "prediction": prediction_output_payload(),
            "logs": prediction_logs_payload()["logs"],
            "collection": collection_status_payload(),
        }
    )


@app.post("/train")
def train():
    with _training_lock:
        if _training_state["status"] == "training":
            return jsonify({"message": "Training is already running.", "training": training_status_payload()}), 409

    payload = request.get_json(silent=True) or {}
    source_key = payload.get("source", "auto")
    set_training_state("training", "Training in progress...")

    try:
        message = train_model_for_ui(source_key)
    except Exception as exc:
        error_message = f"Training failed: {exc}"
        set_training_state("failed", error_message)
        return jsonify({"message": error_message, "training": training_status_payload(), "stats": stats_payload()}), 400

    with _model_lock:
        _model_cache["model"] = None
        _model_cache["label_encoder"] = None
    invalidate_model_metadata_cache()
    set_training_state("completed", message)
    return jsonify(
        {
            "message": message,
            "training": training_status_payload(),
            "stats": stats_payload(),
            "training_source": source_key,
        }
    )


@app.get("/collection/status")
def collection_status():
    return jsonify(collection_status_payload())


@app.get("/prediction/status")
def prediction_status():
    return jsonify(prediction_status_payload())


@app.get("/predict")
def predict():
    return jsonify(prediction_output_payload())


@app.get("/logs")
def logs():
    return jsonify(prediction_logs_payload())


@app.get("/stats")
def stats():
    return jsonify(stats_payload())


@app.post("/collect/start")
@app.post("/collection/start")
def collection_start():
    payload = request.get_json(silent=True) or {}
    success, message = start_collection(payload.get("label", ""))
    status_code = 200 if success else 400
    return jsonify({"message": message, **collection_status_payload()}), status_code


@app.post("/collect/stop")
@app.post("/collection/stop")
def collection_stop():
    message = stop_collection()
    return jsonify({"message": message, **collection_status_payload()})


@app.get("/model-info")
def model_info():
    return jsonify(get_model_metadata())


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
