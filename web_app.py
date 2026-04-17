import csv
import os
import time
from collections import deque
from threading import Lock, Thread

import cv2
import joblib
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from hand_tracking import create_hand_landmarker, detect_hands, draw_hand_landmarks
from train_model import resolve_training_data_dir, train_model

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.jinja_env.auto_reload = True

_camera_lock = Lock()
_model_lock = Lock()
_model_cache = {"model": None, "label_encoder": None, "version": 0}
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
    "stage": "idle",
    "progress": 0.0,
    "last_started_at": None,
    "last_completed_at": None,
    "source_key": "auto",
    "source_title": "Auto Select",
    "resolved_source_key": "auto",
    "resolved_source_title": "Auto Select",
    "samples": 0,
    "classes": 0,
    "accuracy": None,
    "csv_files": 0,
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


def invalidate_loaded_model():
    with _model_lock:
        _model_cache["model"] = None
        _model_cache["label_encoder"] = None
        _model_cache["version"] += 1


def current_model_version():
    with _model_lock:
        return _model_cache["version"]


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
    label_count = 0
    available = False

    if os.path.isdir(data_dir):
        csv_files = [file for file in os.listdir(data_dir) if file.endswith(".csv")]
        csv_count = len(csv_files)
        label_count = csv_count
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
        "label_count": label_count,
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


def training_source_title(source_key):
    return _training_source_options.get(source_key, _training_source_options["auto"])["title"]


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


def normalized_label_key(label):
    return safe_label_name(label).lower()


def find_trained_label_conflict(label):
    candidate_key = normalized_label_key(label)
    for existing_label in get_model_metadata().get("labels", []):
        if normalized_label_key(existing_label) == candidate_key:
            return str(existing_label)
    return None


def start_collection(label):
    clean_label = label.strip()
    if not clean_label:
        return False, "Gesture label is required.", "missing_label"

    conflicting_label = find_trained_label_conflict(clean_label)
    if conflicting_label:
        return (
            False,
            f"Gesture '{conflicting_label}' is already used by the trained model. Choose a new gesture label.",
            "duplicate_trained_label",
        )

    with _collection_lock:
        if _collection_state["active"]:
            return False, f"Collection is already running for '{_collection_state['label']}'. Stop it first.", "collection_active"

        _collection_state["active"] = True
        _collection_state["label"] = clean_label
        _collection_state["file_label"] = safe_label_name(clean_label)
        _collection_state["rows"] = []
        _collection_state["count"] = 0
        _collection_state["last_message"] = f"Collecting '{clean_label}'. Keep your gesture visible in the camera."

    return True, _collection_state["last_message"], None


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


def clear_training_dataset(source_key):
    normalized_source_key, training_data_dir = resolve_requested_training_dir(source_key)

    with _collection_lock:
        if _collection_state["active"]:
            raise RuntimeError("Stop sample collection before clearing the dataset.")

    with _training_lock:
        if _training_state["status"] == "training":
            raise RuntimeError("Wait for training to finish before clearing the dataset.")

    deleted_files = 0
    if os.path.isdir(training_data_dir):
        for file_name in os.listdir(training_data_dir):
            if not file_name.endswith(".csv"):
                continue
            file_path = os.path.join(training_data_dir, file_name)
            try:
                os.remove(file_path)
                deleted_files += 1
            except OSError:
                continue

    invalidate_model_metadata_cache()

    resolved_title = get_dataset_title(training_data_dir)
    return {
        "source_key": normalized_source_key,
        "resolved_source_title": resolved_title,
        "deleted_files": deleted_files,
        "message": (
            f"Cleared {deleted_files} dataset files from {resolved_title}."
            if deleted_files
            else f"No dataset files were found in {resolved_title}."
        ),
    }


def set_training_state(status, message, **updates):
    with _training_lock:
        previous_status = _training_state["status"]
        _training_state["status"] = status
        _training_state["message"] = message
        if status == "training" and previous_status != "training":
            _training_state["last_started_at"] = time.time()
            _training_state["last_completed_at"] = None
        if status in {"completed", "failed"}:
            _training_state["last_completed_at"] = time.time()
        for key, value in updates.items():
            if value is not None or key in {"accuracy"}:
                _training_state[key] = value


def training_status_payload():
    with _training_lock:
        payload = dict(_training_state)

    started_at = payload.get("last_started_at")
    completed_at = payload.get("last_completed_at")
    elapsed_seconds = 0.0
    if started_at:
        elapsed_seconds = (time.time() - started_at) if payload["status"] == "training" else max((completed_at or started_at) - started_at, 0.0)

    payload["elapsed_seconds"] = round(elapsed_seconds, 1)
    return payload


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
        "training_stage": training["stage"],
        "training_progress": training["progress"],
        "training_elapsed_seconds": training["elapsed_seconds"],
        "training_source_key": training["source_key"],
        "training_source_title": training["resolved_source_title"],
        "training_requested_source_title": training["source_title"],
        "training_accuracy": training["accuracy"],
        "training_csv_files": training["csv_files"],
        "current_gesture": collection["label"],
        "current_gesture_samples": collection["count"],
        "collection_active": collection["active"],
    }


def train_model_for_ui(source_key="auto", progress_callback=None):
    normalized_source_key, training_data_dir = resolve_requested_training_dir(source_key)
    resolved_source_key = "imported" if training_data_dir == "datasets_leapgestrecog" else "custom"
    summary = train_model(data_dir=training_data_dir, progress_callback=progress_callback)
    dataset_title = get_dataset_title(training_data_dir)

    if summary["accuracy"] is None:
        message = (
            f"Model training completed from {dataset_title} with {summary['samples']} samples "
            f"across {summary['classes']} labels. Add more data for an accuracy score."
        )
    else:
        message = (
            f"Model training completed from {dataset_title} with {summary['accuracy'] * 100:.2f}% "
            f"accuracy using {summary['samples']} samples across {summary['classes']} labels."
        )

    return {
        "message": message,
        "summary": summary,
        "source_key": normalized_source_key,
        "source_title": training_source_title(normalized_source_key),
        "resolved_source_key": resolved_source_key,
        "resolved_source_title": dataset_title,
    }


def launch_training_job(source_key="auto"):
    normalized_source_key, training_data_dir = resolve_requested_training_dir(source_key)
    resolved_source_key = "imported" if training_data_dir == "datasets_leapgestrecog" else "custom"
    set_training_state(
        "training",
        "Preparing the training pipeline.",
        stage="queued",
        progress=0.03,
        source_key=normalized_source_key,
        source_title=training_source_title(normalized_source_key),
        resolved_source_key=resolved_source_key,
        resolved_source_title=get_dataset_title(training_data_dir),
        samples=0,
        classes=0,
        accuracy=None,
        csv_files=0,
    )

    def run_job():
        try:
            def handle_progress(update):
                set_training_state(
                    "training",
                    update.get("message", "Training in progress."),
                    stage=update.get("stage", "training"),
                    progress=round(float(update.get("progress", 0.0)), 2),
                    source_key=normalized_source_key,
                    source_title=training_source_title(normalized_source_key),
                    resolved_source_key=resolved_source_key,
                    resolved_source_title=get_dataset_title(training_data_dir),
                    samples=update.get("samples"),
                    classes=update.get("classes"),
                    accuracy=update.get("accuracy"),
                    csv_files=update.get("csv_files"),
                )

            training_result = train_model_for_ui(normalized_source_key, progress_callback=handle_progress)
            summary = training_result["summary"]

            set_training_state(
                "training",
                "Refreshing the live model cache.",
                stage="refreshing",
                progress=0.99,
                source_key=training_result["source_key"],
                source_title=training_result["source_title"],
                resolved_source_key=training_result["resolved_source_key"],
                resolved_source_title=training_result["resolved_source_title"],
                samples=summary["samples"],
                classes=summary["classes"],
                accuracy=summary["accuracy"],
                csv_files=summary["csv_files"],
            )
            invalidate_loaded_model()
            invalidate_model_metadata_cache()
            load_model()
            set_training_state(
                "completed",
                training_result["message"],
                stage="completed",
                progress=1.0,
                source_key=training_result["source_key"],
                source_title=training_result["source_title"],
                resolved_source_key=training_result["resolved_source_key"],
                resolved_source_title=training_result["resolved_source_title"],
                samples=summary["samples"],
                classes=summary["classes"],
                accuracy=summary["accuracy"],
                csv_files=summary["csv_files"],
            )
        except Exception as exc:
            set_training_state(
                "failed",
                f"Training failed: {exc}",
                stage="failed",
                progress=1.0,
            )

    worker = Thread(target=run_job, daemon=True)
    worker.start()
    return training_status_payload()


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
        model = None
        label_encoder = None
        active_model_version = -1

        try:
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
                latest_model_version = current_model_version()
                if latest_model_version != active_model_version:
                    model, label_encoder = load_model()
                    active_model_version = latest_model_version
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
            "training": training_status_payload(),
            "model": get_model_metadata(),
        }
    )


@app.post("/train")
def train():
    with _training_lock:
        if _training_state["status"] == "training":
            return jsonify({"message": "Training is already running.", "training": training_status_payload()}), 409

    payload = request.get_json(silent=True) or {}
    try:
        training_state = launch_training_job(payload.get("source", "auto"))
    except ValueError as exc:
        return jsonify({"message": str(exc), "training": training_status_payload(), "stats": stats_payload()}), 400
    return (
        jsonify(
            {
                "message": "Training started.",
                "training": training_state,
                "stats": stats_payload(),
            }
        ),
        202,
    )


@app.get("/training/status")
def training_status():
    return jsonify(training_status_payload())


@app.get("/training/sources")
def training_sources():
    metadata = get_model_metadata()
    return jsonify(
        {
            "sources": metadata["training_sources"],
            "default_source": metadata["default_training_source"],
            "resolved_auto_source": metadata["resolved_auto_source"],
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
    success, message, error_type = start_collection(payload.get("label", ""))
    status_code = 200 if success else 400
    return jsonify({"message": message, "error_type": error_type, **collection_status_payload()}), status_code


@app.post("/collect/stop")
@app.post("/collection/stop")
def collection_stop():
    message = stop_collection()
    return jsonify({"message": message, **collection_status_payload()})


@app.get("/model-info")
def model_info():
    return jsonify(get_model_metadata())


@app.post("/dataset/clear")
def dataset_clear():
    payload = request.get_json(silent=True) or {}
    try:
        result = clear_training_dataset(payload.get("source", "auto"))
    except ValueError as exc:
        return jsonify({"message": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"message": str(exc)}), 409

    return jsonify(
        {
            **result,
            "stats": stats_payload(),
            "model": get_model_metadata(),
        }
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
