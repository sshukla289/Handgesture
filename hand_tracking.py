import os
import urllib.request
from threading import Lock

import cv2
import mediapipe as mp

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_LANDMARKER_PATH = os.path.join("model", "hand_landmarker.task")
_landmarker_lock = Lock()
_landmarker_cache = {}


def ensure_hand_landmarker_model():
    os.makedirs("model", exist_ok=True)
    if not os.path.exists(HAND_LANDMARKER_PATH):
        urllib.request.urlretrieve(HAND_LANDMARKER_URL, HAND_LANDMARKER_PATH)
    return HAND_LANDMARKER_PATH


def create_hand_landmarker(num_hands=2):
    with _landmarker_lock:
        cached_landmarker = _landmarker_cache.get(num_hands)
        if cached_landmarker is not None:
            return cached_landmarker

        model_path = os.path.abspath(ensure_hand_landmarker_model())
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        _landmarker_cache[num_hands] = landmarker
        return landmarker


def detect_hands(landmarker, frame, timestamp_ms):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    detected_hands = []
    for hand_landmarks in result.hand_landmarks:
        flattened = []
        for landmark in hand_landmarks:
            flattened.extend([landmark.x, landmark.y, landmark.z])
        detected_hands.append(
            {
                "landmarks": hand_landmarks,
                "flattened": flattened,
            }
        )

    return detected_hands


def draw_hand_landmarks(frame, hand_landmarks, color=(80, 220, 120)):
    height, width = frame.shape[:2]
    connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

    for connection in connections:
        start = hand_landmarks[connection.start]
        end = hand_landmarks[connection.end]
        start_point = (int(start.x * width), int(start.y * height))
        end_point = (int(end.x * width), int(end.y * height))
        cv2.line(frame, start_point, end_point, color, 2)

    for landmark in hand_landmarks:
        point = (int(landmark.x * width), int(landmark.y * height))
        cv2.circle(frame, point, 4, (255, 255, 255), -1)
        cv2.circle(frame, point, 6, color, 1)
