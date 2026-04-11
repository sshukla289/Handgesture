# run_real_time.py
import cv2
import numpy as np
import joblib
import time

from hand_tracking import create_hand_landmarker, detect_hands, draw_hand_landmarks

def run_detection():
    # Load trained SVM model and label encoder
    model = joblib.load('model/svm_model.pkl')
    label_encoder = joblib.load('model/label_encoder.pkl')

    hands = create_hand_landmarker(num_hands=2)

    # Start webcam
    cap = cv2.VideoCapture(0)

    print("[INFO] Starting real-time detection. Press ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)

        detected_hands = detect_hands(hands, frame, int(time.time() * 1000))

        for idx, detected_hand in enumerate(detected_hands):
            landmarks = detected_hand["flattened"]
            if len(landmarks) == model.n_features_in_:
                    input_data = np.array(landmarks, dtype=float).reshape(1, -1)
                    prediction = model.predict(input_data)
                    label = label_encoder.inverse_transform(prediction)[0]

                    draw_hand_landmarks(frame, detected_hand["landmarks"])

                    # Display label text for each hand at different Y positions
                    y_position = 60 + idx * 60
                    color = (0, 255 - idx * 100, 255)  # Greenish for first hand, pinkish for second
                    cv2.putText(frame, f"Hand {idx + 1}: {label}", (10, y_position),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        # Show result
        cv2.imshow("Real-Time Detection", frame)

        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

# Run if this script is executed
if __name__ == "__main__":
    run_detection()
