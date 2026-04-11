# data_collection.py
import cv2
import csv
import os
import time

from hand_tracking import create_hand_landmarker, detect_hands, draw_hand_landmarks

def collect_data(label):
    filename = f"datasets/{label}_data.csv"
    os.makedirs("datasets", exist_ok=True)

    hands = create_hand_landmarker(num_hands=2)

    cap = cv2.VideoCapture(0)
    data = []

    print(f"[INFO] Collecting data for '{label}'. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        image = cv2.flip(frame, 1)
        detected_hands = detect_hands(hands, image, int(time.time() * 1000))

        for detected_hand in detected_hands:
            data.append([label] + detected_hand["flattened"])
            draw_hand_landmarks(image, detected_hand["landmarks"])

        cv2.imshow("Data Collection", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    # Save CSV
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

    print(f"[INFO] Saved {len(data)} samples to {filename}")

if __name__ == "__main__":
    label = input("Enter gesture label: ").strip()
    if label:
        collect_data(label)
    else:
        print("[ERROR] Gesture label is required.")
