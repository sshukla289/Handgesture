import argparse
import csv
import os
from collections import defaultdict

import cv2

from hand_tracking import create_hand_landmarker, detect_hands
from train_model import train_model


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def normalize_label(folder_name):
    parts = folder_name.split("_", 1)
    label = parts[1] if len(parts) == 2 and parts[0].isdigit() else folder_name
    return label.strip().replace(" ", "_")


def resolve_dataset_root(path):
    candidate = os.path.abspath(path)
    if os.path.isdir(os.path.join(candidate, "00")):
        return candidate

    nested = os.path.join(candidate, "leapGestRecog")
    if os.path.isdir(os.path.join(nested, "00")):
        return nested

    raise FileNotFoundError(
        "Could not find the leapGestRecog dataset folders. "
        "Expected a directory containing subfolders like '00', '01', ... '09'."
    )


def clear_existing_csvs(output_dir):
    if not os.path.isdir(output_dir):
        return

    for file_name in os.listdir(output_dir):
        if file_name.endswith(".csv"):
            os.remove(os.path.join(output_dir, file_name))


def import_dataset(dataset_root, output_dir, num_hands=1):
    os.makedirs(output_dir, exist_ok=True)
    clear_existing_csvs(output_dir)

    detector = create_hand_landmarker(num_hands=num_hands)
    rows_by_label = defaultdict(list)
    stats = defaultdict(int)
    skipped_images = []
    timestamp_ms = 0

    try:
        for subject_name in sorted(os.listdir(dataset_root)):
            subject_path = os.path.join(dataset_root, subject_name)
            if not os.path.isdir(subject_path):
                continue

            for gesture_dir in sorted(os.listdir(subject_path)):
                gesture_path = os.path.join(subject_path, gesture_dir)
                if not os.path.isdir(gesture_path):
                    continue

                label = normalize_label(gesture_dir)

                for image_name in sorted(os.listdir(gesture_path)):
                    _, ext = os.path.splitext(image_name)
                    if ext.lower() not in IMAGE_EXTENSIONS:
                        continue

                    image_path = os.path.join(gesture_path, image_name)
                    frame = cv2.imread(image_path)
                    if frame is None:
                        skipped_images.append(image_path)
                        continue

                    timestamp_ms += 1
                    detected_hands = detect_hands(detector, frame, timestamp_ms)
                    if not detected_hands:
                        skipped_images.append(image_path)
                        continue

                    hand = detected_hands[0]
                    rows_by_label[label].append([label] + hand["flattened"])
                    stats[label] += 1
    finally:
        detector.close()

    if not rows_by_label:
        raise ValueError(
            "No usable hand landmarks were extracted from the dataset images. "
            "The dataset may be missing, unreadable, or incompatible."
        )

    for label, rows in rows_by_label.items():
        file_path = os.path.join(output_dir, f"{label}_data.csv")
        with open(file_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(rows)

    return stats, skipped_images


def main():
    parser = argparse.ArgumentParser(
        description="Import the leapGestRecog image dataset into landmark CSVs and train the project model."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to the leapGestRecog dataset root or its parent directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets_leapgestrecog",
        help="Folder where extracted landmark CSV files will be written.",
    )
    parser.add_argument(
        "--model-dir",
        default="model",
        help="Folder where the trained svm_model.pkl and label_encoder.pkl will be saved.",
    )
    args = parser.parse_args()

    dataset_root = resolve_dataset_root(args.data_dir)
    print(f"[INFO] Using dataset root: {dataset_root}")

    stats, skipped_images = import_dataset(dataset_root, args.output_dir)
    total_samples = sum(stats.values())
    print(f"[INFO] Extracted {total_samples} landmark samples across {len(stats)} labels.")
    for label in sorted(stats):
        print(f"[INFO] {label}: {stats[label]} samples")

    if skipped_images:
        print(f"[WARNING] Skipped {len(skipped_images)} images where no hand was detected.")

    summary = train_model(data_dir=args.output_dir, model_dir=args.model_dir)
    if summary["accuracy"] is None:
        print(
            f"[INFO] Training finished with {summary['samples']} samples across "
            f"{summary['classes']} labels."
        )
    else:
        print(
            f"[INFO] Training finished with {summary['accuracy'] * 100:.2f}% accuracy "
            f"using {summary['samples']} samples across {summary['classes']} labels."
        )


if __name__ == "__main__":
    main()
