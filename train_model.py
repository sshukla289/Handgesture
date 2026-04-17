# train_model.py
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


def emit_progress(progress_callback, stage, progress, message, **extra):
    if callable(progress_callback):
        progress_callback(
            {
                "stage": stage,
                "progress": progress,
                "message": message,
                **extra,
            }
        )

def resolve_training_data_dir(data_dir):
    if data_dir is not None:
        return data_dir

    preferred_dir = 'datasets_leapgestrecog'
    if os.path.isdir(preferred_dir) and any(file.endswith(".csv") for file in os.listdir(preferred_dir)):
        return preferred_dir

    return 'datasets'


def train_model(data_dir=None, model_dir='model', progress_callback=None):
    data_dir = resolve_training_data_dir(data_dir)
    all_data = []

    if not os.path.isdir(data_dir):
        raise FileNotFoundError("The datasets folder was not found.")

    csv_files = sorted(file for file in os.listdir(data_dir) if file.endswith(".csv"))
    emit_progress(
        progress_callback,
        "scanning",
        0.08,
        f"Scanning {len(csv_files)} training files in {data_dir}.",
        csv_files=len(csv_files),
        data_dir=data_dir,
    )

    for index, file in enumerate(csv_files, start=1):
        if file.endswith(".csv"):
            file_path = os.path.join(data_dir, file)
            df = pd.read_csv(file_path, header=None)

            # Check if first column is the label, and rest are landmarks
            if df.shape[1] == 64:  # 1 label + 21 landmarks * 3 = 63 features
                df.columns = ['label'] + [f'f{i}' for i in range(1, 64)]
                all_data.append(df)
            else:
                print(f"[WARNING] Skipping {file} due to unexpected format.")

            base_progress = 0.16
            file_progress = (index / max(len(csv_files), 1)) * 0.28
            emit_progress(
                progress_callback,
                "loading",
                base_progress + file_progress,
                f"Loading samples from {file}.",
                current_file=file,
                files_processed=index,
                csv_files=len(csv_files),
            )

    if not all_data:
        raise ValueError("No valid training data was found in the datasets folder.")

    df = pd.concat(all_data, ignore_index=True)
    X = df.drop('label', axis=1).values
    y = df['label'].values

    emit_progress(
        progress_callback,
        "preparing",
        0.5,
        "Encoding labels and preparing the training split.",
        samples=len(df),
    )

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_count = len(label_encoder.classes_)

    if class_count < 2:
        raise ValueError("At least two different gesture labels are required to train the model.")

    counts = pd.Series(y_encoded).value_counts()
    can_create_holdout = len(df) >= 10 and counts.min() >= 2

    if can_create_holdout:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=0.2,
            random_state=42,
            stratify=y_encoded,
        )
    else:
        X_train, y_train = X, y_encoded
        X_test, y_test = None, None

    # Train SVM
    emit_progress(
        progress_callback,
        "training",
        0.72,
        "Fitting the SVM classifier.",
        samples=len(df),
        classes=class_count,
    )
    clf = SVC(kernel='rbf', probability=True)
    clf.fit(X_train, y_train)

    # Accuracy
    accuracy = None
    emit_progress(
        progress_callback,
        "evaluating",
        0.88,
        "Evaluating the refreshed model.",
    )
    if X_test is not None and y_test is not None:
        accuracy = clf.score(X_test, y_test)
        print(f"[INFO] Model trained with accuracy: {accuracy * 100:.2f}%")
    else:
        print("[INFO] Model trained without a holdout accuracy score because the dataset is still small.")

    # Save model and label encoder
    emit_progress(
        progress_callback,
        "saving",
        0.96,
        "Saving model artifacts and label encoder.",
    )
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(model_dir, 'svm_model.pkl'))
    joblib.dump(label_encoder, os.path.join(model_dir, 'label_encoder.pkl'))
    print(f"[INFO] Model and encoder saved in '{model_dir}/' directory.")
    emit_progress(
        progress_callback,
        "saved",
        1.0,
        "Model artifacts saved successfully.",
        samples=len(df),
        classes=class_count,
        accuracy=accuracy,
    )
    return {
        "accuracy": accuracy,
        "samples": len(df),
        "classes": class_count,
        "data_dir": data_dir,
        "csv_files": len(csv_files),
    }

if __name__ == "__main__":
    summary = train_model()
    if summary["accuracy"] is None:
        print(f"[INFO] Trained on {summary['samples']} samples across {summary['classes']} labels.")
    else:
        print(
            f"[INFO] Trained on {summary['samples']} samples across "
            f"{summary['classes']} labels with {summary['accuracy'] * 100:.2f}% accuracy."
        )
