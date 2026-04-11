# Gesture to Language

Gesture to Language is a hand gesture recognition project that uses your webcam to detect hand signs and show the predicted gesture on screen.

You can use it in a web browser, collect your own gesture samples, and train the model again with a button click.

## What This Project Does

This project looks at your hand through the webcam.

It then:

- finds hand landmarks
- turns those landmarks into numbers
- sends those numbers to a trained machine learning model
- predicts which gesture you are showing

In simple words:

You show a hand sign, and the project tries to understand which sign it is.

## Who Can Use It

This project is made for:

- students
- beginners in AI or computer vision
- anyone who wants to test gesture recognition with a webcam

You do not need to understand coding to use the browser version once it is set up.

## Main Features

- live hand gesture detection through webcam
- browser-based interface
- collect new gesture samples from the same page
- train the model again after collecting new samples
- saved trained model for later use

## Tech Stack

### Simple explanation

- `Python`: the main programming language used for the project
- `Flask`: runs the local web app in your browser
- `OpenCV`: reads webcam video and handles image processing
- `MediaPipe`: detects hand landmarks
- `scikit-learn`: trains the gesture classification model
- `NumPy` and `pandas`: help process data
- `joblib`: saves and loads the trained model
- `Tkinter`: older desktop version of the app

### Technical stack

- Frontend:
  - HTML
  - CSS
  - JavaScript
- Backend:
  - Python
  - Flask
- Computer vision:
  - OpenCV
  - MediaPipe Hand Landmarker
- Machine learning:
  - scikit-learn SVM classifier
- Data handling:
  - NumPy
  - pandas
  - CSV files
- Model persistence:
  - joblib

## Current Model Information

The project currently includes a trained gesture model.

The model was trained using converted hand landmark data and is currently set up for these gesture labels:

- `palm`
- `l`
- `fist`
- `fist_moved`
- `thumb`
- `index`
- `ok`
- `palm_moved`
- `c`
- `down`

## What You Need

- A Windows computer
- Python 3.13 installed
- A webcam
- Internet connection for first-time setup

## How To Start The Project

### 1. Open the project folder

Open PowerShell inside the project folder.

Example:

```powershell
cd "c:\path\to\Gesture-to-Language--main"
```

### 2. Create a virtual environment

```powershell
py -3.13 -m venv .venv
```

If that does not work, use:

```powershell
python -m venv .venv
```

### 3. Activate the environment

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks it, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install the required packages

```powershell
pip install -r requirements.txt
```

## How To Run The Project

Run:

```powershell
python web_app.py
```

Then open this in your browser:

```text
http://127.0.0.1:5000
```

## How A Non-Technical Person Can Use It

### To only test the project

1. Start the app with `python web_app.py`
2. Open `http://127.0.0.1:5000`
3. Allow the webcam if Windows asks
4. Show one of the trained hand gestures to the camera
5. The project will try to display the gesture name

### To add your own new gesture

1. Open the browser app
2. In the `Gesture label` box, type the name of the gesture
3. Click `Start Collecting`
4. Show the gesture clearly to the webcam for a few seconds
5. Click `Stop And Save`
6. Click `Train Model`
7. Refresh the page if needed
8. Test the new gesture

## What To Type In Gesture Label

Gesture label means the name of the hand sign you are recording.

Examples:

- `A`
- `B`
- `Peace`
- `Thumbs_up`
- `Hello`

Tips:

- use one name for one gesture
- keep the spelling the same every time
- use short and simple names

Good example:

- always use `Thumbs_up`

Bad example:

- `thumbs up`
- `Thumbsup`
- `thumb_up`

These look similar to people, but the computer treats them as different labels.

## Project Files

- [web_app.py](c:/Users/ACER/Downloads/Gesture-to-Language--main/Gesture-to-Language--main/web_app.py:1): main browser app
- [train_model.py](c:/Users/ACER/Downloads/Gesture-to-Language--main/Gesture-to-Language--main/train_model.py:1): trains the machine learning model
- [data_collection.py](c:/Users/ACER/Downloads/Gesture-to-Language--main/Gesture-to-Language--main/data_collection.py:1): older desktop data collection script
- [run_real_time.py](c:/Users/ACER/Downloads/Gesture-to-Language--main/Gesture-to-Language--main/run_real_time.py:1): older desktop real-time detection script
- [hand_tracking.py](c:/Users/ACER/Downloads/Gesture-to-Language--main/Gesture-to-Language--main/hand_tracking.py:1): shared hand landmark detection logic
- [import_leapgestrecog.py](c:/Users/ACER/Downloads/Gesture-to-Language--main/Gesture-to-Language--main/import_leapgestrecog.py:1): imports and converts the `leapGestRecog` dataset into this project format

## Folder Overview

- `model/`: saved trained model files
- `datasets/`: user-collected gesture CSV files
- `datasets_leapgestrecog/`: converted landmark CSV files from the imported Kaggle dataset

## Common Problems And Easy Fixes

### Camera is not working

Try these steps:

- close Zoom, Teams, Camera app, or any other app using the webcam
- make sure Windows camera permission is enabled
- refresh the browser page
- restart `python web_app.py`

On Windows, check:

- `Settings > Privacy & security > Camera`
- turn on `Let apps access your camera`
- turn on `Let desktop apps access your camera`

### The page opens but no prediction appears

- make sure your hand is clearly visible
- keep good lighting
- move your hand slightly back from the camera
- try gestures that are already part of the trained model

### Training does not work

- make sure you collected samples first
- make sure at least two different gesture labels exist if you are training from your own custom data
- restart the app and try again

### Python packages are missing

Run:

```powershell
pip install -r requirements.txt
```

## Older Desktop Version

This project also contains an older desktop GUI version.

You can run it with:

```powershell
python main_app.py
```

But for most people, the browser version is easier to use.

## Verified Working Commands

These commands were successfully tested in this project:

```powershell
python web_app.py
python train_model.py
python import_leapgestrecog.py --data-dir "C:\Users\ACER\.cache\kagglehub\datasets\gti-upm\leapgestrecog\versions\1\leapGestRecog"
```

## Summary

This project is a webcam-based hand gesture recognition system.

The easiest way to use it is:

1. install the requirements
2. run `python web_app.py`
3. open `http://127.0.0.1:5000`
4. show your hand gesture to the webcam

If you want, I can also make the README look more like a GitHub project page with screenshots and a cleaner section layout.
