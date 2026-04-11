# Gesture to Language - Setup Guide

This project was verified on Windows with Python 3.13.

## What you need

- Python 3.13 installed
- A webcam
- Internet connection for installing packages

## 1. Open the project folder

Open PowerShell in the project folder:

```powershell
cd "c:\path\to\Gesture-to-Language--main"
```

## 2. Create a virtual environment

```powershell
py -3.13 -m venv .venv
```

If `py -3.13` does not work, use:

```powershell
python -m venv .venv
```

## 3. Activate the virtual environment

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks the script, run this once in the same terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```powershell
.venv\Scripts\Activate.ps1
```

## 4. Upgrade pip

```powershell
python -m pip install --upgrade pip
```

## 5. Install dependencies

```powershell
pip install -r requirements.txt
```

Installed packages:

- `mediapipe`
- `opencv-python`
- `numpy`
- `pandas`
- `scikit-learn`
- `joblib`
- `flask`

Note:
- `tkinter` is used by the GUI and usually comes with the standard Windows Python installer.
- `tensorflow` is not required for this project and was removed from the dependency list because the code does not use it.

## 6. Run the web app

```powershell
python web_app.py
```

Then open:

```text
http://127.0.0.1:5000
```

This opens the browser version of the project with:

- live webcam gesture detection
- browser-based data collection
- a model training button
- a local video stream in the browser

## 7. How to use the project

### Option A: Use the already included trained model

If you only want to test prediction from the browser:

```powershell
python web_app.py
```

Then open:

```text
http://127.0.0.1:5000
```

If you want the older desktop camera window instead:

```powershell
python run_real_time.py
```

Or open the GUI:

```powershell
python main_app.py
```

Then click `Run Real-Time`.

### Option B: Add your own gesture and retrain from the browser

1. Run the web app:

```powershell
python web_app.py
```

2. Open:

```text
http://127.0.0.1:5000
```

3. Enter a gesture label in the page.

4. Click `Start Collecting`.

5. Show the gesture to the camera for a few seconds.

6. Click `Stop And Save`.

7. Click `Train Model`.

8. Refresh the stream if needed and test the new gesture.

### Option C: Add your own gesture from the older desktop tools

1. Start data collection:

```powershell
python data_collection.py
```

Better option:

```powershell
python main_app.py
```

Then click `Collect Data`, enter a label, and press `q` to stop recording.

2. Train the model:

```powershell
python train_model.py
```

3. Run detection:

```powershell
python run_real_time.py
```

## Files used by the project

- `main_app.py`: GUI launcher
- `web_app.py`: browser-based localhost app
- `data_collection.py`: records gesture landmark samples into `datasets/`
- `train_model.py`: trains the SVM model and saves files into `model/`
- `run_real_time.py`: loads the trained model and predicts gestures from webcam input

## Output folders

- `datasets/`: gesture CSV files
- `model/`: trained model files

## Common issues

### Webcam does not open

- Close other apps using the camera
- Check Windows camera permissions
- Restart the app

### `ModuleNotFoundError`

Make sure the virtual environment is activated, then run:

```powershell
pip install -r requirements.txt
```

### PowerShell says running scripts is disabled

Use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again.

## Verified commands

These commands were successfully verified in this project:

```powershell
python train_model.py
python web_app.py
```
