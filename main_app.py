# main_app.py
import tkinter as tk
from tkinter import simpledialog, messagebox
import threading

from data_collection import collect_data
from train_model import train_model
from run_real_time import run_detection

def start_data_collection():
    label = simpledialog.askstring("Input", "Enter gesture label:")
    if label:
        threading.Thread(target=collect_data, args=(label,)).start()

def start_training():
    threading.Thread(target=lambda: (train_model(), messagebox.showinfo("Done", "Model Trained!"))).start()

def start_detection():
    threading.Thread(target=run_detection).start()

# --- GUI Setup ---
window = tk.Tk()
window.title("Hand Gesture Recognition - SVM")
window.geometry("600x600")

tk.Label(window, text="Choose an Action", font=("Arial", 14)).pack(pady=10)
tk.Button(window, text="📝 Collect Data", font=("Arial", 12), command=start_data_collection).pack(pady=5)
tk.Button(window, text="🧠 Train Model", font=("Arial", 12), command=start_training).pack(pady=5)
tk.Button(window, text="▶ Run Real-Time", font=("Arial", 12), command=start_detection).pack(pady=5)

window.mainloop()
