import tkinter as tk
from tkinter import messagebox
import pandas as pd
import joblib
import numpy as np

# --- 1. SETUP MAIN WINDOW ---
root = tk.Tk()
root.title("Smartphone Price Predictor")
root.geometry("400x550")
root.configure(bg="#f0f0f0")

# --- 2. LOAD MODEL & SCALER ---
# (Make sure these files exist in a 'models' folder)
try:
    model = joblib.load('C:\\Users\\Youssef Elshemy\\Documents\\Projects\\Smartphone-Price-Prediction\\models\\best_random_forest.pkl')
    scaler = joblib.load('C:\\Users\\Youssef Elshemy\\Documents\\Projects\\Smartphone-Price-Prediction\\models\\scaler.pkl')
except FileNotFoundError:
    messagebox.showerror("Error", "Model files not found!\nRun preprocessing/training first.")
    exit()

# --- 3. DEFINE UI ELEMENTS (So the function can see them) ---

# Header
tk.Label(root, text="Phone Price Predictor", font=("Helvetica", 16, "bold"), bg="#f0f0f0").pack(pady=20)

# Input Frame
input_frame = tk.Frame(root, bg="#f0f0f0")
input_frame.pack(pady=10)


def create_input(label_text, row):
    tk.Label(input_frame, text=label_text, bg="#f0f0f0", font=("Arial", 10)).grid(row=row, column=0, padx=10, pady=5,
                                                                                  sticky="e")
    entry = tk.Entry(input_frame, width=20)
    entry.grid(row=row, column=1, padx=10, pady=5)
    return entry


entry_ram = create_input("RAM (GB):", 0)
entry_storage = create_input("Storage (GB):", 1)
entry_speed = create_input("Clock Speed (GHz):", 2)
entry_battery = create_input("Battery (mAh):", 3)

# Result Label (Defined HERE so the function knows it exists)
result_label = tk.Label(root, text="Enter specs to see result...", font=("Helvetica", 12), bg="#f0f0f0")


# --- 4. PREDICTION FUNCTION ---
def predict_price():
    try:
        # Get inputs
        ram = float(entry_ram.get())
        storage = float(entry_storage.get())
        speed = float(entry_speed.get())
        battery = float(entry_battery.get())

        # Prepare Data
        # Get all columns the model was trained on
        model_cols = scaler.feature_names_in_

        # Create a row of zeros for all columns
        input_data = pd.DataFrame(0, index=[0], columns=model_cols)

        # Fill in the specific values we have
        # (Ensure these names match your CSV column headers exactly!)
        input_data['RAM Size GB'] = ram
        input_data['Storage Size GB'] = storage
        input_data['Clock_Speed_GHz'] = speed
        input_data['battery_capacity'] = battery

        # Scale
        scaled_data = scaler.transform(input_data)

        # Predict
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]

        # Update Label
        if prediction == 1:
            result_label.config(text=f"Result: EXPENSIVE 💎\nConfidence: {probability * 100:.1f}%", fg="red")
        else:
            result_label.config(text=f"Result: Non-expensive 💰\nConfidence: {(1 - probability) * 100:.1f}%", fg="green")

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")


# --- 5. BUTTON & PACKING ---
btn = tk.Button(root, text="PREDICT PRICE", command=predict_price,
                bg="#2980b9", fg="white", font=("Arial", 12, "bold"), height=2, width=20)
btn.pack(pady=30)

# Pack the result label last
result_label.pack()

# Start App
root.mainloop()