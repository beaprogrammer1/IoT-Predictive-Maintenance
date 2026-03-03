import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# --- STEP 1: DATA GENERATION (Robust) ---
def prepare_data():
    file_name = 'sensor_data.csv'
    # Hum purani file delete kar dete hain takay naye columns ke saath fresh start ho
    if os.path.exists(file_name):
        os.remove(file_name)
        
    print("--- 🛠️ Generating Fresh Professional Dataset ---")
    np.random.seed(42)
    temp = np.random.normal(75, 5, 1000)
    vib = np.random.normal(0.05, 0.02, 1000)
    pres = np.random.normal(101, 2, 1000)
    # Formula: Life drops when temp and vib increase
    rul = 250 - (temp * 2) - (vib * 200) + np.random.normal(0, 2, 1000)
    rul = np.clip(rul, 0, 100)
    
    df = pd.DataFrame({'temp': temp, 'vib': vib, 'pres': pres, 'RUL': rul})
    df.to_csv(file_name, index=False)
    return df

# --- STEP 2: MODEL TRAINING ---
def train_model(df):
    print("--- 🧠 Training AI Model ---")
    X = df[['temp', 'vib', 'pres']]
    y = df['RUL']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    print("✅ AI is Trained and Ready!")
    return model

# --- STEP 3: LIVE APP INTERFACE ---
def start_app():
    data = prepare_data()
    model = train_model(data)
    
    print("\n" + "="*45)
    print("     🚀 IoT PREDICTIVE MAINTENANCE SYSTEM")
    print("="*45)
    
    while True:
        try:
            print("\n[Input Current Sensor Telemetry]")
            t = float(input("Current Temp (°C) [Avg 70-90]: "))
            v = float(input("Current Vibration (mm/s) [Avg 0.02-0.1]: "))
            p = float(input("Current Pressure (psi) [Avg 100-110]: "))
            
            # Predict
            pred = model.predict([[t, v, p]])[0]
            
            print(f"\n--- 📊 PREDICTION RESULT ---")
            print(f"Machine Health: {pred:.1f}% Remaining Life")
            
            if pred < 25:
                print("🚨 ALERT: CRITICAL! Machine failure predicted very soon.")
                print("ACTION: Triggering Emergency Maintenance Ticket...")
            elif pred < 60:
                print("🔔 NOTICE: MODERATE WEAR. Schedule a checkup in the next shift.")
            else:
                print("🟢 STATUS: OPTIMAL. No maintenance required.")
                
            choice = input("\nAnalyze another machine? (y/n): ")
            if choice.lower() != 'y':
                print("Exiting System... Goodbye!")
                break
        except Exception as e:
            print(f"❌ Invalid Input! Please enter numbers. Error: {e}")

if __name__ == "__main__":
    start_app()