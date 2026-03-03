import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# --- CONFIGURATION ---
DATA_FILE = 'sensor_data.csv' 

def create_sample_data():
    """Agar file khali ho toh ye sample data bana degi"""
    data = {
        'timestamp': pd.date_range(start='2023-01-01', periods=10, freq='H'),
        'sensor_temp': [70.5, 71.2, 70.8, 72.5, 73.1, 85.0, 87.2, 75.0, 72.0, 71.5],
        'sensor_vib': [0.02, 0.02, 0.03, 0.05, 0.04, 0.15, 0.20, 0.08, 0.03, 0.02],
        'sensor_pres': [101.3, 101.5, 101.2, 101.8, 101.4, 105.0, 106.5, 102.0, 101.5, 101.2],
        'RUL': [100, 90, 80, 70, 60, 10, 5, 40, 70, 85]
    }
    df = pd.DataFrame(data)
    df.to_csv(DATA_FILE, index=False)
    print(f"Created sample data file: {DATA_FILE}")

def run_pipeline():
    print("--- Phase 1: Data Loading & Cleaning ---")
    
    # 1. Check if file exists
    if not os.path.exists(DATA_FILE):
        create_sample_data()

    # 2. Load Data
    df = pd.read_csv(DATA_FILE)
    df = df.ffill()
    
    # 3. Feature Engineering
    df['temp_rolling_avg'] = df['sensor_temp'].rolling(window=3, min_periods=1).mean()
    df['vib_rolling_avg'] = df['sensor_vib'].rolling(window=3, min_periods=1).mean()
    print("✅ Data Preprocessing Complete.")

    print("\n--- Phase 2: Model Training ---")
    features = ['temp_rolling_avg', 'vib_rolling_avg', 'sensor_pres']
    X = df[features]
    y = df['RUL']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("✅ Model Trained.")

    print("\n--- Phase 3: Results ---")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"📊 Mean Absolute Error (MAE): {mae:.2f}")
    print("\nSample Predictions:")
    for i in range(len(y_pred)):
        print(f"Actual: {y_test.iloc[i]} | Predicted: {y_pred[i]:.1f}")

# YE LINES BOHT ZAROORI HAIN - INKE BAGHAIR CODE NAHI CHALAY GA
if __name__ == "__main__":
    run_pipeline()