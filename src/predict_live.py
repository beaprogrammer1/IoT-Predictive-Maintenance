import joblib
import pandas as pd

def live_prediction():
    # 1. Load the trained model
    try:
        model = joblib.load('src/model.pkl')
    except:
        print("❌ Model file not found! Please run main_project.py first.")
        return

    print("--- 🛠️ IoT Machine Health Monitor (Live) ---")
    print("Enter the current sensor readings to predict Remaining Life.")
    
    try:
        # 2. Get User Input
        temp = float(input("Enter Temperature (°C): "))
        vib = float(input("Enter Vibration Level (mm/s): "))
        pres = float(input("Enter Pressure (psi): "))

        # 3. Format input for model
        # We use the same feature names as training
        input_data = pd.DataFrame([[temp, vib, pres]], 
                                 columns=['temp_rolling_avg', 'vib_rolling_avg', 'sensor_pres'])

        # 4. Predict
        prediction = model.predict(input_data)[0]

        print("\n--- 📊 AI PREDICTION ---")
        print(f"Estimated Remaining Useful Life (RUL): {prediction:.1f} Hours")
        
        if prediction < 20:
            print("⚠️ WARNING: Machine failure imminent! Schedule maintenance immediately.")
        elif prediction < 50:
            print("🔔 NOTICE: Machine showing signs of wear. Plan checkup soon.")
        else:
            print("✅ STATUS: Machine is healthy.")

    except ValueError:
        print("❌ Invalid input! Please enter numbers only.")

if __name__ == "__main__":
    while True:
        live_prediction()
        cont = input("\nPredict again? (y/n): ")
        if cont.lower() != 'y':
            break