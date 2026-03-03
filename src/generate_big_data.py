import pandas as pd
import numpy as np

def generate_big_iot_data(rows=1000):
    np.random.seed(42)
    
    # Generate 1000 rows of synthetic IoT data
    temp = np.random.normal(75, 5, rows)
    vib = np.random.normal(0.05, 0.02, rows)
    pres = np.random.normal(101, 2, rows)
    
    # Create RUL based on a formula (Higher temp/vib = Lower RUL)
    # RUL = 200 - (temp * 1.5) - (vib * 100) + noise
    rul = 250 - (temp * 2) - (vib * 200) + np.random.normal(0, 2, rows)
    rul = np.clip(rul, 0, 100) # Keep RUL between 0 and 100
    
    df = pd.DataFrame({
        'sensor_temp': temp,
        'sensor_vib': vib,
        'sensor_pres': pres,
        'RUL': rul
    })
    
    df.to_csv('sensor_data.csv', index=False)
    print(f"✅ Professional Dataset Generated: 1000 rows saved to sensor_data.csv")

if __name__ == "__main__":
    generate_big_iot_data()