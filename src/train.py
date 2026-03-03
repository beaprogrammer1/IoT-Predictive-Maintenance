import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

def train_model(data_path):
    print("Checking for data...")
    if not os.path.exists(data_path):
        print("Error: data/processed_data.csv missing! Pehle preprocessing chalayen.")
        return

    df = pd.read_csv(data_path)
    features = ['temp_rolling_avg', 'vib_rolling_avg', 'sensor_pres']
    X = df[features]
    y = df['RUL']
    
    print("Splitting data...")
    # Humein kam az kam 2 samples chahiye test ke liye
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, 'src/model.pkl')
    
    # Save test results
    test_df = X_test.copy()
    test_df['actual_RUL'] = y_test
    test_df.to_csv('data/test_results.csv', index=False)
    
    print("✅ Training Complete: Model and test_results saved!")

if __name__ == "__main__":
    train_model('data/processed_data.csv')