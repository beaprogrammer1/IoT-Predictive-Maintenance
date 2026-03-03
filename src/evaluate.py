import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error

def evaluate_performance(test_data_path, model_path):
    df_test = pd.read_csv(test_data_path)
    model = joblib.load(model_path)
    
    features = ['temp_rolling_avg', 'vib_rolling_avg', 'sensor_pres']
    X_test = df_test[features]
    y_true = df_test['actual_RUL']
    
    # Make Predictions
    y_pred = model.predict(X_test)
    df_test['predicted_RUL'] = y_pred
    
    # 4. Error Analysis
    mae = mean_absolute_error(y_true, y_pred)
    df_test['absolute_error'] = abs(y_true - y_pred)
    
    # Identify specific data points with highest error (Top 3)
    high_error_points = df_test.sort_values(by='absolute_error', ascending=False).head(3)
    
    print(f"Model Mean Absolute Error (MAE): {mae:.2f}")
    print("\n--- Top 3 Highest Error Data Points ---")
    print(high_error_points)

if __name__ == "__main__":
    evaluate_performance('data/test_results.csv', 'src/model.pkl')