import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_plots():
    # File ka rasta check karen
    file_path = 'sensor_data.csv'
    
    print(f"--- Checking for {file_path} ---")
    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} nahi mili! Kya aapne main_project.py run kiya tha?")
        return

    # 1. Load the data
    df = pd.read_csv(file_path)
    print(f"✅ Data Loaded: {len(df)} rows found.")

    # Set the style
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(15, 10))

    print("--- Generating Charts ---")
    
    # Chart 1: Temp vs RUL
    plt.subplot(2, 2, 1)
    sns.lineplot(data=df, x='RUL', y='sensor_temp', marker='o', color='red')
    plt.gca().invert_xaxis()
    plt.title('Temperature vs. Machine Life')

    # Chart 2: Vibration vs RUL
    plt.subplot(2, 2, 2)
    sns.lineplot(data=df, x='RUL', y='sensor_vib', marker='o', color='blue')
    plt.gca().invert_xaxis()
    plt.title('Vibration vs. Machine Life')

    # Chart 3: Correlation
    plt.subplot(2, 2, 3)
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Map')

    plt.tight_layout()
    
    # Save the plot
    output_path = 'maintenance_dashboard.png'
    plt.savefig(output_path)
    print(f"✅ Success! Dashboard saved as: {output_path}")

    # Window open karne ki koshish
    print("--- Opening Plot Window ---")
    plt.show()

if __name__ == "__main__":
    create_plots()