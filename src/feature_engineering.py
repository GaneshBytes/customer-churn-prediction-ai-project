import pandas as pd
import os

def engineer_features(input_path: str, output_path: str):
    """
    Perform feature engineering.
    For this baseline, we just pass the data through, but this script
    is reserved for future feature generation (e.g. crossing features).
    """
    print(f"Loading data for feature engineering from {input_path}")
    df = pd.read_csv(input_path)
    
    # Example feature engineering (placeholder)
    # df["Tenure_MonthlyCharges_Ratio"] = df["tenure"] / (df["MonthlyCharges"] + 1e-5)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Feature-engineered data saved to {output_path}")

if __name__ == "__main__":
    input_csv = "../data/processed_data.csv"
    output_csv = "../data/featured_data.csv"
    
    if not os.path.exists(input_csv):
        input_csv = "data/processed_data.csv"
        output_csv = "data/featured_data.csv"

    if os.path.exists(input_csv):
        engineer_features(input_csv, output_csv)
    else:
        print("Input file not found. Run data_preprocessing.py first.")
