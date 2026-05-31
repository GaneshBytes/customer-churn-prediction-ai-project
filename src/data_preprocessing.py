import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

def load_and_preprocess_data(input_path: str, output_path: str):
    """
    Load raw data, clean it, apply label encoding, and save the processed data.
    """
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)

    # Data Cleaning
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)

    # Feature Encoding
    le = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = le.fit_transform(df[column])

    # Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    input_csv = "../WA_Fn-UseC_-Telco-Customer-Churn.csv"
    output_csv = "../data/processed_data.csv"
    
    # Allow running from root or src
    if not os.path.exists(input_csv):
        input_csv = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        output_csv = "data/processed_data.csv"

    load_and_preprocess_data(input_csv, output_csv)
