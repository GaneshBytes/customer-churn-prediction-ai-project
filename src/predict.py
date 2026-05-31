import joblib
import pandas as pd
import sys
import os

def predict_single_instance(model_path, data_path, instance_index=0):
    model = joblib.load(model_path)
    X_test = pd.read_csv(data_path)
    
    instance = X_test.iloc[[instance_index]]
    prediction = model.predict(instance)[0]
    proba = model.predict_proba(instance)[0][1]
    
    print(f"Prediction for instance {instance_index}: {'Churn' if prediction == 1 else 'No Churn'}")
    print(f"Probability of Churn: {proba:.4f}")
    return prediction, proba

if __name__ == "__main__":
    m_path = "models/churn_model.pkl" if os.path.exists("models/churn_model.pkl") else "../models/churn_model.pkl"
    x_path = "data/X_test.csv" if os.path.exists("data/X_test.csv") else "../data/X_test.csv"
    
    if os.path.exists(m_path) and os.path.exists(x_path):
        predict_single_instance(m_path, x_path, 0)
    else:
        print("Model or data not found. Run training first.")
