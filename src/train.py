import pandas as pd
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model(input_path: str, model_output_dir: str):
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Save test data for evaluation and SHAP
    os.makedirs("data", exist_ok=True)
    X_test.to_csv("data/X_test.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)
    
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Customer_Churn_Prediction")
    
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42, n_estimators=100)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log to MLflow
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("accuracy", accuracy)
        
        # Log model in MLflow Model Registry
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        # Save locally as well
        os.makedirs(model_output_dir, exist_ok=True)
        model_path = os.path.join(model_output_dir, "churn_model.pkl")
        joblib.dump(model, model_path)
        
        print(f"Model trained with Accuracy: {accuracy:.4f}")
        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    input_csv = "../data/featured_data.csv"
    output_dir = "../models"
    
    if not os.path.exists(input_csv):
        input_csv = "data/featured_data.csv"
        output_dir = "models"
        
    if os.path.exists(input_csv):
        train_model(input_csv, output_dir)
    else:
        print("Input data not found. Run feature_engineering.py first.")
