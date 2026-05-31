import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model_path: str, x_test_path: str, y_test_path: str):
    print("Evaluating model...")
    model = joblib.load(model_path)
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("Accuracy:", accuracy)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    
    os.makedirs("../reports/figures", exist_ok=True)
    out_img = "../reports/figures/confusion_matrix.png"
    if not os.path.exists("../reports/figures"):
        os.makedirs("reports/figures", exist_ok=True)
        out_img = "reports/figures/confusion_matrix.png"
        
    plt.savefig(out_img)
    print(f"Confusion matrix saved to {out_img}")

if __name__ == "__main__":
    m_path = "../models/churn_model.pkl"
    x_path = "../data/X_test.csv"
    y_path = "../data/y_test.csv"
    
    if not os.path.exists(m_path):
        m_path = "models/churn_model.pkl"
        x_path = "data/X_test.csv"
        y_path = "data/y_test.csv"
        
    if os.path.exists(m_path) and os.path.exists(x_path):
        evaluate_model(m_path, x_path, y_path)
    else:
        print("Required files not found. Run train.py first.")
