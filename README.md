# Customer Churn Prediction AI Project

## Project Overview

This project was developed as part of the EPITA AI Project Methodology course.

The goal of the project is to predict customer churn for an e-commerce / telecom company using Machine Learning techniques. Customer churn prediction helps companies identify customers who are likely to leave the service and allows businesses to take preventive actions.

The project also includes Explainable AI (XAI) using SHAP to better understand model predictions and feature importance.

---

## Dataset

The project uses the IBM Telco Customer Churn dataset.

Dataset source:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The dataset contains customer information such as:
- Customer demographics
- Subscription services
- Contract information
- Monthly charges
- Customer tenure
- Churn status

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- MLflow
- SHAP
- Git & GitHub

---

## Project Structure

```text
customer-churn-prediction-ai-project/
│
├── data/
├── models/
├── notebooks/
├── reports/
├── src/
├── mlruns/
├── requirements.txt
└── README.md

Machine Learning Workflow
1.Data loading
2.Data cleaning and preprocessing
3.Feature encoding
4.Train-test split
5.Model training
6.Model evaluation
7.MLflow experiment tracking
8.Explainable AI using SHAP



## Models Used
1.Logistic Regression
2.Decision Tree Classifier
3.Random Forest Classifier

-Random Forest achieved the best performance and was selected as the final model.
Explainable AI (XAI)

-SHAP was used to explain model predictions and identify the most important features influencing customer churn.

## Generated visualizations include:

SHAP summary plot
SHAP feature importance plot
SHAP waterfall plot
Results

The Random Forest model achieved approximately 79% accuracy on the test dataset.

Important churn-related features identified:

Contract type
Monthly charges
Tenure
Total charges


## Team Members & Collaboration
This project was successfully completed as a joint collaborative effort by all three team members:
- **Priscilla Gilbert**
- **Ganesh Reddy**
- **Rajat Patial**

All team members contributed equally to the Functional Methodologies (Part 1), Technical Implementation and MLOps (Part 2), and Explainable AI components (Part 3) of the Graded Project.

## Future Improvements
1.Hyperparameter tuning
2.Deployment using cloud services
3.Real-time prediction API
4.Improved feature engineering