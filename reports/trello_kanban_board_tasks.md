# Trello Kanban Board: Customer Churn Prediction Project

*Note: You can copy these tasks into a Trello board to simulate managing this AI project.*

## Columns / Lists
1. **Backlog**: Tasks that are planned but not yet prioritized for the current sprint.
2. **To Do (Sprint)**: Tasks selected for the current sprint.
3. **In Progress**: Tasks actively being worked on.
4. **Review / QA**: Tasks completed by the assignee, awaiting code review or validation.
5. **Done**: Fully completed tasks.

---

## Epic 1: Business Understanding & Project Setup
- **[Task]** Define Strategic Objectives and KPIs with Stakeholders
  - *Description*: Meet with Marketing VP to define exact success metrics for churn reduction.
- **[Task]** Set up GitHub Repository and Project Structure
  - *Description*: Initialize repo using cookiecutter-data-science template and set up Git workflows.
- **[Task]** Provision Cloud Resources / Environment Setup
  - *Description*: Set up local environment (`requirements.txt`), MLflow tracking server, and development environment.

## Epic 2: Data Understanding & Preparation
- **[Task]** Data Ingestion
  - *Description*: Load the IBM Telco Customer Churn dataset into pandas/dask dataframes.
- **[Task]** Exploratory Data Analysis (EDA)
  - *Description*: Analyze distributions, handle missing values (e.g., in TotalCharges), and identify correlations.
- **[Task]** Feature Engineering
  - *Description*: Create new features, encode categorical variables (One-Hot Encoding, Label Encoding), and scale numerical features.
- **[Task]** Train/Test Split & Handling Imbalance
  - *Description*: Split data maintaining stratification. Implement SMOTE or class weighting if necessary.

## Epic 3: Modeling & Evaluation
- **[Task]** Baseline Model (Logistic Regression)
  - *Description*: Train a simple baseline model to establish a minimum performance threshold. Log metrics with MLflow.
- **[Task]** Advanced Modeling (Random Forest)
  - *Description*: Train a Random Forest classifier. Perform hyperparameter tuning. Log runs with MLflow.
- **[Task]** Model Evaluation
  - *Description*: Evaluate models on the test set using Accuracy, Precision, Recall, F1-Score, and ROC-AUC. Select the best model.
- **[Task]** Packaging with MLflow Projects
  - *Description*: Create an `MLproject` file to ensure the training pipeline is reproducible.

## Epic 4: Explainable AI (XAI)
- **[Task]** Implement SHAP Global Explanations
  - *Description*: Generate SHAP Summary plot and Mean SHAP plot to explain overall feature importance.
- **[Task]** Implement SHAP Local Explanations
  - *Description*: Generate Waterfall, Force, Beeswarm, and Dependence plots for specific at-risk customers to provide actionable insights to marketing.
- **[Task]** XAI Report Documentation
  - *Description*: Compile SHAP visualizations into a business-friendly report.

## Epic 5: Deployment & MLOps
- **[Task]** Code Formatting and Linting (PEP8)
  - *Description*: Run `black` and `flake8` to ensure codebase meets PEP8 standards.
- **[Task]** Automated Documentation
  - *Description*: Set up Sphinx to generate documentation from docstrings.
- **[Task]** Local Model Serving
  - *Description*: Expose the trained model as a REST API using `mlflow models serve`.
- **[Task]** Model Registry
  - *Description*: Register the best performing model in the MLflow Model Registry and transition to "Production" stage.
- **[Task] (Bonus)** Cloud Deployment
  - *Description*: Containerize the model and deploy to a cloud service (AWS / GCP / Azure).
