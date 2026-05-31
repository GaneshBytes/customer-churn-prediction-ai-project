import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_shap_explanations(model_path, data_path, output_dir, instance_idx=0):
    print("Generating SHAP Explanations...")
    model = joblib.load(model_path)
    X_test = pd.read_csv(data_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Integration: Build TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Random Forest returns a list of shap_values [class 0, class 1]
    if isinstance(shap_values, list):
        shap_values_churn = shap_values[1]
        base_value_churn = explainer.expected_value[1]
    elif len(shap_values.shape) == 3:
        shap_values_churn = shap_values[:, :, 1]
        base_value_churn = explainer.expected_value[1]
    else:
        shap_values_churn = shap_values
        base_value_churn = explainer.expected_value
        
    print("Global Explanations...")
    # 2. Summary Plot (All points)
    plt.figure()
    shap.summary_plot(shap_values_churn, X_test, show=False)
    plt.savefig(os.path.join(output_dir, "shap_summary.png"), bbox_inches='tight')
    plt.close()
    
    # 3. Mean SHAP Plot
    plt.figure()
    shap.summary_plot(shap_values_churn, X_test, plot_type="bar", show=False)
    plt.savefig(os.path.join(output_dir, "shap_mean_bar.png"), bbox_inches='tight')
    plt.close()
    
    print(f"Local Explanations for instance {instance_idx}...")
    instance = X_test.iloc[instance_idx]
    shap_val_instance = shap_values_churn[instance_idx]
    
    # 4. Waterfall Plot
    plt.figure()
    explanation = shap.Explanation(
        values=shap_val_instance,
        base_values=base_value_churn,
        data=instance,
        feature_names=X_test.columns
    )
    shap.plots.waterfall(explanation, show=False)
    plt.savefig(os.path.join(output_dir, f"shap_waterfall_inst{instance_idx}.png"), bbox_inches='tight')
    plt.close()
    
    # 5. Force Plot
    # Force plot usually requires JS, but can be saved as HTML
    force_html = shap.force_plot(
        base_value_churn, 
        shap_val_instance, 
        instance, 
        matplotlib=False
    )
    shap.save_html(os.path.join(output_dir, f"shap_force_inst{instance_idx}.html"), force_html)
    
    # 6. Beeswarm Plot (requires Explanation object over all dataset for beeswarm)
    plt.figure()
    exp_all = shap.Explanation(
        values=shap_values_churn,
        base_values=base_value_churn,
        data=X_test,
        feature_names=X_test.columns
    )
    shap.plots.beeswarm(exp_all, show=False)
    plt.savefig(os.path.join(output_dir, "shap_beeswarm.png"), bbox_inches='tight')
    plt.close()
    
    # 7. Dependence Plot (for a key feature, e.g. MonthlyCharges)
    plt.figure()
    shap.dependence_plot("MonthlyCharges", shap_values_churn, X_test, show=False)
    plt.savefig(os.path.join(output_dir, "shap_dependence_monthlycharges.png"), bbox_inches='tight')
    plt.close()
    
    print(f"All SHAP explanations saved to {output_dir}")

if __name__ == "__main__":
    m_path = "../models/churn_model.pkl"
    x_path = "../data/X_test.csv"
    o_dir = "../reports/figures"
    
    if not os.path.exists(m_path):
        m_path = "models/churn_model.pkl"
        x_path = "data/X_test.csv"
        o_dir = "reports/figures"
        
    if os.path.exists(m_path) and os.path.exists(x_path):
        generate_shap_explanations(m_path, x_path, o_dir, instance_idx=15)
    else:
        print("Model or test data missing.")
