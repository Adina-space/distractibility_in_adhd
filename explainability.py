import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class SHAPExplainer:
    def __init__(self, model, X_train, feature_names):
        """
        model: Trained model (SVM, Logistic Regression, Random Forest, etc.)
        X_train: Training data (features)
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names

    def explain(self):
        """
        Explains the model using SHAP values.
        Automatically chooses the correct explainer based on model type.
        """
        if hasattr(self.model, "predict_proba"):  # Check if the model has 'predict_proba' method
            if isinstance(self.model, RandomForestClassifier):
                # For tree-based models like Random Forest, XGBoost, etc.
                self.explain_tree_based_models()
            else:
                # For non-tree-based models (e.g., SVC, Logistic Regression)
                self.explain_non_tree_based_models()
        else:
            raise ValueError("Model does not support probability prediction.")

    def explain_tree_based_models(self):
        """
        Explains tree-based models (e.g., Random Forest, XGBoost).
        """
        # Use TreeExplainer for tree-based models
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X_train)
        self.plot_summary(shap_values)

    def explain_non_tree_based_models(self):
        """
        Explains non-tree-based models (e.g., SVC, Logistic Regression).
        """
        # Use KernelExplainer for non-tree-based models like SVC, Logistic Regression
        explainer = shap.KernelExplainer(self.model.predict_proba, self.X_train)
        shap_values = explainer.shap_values(self.X_train)
        self.plot_summary(shap_values)

    def plot_summary(self, shap_values):
        """
        Create a summary plot to show feature importance.
        """
        #plt.figure(figsize=(8, 5))
        ## Class 1 for binary classification
        #shap.summary_plot(shap_values[:,:,1], self.X_train, feature_names=self.feature_names)
        #plt.savefig(f"feature_importance.png", dpi=300)
        #plt.show()
        
        mean_abs_shap = np.abs(shap_values[:, :, 1]).mean(axis=0)  # for class 1
        # Combine with feature names
        feature_importance = pd.DataFrame({
        'feature': self.feature_names,
        'mean_abs_shap': mean_abs_shap
        }).sort_values(by='mean_abs_shap', ascending=False)
        print(feature_importance)

    def plot_force(self, shap_values, instance_idx=0):
        """
        Create a force plot for a single instance to visualize individual feature contributions.
        """
        plt.figure(figsize=(10, 5))
        shap.force_plot(shap_values[1][instance_idx], self.X_train.iloc[instance_idx])
        plt.show()
