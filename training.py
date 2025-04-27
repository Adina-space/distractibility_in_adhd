from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer
import numpy as np

class Trainer:
    def __init__(self, models, scoring=None, n_splits=5, random_state=42):
        """
        models: dict of model_name: sklearn_model
        scoring: dict of scoring_name: scorer
        """
        self.models = models
        self.scoring = scoring or {
            "Accuracy": make_scorer(accuracy_score),
            "F1-score": make_scorer(f1_score),
            "ROC AUC": "roc_auc"
        }
        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.results = {}
        self.best_model = None
        self.best_model_name = None

    def train(self, X, y):
        """
        Train and evaluate all models using cross-validation.
        Stores results in self.results
        """
        for model_name, model in self.models.items():
            print(f"\n🔍 Training: {model_name}")
            self.results[model_name] = {}
            
            for score_name, scorer in self.scoring.items():
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', model)
                ])
                scores = cross_val_score(pipeline, X, y, cv=self.cv, scoring=scorer)
                mean_score = scores.mean()
                std_score = scores.std()
                self.results[model_name][score_name] = (mean_score, std_score)
                print(f"{score_name}: {mean_score:.3f} ± {std_score:.3f}")

        # After training all models, determine the best model based on ROC AUC
        self.best_model_name = max(self.results, key=lambda model: self.results[model]["ROC AUC"][0])
        self.best_model = self.models[self.best_model_name]
        print(f"\nBest model based on ROC AUC: {self.best_model_name}")

    def get_results(self):
        """Return stored CV results"""
        return self.results

    def retrain_on_full_dataset(self, X, y):
        """
        Retrain the best model on the entire dataset.
        This does not perform cross-validation.
        """
        if not self.best_model:
            print("No best model has been selected. Please run training first.")
            return None

        print(f"Training the best model ({self.best_model_name}) on the full dataset")
        # Pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', self.best_model)
        ])
        # Fit on the full dataset
        pipeline.fit(X, y)
        return pipeline

