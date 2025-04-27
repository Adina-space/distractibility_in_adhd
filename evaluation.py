import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Evaluator:
    def __init__(self, results):
        """
        results: dictionary in the format:
        {
            'Model Name': {
                'Metric Name': (mean, std),
                ...
            },
            ...
        }
        """
        self.results = results
        
    def get_latex_table(self):
        res = pd.DataFrame(self.results).T
        # Format the DataFrame into LaTeX table format
        latex_code = res.map(lambda x: f"{x[0]:.3f} ± {x[1]:.3f}")
        # Convert the DataFrame to LaTeX table
        latex_table = latex_code.to_latex(header=True,index=True,column_format='|l|c|c|c|',bold_rows=True)
        return latex_table

    def plot_metrics(self, metrics=["Accuracy", "F1-score", "ROC AUC"]):
        """
        Plot bar charts for each specified metric across models.
        """
        
        colors = ['#A1C9F4', '#FFB482', '#8DE5A1']
        
        for metric in metrics:
            means = []
            stds = []
            labels = []

            for model, scores in self.results.items():
                if metric in scores:
                    mean, std = scores[metric]
                    means.append(mean)
                    stds.append(std)
                    labels.append(model)

            plt.figure(figsize=(4.5, 2.8))
            bars = plt.bar(labels, means, yerr=stds, capsize=4,
                           color=colors[:len(labels)], edgecolor='black', linewidth=0.5)
            plt.title(f"{metric}", fontsize=10, pad=6)
            plt.ylabel(metric, fontsize=9)
            plt.xticks(rotation=10, fontsize=8)
            plt.yticks(fontsize=8)
            plt.ylim(0, 1)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout(pad=1)
            plt.show()


    def plot_combined_metrics(self, metrics=["Accuracy", "F1-score", "ROC AUC"]):

        metric_colors = {
            "Accuracy": '#A1C9F4', 
            "F1-score": '#FFB482', 
            "ROC AUC": '#8DE5A1'
        }

        model_names = list(self.results.keys())
        x = np.arange(len(model_names))
        width = 0.25
        offsets = [-width, 0, width]

        fig, ax = plt.subplots(figsize=(6.5, 3.5))

        for i, metric in enumerate(metrics):
            means = [self.results[model][metric][0] for model in model_names]
            stds = [self.results[model][metric][1] for model in model_names]

            ax.bar(x + offsets[i], means, width,
                   label=metric,
                   yerr=stds,
                   capsize=4,
                   color=metric_colors[metric],
                   edgecolor='black',
                   linewidth=0.5)

        ax.set_ylabel('Score', fontsize=9)
        ax.set_title('Model Performance Comparison', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=9)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_ylim(0, 1)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.legend(fontsize=8, frameon=False, loc='lower right')
        plt.tight_layout()
        plt.show()
