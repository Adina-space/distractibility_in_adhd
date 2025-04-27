import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

class EyeTrackingAnalysis:
    def __init__(self, data, exp_data):
        """
        Initialize the analysis class with the dataset.
        
        :param data: pandas DataFrame containing the dataset
        """
        self.data = data
        self.exp_data = exp_data

    def plot_boxplot(self, data, x_col, y_col, title, xlabel, ylabel):
        """Plot boxplot for any given variable"""
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=data, x=x_col, y=y_col)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def distractibility_ttest(self):
        """Perform two-sample t-test with Welch adjustment on distractibility"""
        adhd_data = self.data[self.data['Group'] == 'ADHD']
        neuro_data = self.data[self.data['Group'] == 'NT']
        
        # Extract distractibility values
        adhd_distractibility = adhd_data['distractors_dwell_time_percent']
        neuro_distractibility = neuro_data['distractors_dwell_time_percent']
        
        t_stat, p_val = stats.ttest_ind(adhd_distractibility, neuro_distractibility, equal_var=False)
        return t_stat, p_val
        
    def preprocess_data_for_anova(self):
        """Preprocess the data to average target_dwell_time_percent across clips for each participant and condition"""
        data_grouped = self.exp_data.groupby(['Participant', 'Group', 'dot_car', 'gray_background'], as_index=False)['target_dwell_time_percent'].mean()
        return data_grouped
      
    def tracking_performance_mixed_anova(self):
        """Perform Mixed-ANOVA on target-tracking performance for the 2D task"""
        
        # Ensure that the data is in the correct format and includes only relevant columns
        data = self.preprocess_data_for_anova()
        
        # Convert categorical columns to appropriate types
        data['Group'] = pd.Categorical(data['Group'], categories=['ADHD', 'NT'])
        data['dot_car'] = pd.Categorical(data['dot_car'], categories=['Car', 'Dot'])
        data['gray_background'] = pd.Categorical(data['gray_background'], categories=['Gray', 'Scene'])
        
        # Perform the Mixed-ANOVA for the 2D task
        model = sm.formula.ols('target_dwell_time_percent ~ C(Group) * C(dot_car) * C(gray_background)', data=data).fit()
        
        # Perform the ANOVA and return the table
        anova_table = anova_lm(model, typ=2)
        
        return anova_table

    def anti_saccade_mannwhitney(self):
        """Perform Mann-Whitney-Wilcoxon test for errors in the anti-saccade task"""
        adhd_data = self.data[self.data['Group'] == 'ADHD']
        neuro_data = self.data[self.data['Group'] == 'NT']
        
        adhd_errors = adhd_data['error rate']
        neuro_errors = neuro_data['error rate']
        
        u_stat, p_val = stats.mannwhitneyu(adhd_errors, neuro_errors, alternative='two-sided')
        return u_stat, p_val

    def one_d_tracking_ttest(self):
        """Perform two-sample t-test with Welch adjustment for 1D object tracking performance"""
        adhd_data = self.data[self.data['Group'] == 'ADHD']
        neuro_data = self.data[self.data['Group'] == 'NT']
        
        adhd_tracking = adhd_data['sp_dwell_time_percent']
        neuro_tracking = neuro_data['sp_dwell_time_percent']
        
        t_stat, p_val = stats.ttest_ind(adhd_tracking, neuro_tracking, equal_var=False)
        return t_stat, p_val

    def correlation_pearson(self):
        """Perform Pearson correlation between different measures"""
        distractibility = self.data['distractors_dwell_time_percent']
        one_d_tracking_performance = self.data['sp_dwell_time_percent']
        two_d_tracking_performance = self.data['target_dwell_time_percent']
        anti_saccade_errors = self.data['error rate']
        
        corr_distractibility_tracking = stats.pearsonr(distractibility, one_d_tracking_performance)
        corr_distractibility_errors = stats.pearsonr(distractibility, anti_saccade_errors)
        corr_tracking_errors = stats.pearsonr(one_d_tracking_performance, anti_saccade_errors)
        corr_two_d_tracking = stats.pearsonr(distractibility, two_d_tracking_performance)
        
        return corr_distractibility_tracking, corr_distractibility_errors, corr_tracking_errors, corr_two_d_tracking
		

    def plot_correlation_matrix(self):
        """Plot correlation matrix for continuous variables"""
        correlation_matrix = self.data[['distractors_dwell_time_percent', 
        'sp_dwell_time_percent','target_dwell_time_percent', 
		'error rate']].corr()
        plt.figure(figsize=(6, 4))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.xticks(rotation=45)
        plt.title("Correlation Matrix")
        plt.savefig(f"correlation_matrix.png", dpi=300)
        plt.show()