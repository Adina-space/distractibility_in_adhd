import seaborn as sns
import matplotlib.pyplot as plt
from preprocess import get_calculated_measurements

GROUPS_PALLETE = {'NT': '#a6c1ee', 'ADHD': '#f7b7a3'}

def box_plot(df, y_column, title, measurement_desc):
  df = df[['Group','Participant',y_column]].drop_duplicates()
  dot_palette = {'NT': '#6b91d6', 'ADHD': '#d67c68'}

  plt.figure(figsize=(4, 4))
  sns.boxplot(x='Group', y=y_column, data=df, hue='Group', palette=GROUPS_PALLETE)
  sns.stripplot(x='Group', y=y_column, data=df, hue='Group',
              palette=dot_palette, jitter=True, alpha=0.6, marker='o',
              edgecolor='black', linewidth=0.5)

  # Remove duplicate legend
  plt.legend([],[], frameon=False)

  plt.title(f"{title}: ADHD vs. Neurotypical")
  plt.ylabel(f"{measurement_desc} (%)")
  plt.xlabel("Participant Group")
  plt.tight_layout()
  plt.savefig(f"{title.lower().replace(' ','_')}.png", dpi=300)
  plt.show()


def plot_tracking_performance(df, y_column='target_dwell_time_percent'):
  df['gray_background'] = df['gray_background'].replace({
      'Background': 'Scene',
      'Gray': 'Gray'
  })

  # Set compact style
  sns.set(style="whitegrid", font_scale=0.8)

  # Plot with group separation by ADHD vs NT, using standard error as error bars
  g = sns.catplot(
      data=df,
      kind="bar",
      x="gray_background", 
      y=y_column,
      hue="Group",
      col="dot_car",
      errorbar="se",
      palette=GROUPS_PALLETE,
      estimator="mean",
      height=4,
      aspect=0.8,
      dodge=True,
      legend=True
  )

  # Customize labels and titles
  g.legend.set_loc('upper right')
  g.legend.set_title('')
  g.set_titles("{col_name}")
  g.set_axis_labels("", "Avg. time on Target (%)")
  g.set(ylim=(0, df[y_column].max() * 1.1))
  g.despine(left=True)
  plt.tight_layout()
  plt.savefig("tracking_performance.png", dpi=300)
  plt.show()


