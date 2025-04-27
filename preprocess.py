import pandas as pd


def get_calculated_measurements():
  data = pd.read_csv('/data/data_by_trial_2_2_exp_1_2.csv')
  data = data[data.exp_num == 1.0]

  by_participant_columns = ['Participant','sp_time_duration_ms','sp_dwell_time_ms','target_pach_dwell_time_ms','pach_clip_time_duration_ms',
                          'distractors_pach_dwell_time_ms','distractors_duration','sp_TR[%]','is_ADHD']
  data = data[by_participant_columns].copy()
  data['Group'] = data['is_ADHD'].map({0.0: 'NT', 1.0: 'ADHD'})

  data['sp_dwell_time_percent'] = data.sp_dwell_time_ms / data.sp_time_duration_ms
  data['target_dwell_time_percent'] = data.target_pach_dwell_time_ms / data.pach_clip_time_duration_ms
  data['distractors_dwell_time_percent'] = data.distractors_pach_dwell_time_ms / data.distractors_duration

  redundant_columns = ['target_pach_dwell_time_ms','pach_clip_time_duration_ms',
 'distractors_pach_dwell_time_ms','distractors_duration','sp_time_duration_ms','sp_dwell_time_ms']
  data.drop(columns=redundant_columns, inplace=True)
  return data.drop_duplicates()
  
  
def get_full_experiments_data():
    df = pd.read_csv('/data/data_by_trial_2_2_exp_1_2.csv')
    df['Group'] = df['is_ADHD'].map({0.0: 'NT', 1.0: 'ADHD'})
    df['target_dwell_time_percent'] = (df.target_dwell_time_ms / df.clip_time_duration_ms) * 100
    return df

  
  
def get_anti_saccade_data():
  sc_df = pd.read_csv('/data/anti_saccade_exp_1.csv')
  sc_df['Group'] = sc_df['is_ADHD'].map({0.0: 'NT', 1.0: 'ADHD'})
  return sc_df


def get_flattened_experiments():
  df= pd.read_csv('data_by_trial_2_2_exp_1_2.csv')
  by_experiment_columns = ['Participant','AOI Name','target_dwell_time_ms','clip_time_duration_ms','dot_car','gray_background','is_ADHD']
  df = df[by_experiment_columns].copy()
  df['target_dwell_time_percent'] = df.target_dwell_time_ms / df.clip_time_duration_ms

  # Step 1: Create a new column that encodes the experiment name
  df['experiment_name'] = (
      df['AOI Name'] + '_' +
      df['dot_car'].str.lower() + '_' +
      df['gray_background'].str.lower() +
      '_dwell_time_percent'
  )

  wide_df = df.pivot_table(
      index=['Participant', 'is_ADHD'],
      columns='experiment_name',
      values='target_dwell_time_percent'
  ).reset_index()

  return wide_df

