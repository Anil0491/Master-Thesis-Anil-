# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 08:54:08 2024

@author: Palan
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import shapiro,pearsonr

# Define file paths
imu_data_raw = r"C:\Users\Palan\OneDrive\Documents\School\McMaster\1.Master's Thesis\Thesis_Data_Analysis\I.M.U data\StepSessionSummaryExport"
force_plate_path = r"C:\Users\Palan\OneDrive\Documents\School\McMaster\1.Master's Thesis\Thesis_Data_Analysis\raw_data\forcedecks-test-export-12_07_2024.xlsx"
srpe_path = r"C:\Users\Palan\OneDrive\Documents\School\McMaster\1.Master's Thesis\Thesis_Data_Analysis\SRPE\McMaster MBB 2023 - Sessional RPE (Study Fall-2023) (Responses) - Form Responses 1.xlsx"

# Load IMU data
imu_data = pd.DataFrame()
for subfolder in os.listdir(imu_data_raw):
    subfolder_path = os.path.join(imu_data_raw, subfolder)
    if os.path.isdir(subfolder_path):
        for date_folder in os.listdir(subfolder_path):
            date_folder_path = os.path.join(subfolder_path, date_folder)
            if os.path.isdir(date_folder_path):
                for root, dirs, files in os.walk(date_folder_path):
                    for file in files:
                        if file.endswith(".csv"):
                            imu_sheet_path = os.path.join(root, file)
                            temp_imu_table = pd.read_csv(imu_sheet_path)
                            imu_data = pd.concat([imu_data, temp_imu_table], ignore_index=True)

# Load Force Plate and SRPE data
force_plate = pd.read_excel(force_plate_path)
srpe = pd.read_excel(srpe_path)

# Combine "First Name" and "Last Name" in IMU data
imu_data['Name'] = imu_data['First Name'] + ' ' + imu_data['Last Name']
imu_data.rename(columns={'Date (YYYY-MM-DD)': 'Date'}, inplace=True)

# Replace specific names
name_replacements = {'Kazim Raza1': 'Kazim Raza', 'Brendan Amoyaw1': 'Brendan Amoyaw', 'Moody Mohamud': 'Moody Muhammed', 'Thomas Matsell': 'Thomas Mattsel'}
imu_data['Name'].replace(name_replacements, inplace=True)
srpe['Name'].replace(name_replacements, inplace=True)

# Select relevant columns
imu_data = imu_data[['Name', 'Date', 'Footnote', 'Impact Load Total (L+R)', 'Average Intensity (L and R)']]
force_plate = force_plate[['Name', 'ExternalId', 'Test Type', 'Date', 'Time', 'BW [KG]', 'Reps',
       'Tags', 'Additional Load [kg]', 'Jump Height (Imp-Mom) in Inches [in] ',
       'RSI-modified (Imp-Mom) [m/s] ', 'Eccentric Braking RFD / BM [N/s/kg] ',
       'Eccentric Braking Impulse [N s] ',
       'Force at Zero Velocity / BM [N/kg] ',
       'Concentric Impulse (Abs) / BM [N s] ', 'Concentric Impulse [N s] ',
       'Concentric RFD / BM [N/s/kg] ']]
srpe = srpe[['Name', 'Date', 'SRPE']]

# Standardize date formats
imu_data['Date'] = pd.to_datetime(imu_data['Date']).dt.strftime('%Y-%m-%d')
force_plate['Date'] = pd.to_datetime(force_plate['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
srpe['Date'] = pd.to_datetime(srpe['Date'], errors='coerce').dt.strftime('%Y-%m-%d')

# Merge datasets
complete_data = imu_data.merge(force_plate, on=['Name', 'Date'], how='outer').merge(srpe, on=['Name', 'Date'], how='outer')
all_footnote_data = complete_data[complete_data['Footnote'] == 'All']

# Filter and rename columns
pre_data = all_footnote_data[all_footnote_data['Tags'] == 'Pre-Practice']
post_data = all_footnote_data[all_footnote_data['Tags'] == 'Post-Practice']
metric_columns = [col for col in post_data.columns if col not in ['Name', 'Date', 'Tags']]
post_data = post_data.rename(columns={metric: f'{metric}_Post-Practice' for metric in metric_columns})
columns_to_remove = ['Footnote_Post-Practice', 'Impact Load Total per Minute (L and R)_Post-Practice']
post_data = post_data.drop(columns=[col for col in columns_to_remove if col in post_data.columns], errors='ignore')
pre_post_data = pre_data.merge(post_data, on=['Name', 'Date'], suffixes=('_Pre', '_Post'), how='inner')

# Map names to positions
positions = {'Brendan Amoyaw': 'Forward', 'Cashius McNeilly': 'Guard', 'Daniel Graham': 'Guard', 'Elijah Bethune': 'Guard', 'Jeremiah Francis': 'Guard', 
             'Kazim Raza': 'Guard', 'Matthew Groe': 'Guard', 'Mike Demagus': 'Guard', 'Moody Muhammed': 'Forward', 'Parker Davis': 'Guard', 
             'Riaz Saliu': 'Forward', 'Sebastian Di Manno': 'Guard', 'Stevan Japundzic': 'Forward', 'Thomas Mattsel': 'Forward'}

pre_post_data['Position'] = pre_post_data['Name'].map(positions)
unique_names = pre_post_data['Name'].unique()
player_ids = {name: idx for idx, name in enumerate(unique_names, 1)}
pre_post_data['PlayerID'] = pre_post_data['Name'].map(player_ids)
final_data_set = pre_post_data.drop(columns=['Name'])

# Calculate drop-off metrics
metrics = ['Jump Height (Imp-Mom) in Inches [in] ',
'RSI-modified (Imp-Mom) [m/s] ', 'Eccentric Braking RFD / BM [N/s/kg] ',
'Eccentric Braking Impulse [N s] ',
'Force at Zero Velocity / BM [N/kg] ',
'Concentric Impulse (Abs) / BM [N s] ', 'Concentric Impulse [N s] ',
'Concentric RFD / BM [N/s/kg] ']

for metric in metrics:
    final_data_set[f'{metric}_Drop-Off'] = final_data_set[f'{metric}'] - final_data_set[f'{metric}_Post-Practice']

# Variables for the mixed model
variables_for_model = ['Impact Load Total (L+R)', 'Average Intensity (L and R)','Jump Height (Imp-Mom) in Inches [in] ','RSI-modified (Imp-Mom) [m/s] ', 'Eccentric Braking RFD / BM [N/s/kg] ',
       'Eccentric Braking Impulse [N s] ','Force at Zero Velocity / BM [N/kg] ','Concentric Impulse (Abs) / BM [N s] ', 'Concentric Impulse [N s] ',
       'Concentric RFD / BM [N/s/kg] ', 'SRPE','Impact Load Total (L+R)_Post-Practice', 'Average Intensity (L and R)_Post-Practice', 
      'Jump Height (Imp-Mom) in Inches [in] _Post-Practice','RSI-modified (Imp-Mom) [m/s] _Post-Practice','Eccentric Braking RFD / BM [N/s/kg] _Post-Practice',
       'Eccentric Braking Impulse [N s] _Post-Practice','Force at Zero Velocity / BM [N/kg] _Post-Practice','Concentric Impulse (Abs) / BM [N s] _Post-Practice',
       'Concentric Impulse [N s] _Post-Practice','Concentric RFD / BM [N/s/kg] _Post-Practice', 'SRPE_Post-Practice',
       'Position', 'PlayerID','Jump Height (Imp-Mom) in Inches [in] _Drop-Off','RSI-modified (Imp-Mom) [m/s] _Drop-Off','Eccentric Braking RFD / BM [N/s/kg] _Drop-Off',
       'Eccentric Braking Impulse [N s] _Drop-Off','Force at Zero Velocity / BM [N/kg] _Drop-Off','Concentric Impulse (Abs) / BM [N s] _Drop-Off',
       'Concentric Impulse [N s] _Drop-Off','Concentric RFD / BM [N/s/kg] _Drop-Off']

final_data_set = final_data_set[['PlayerID', 'Date', 'Footnote', 'Position'] + variables_for_model]

# Melt the dataframe
linear_mixed_model = pd.melt(final_data_set, id_vars=['PlayerID', 'Date', 'Footnote', 'Position','Impact Load Total (L+R)', 'Average Intensity (L and R)','SRPE'], value_vars=variables_for_model, var_name='Measurement_Type', value_name='Value')
linear_mixed_model['Time'] = linear_mixed_model['Measurement_Type'].apply(lambda x: 'Post' if 'Post' in x else 'Pre')
linear_mixed_model['Measurement_Type'] = linear_mixed_model['Measurement_Type'].str.replace('_Post', '')
linear_mixed_model['Measurement_Type'] = linear_mixed_model['Measurement_Type'].str.replace('-Practice', '')  # Remove '-Practice'
linear_mixed_model = linear_mixed_model[~linear_mixed_model['Measurement_Type'].isin(['Average Intensity (L and R)_Drop-Off', 'Impact Load Total (L+R)_Drop-Off', 'SRPE','Impact Load Total (L+R)-Practice','Average Intensity (L and R)-Practice'])]
linear_mixed_model = linear_mixed_model[~linear_mixed_model['Measurement_Type'].isin(['Impact Load Total (L+R)', 'Average Intensity (L and R)'])]
linear_mixed_model = linear_mixed_model.drop_duplicates()























# Outlier removal using IQR
def remove_outliers_iqr(df, value_vars):
    for var in value_vars:
        for time in ['Pre', 'Post']:
            subset = df[(df['Measurement_Type'] == var) & (df['Time'] == time)]
            Q1 = subset['Value'].quantile(0.25)
            Q3 = subset['Value'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[~((df['Measurement_Type'] == var) & (df['Time'] == time) & ((df['Value'] < lower_bound) | (df['Value'] > upper_bound)))]
    return df

# Remove outliers
value_vars = linear_mixed_model['Measurement_Type'].unique()
linear_mixed_model = remove_outliers_iqr(linear_mixed_model, value_vars)

# Shapiro-Wilk test for normality
shapiro_results = []
for measurement in linear_mixed_model['Measurement_Type'].unique():
    subset = linear_mixed_model[linear_mixed_model['Measurement_Type'] == measurement]
    if len(subset) > 0:
        stat, p = shapiro(subset['Value'])
        shapiro_results.append({'Measurement_Type': measurement, 'Statistic': stat, 'p-value': p, 'Significant': 'Yes' if p <= 0.05 else 'No'})

shapiro_results_df = pd.DataFrame(shapiro_results)

# Calculate descriptive statistics
descriptive_stats = linear_mixed_model.groupby(['Measurement_Type', 'Time'])['Value'].describe().unstack()



# Save the linear_mixed_model DataFrame, descriptive stats, and correlation results to an Excel file
output_path = r"C:\Users\Palan\OneDrive\Documents\School\McMaster\1.Master's Thesis\Thesis_Data_Analysis\linear_mixed_model.xlsx"
with pd.ExcelWriter(output_path) as writer:
    linear_mixed_model.to_excel(writer, sheet_name='Linear Mixed Model', index=False)
    descriptive_stats.to_excel(writer, sheet_name='Descriptive Stats')
    shapiro_results_df.to_excel(writer, sheet_name='Shapiro-Wilk Test')


print(f"The linear_mixed_model DataFrame and additional analysis results have been saved to {output_path}")
