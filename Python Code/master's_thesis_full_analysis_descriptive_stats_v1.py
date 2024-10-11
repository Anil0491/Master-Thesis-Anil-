# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 18:47:31 2024

@author: Palan
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import ttest_rel, wilcoxon

# Function to calculate Cohen's d
def cohen_d(x, y):
    diff = x - y
    return np.mean(diff) / np.std(diff, ddof=1)

# Calculate the smallest worthwhile change (SWC)
def swc(sd):
    return 0.2 * sd

# Function to calculate confidence intervals
def confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    se = np.std(data, ddof=1) / np.sqrt(n)
    h = se * 1.96
    return mean - h, mean + h

# Define file paths for three different data sources
imu_data_raw = r"C:\Users\Palan\OneDrive\Documents\School\McMaster\1.Master's Thesis\Thesis_Data_Analysis\I.M.U data\StepSessionSummaryExport"
force_plate = r"C:\Users\Palan\OneDrive\Documents\School\McMaster\1.Master's Thesis\Thesis_Data_Analysis\raw_data\forcedecks-test-export-07_07_2024.xlsx"
srpe = r"C:\Users\Palan\OneDrive\Documents\School\McMaster\1.Master's Thesis\Thesis_Data_Analysis\SRPE\McMaster MBB 2023 - Sessional RPE (Study Fall-2023) (Responses) - Form Responses 1.xlsx"
output_excel_path = r"C:\Users\Palan\OneDrive\Documents\School\McMaster\1.Master's Thesis\Thesis_Data_Analysis\raw_data\descriptive_stats_with_additional_info.xlsx"

# Initialize an empty DataFrame to store IMU data
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

# Read the Excel files for Force Plate and SRPE Data
force_plate = pd.read_excel(force_plate)
srpe = pd.read_excel(srpe)

# Combine the "First Name" and "Last Name" columns in the imu_data DataFrame
imu_data['Name'] = imu_data['First Name'] + ' ' + imu_data['Last Name']
imu_data.rename(columns={'Date (YYYY-MM-DD)': 'Date'}, inplace=True)

# Replace specific names in the 'Name' column
imu_data['Name'] = imu_data['Name'].replace({'Kazim Raza1': 'Kazim Raza', 'Brendan Amoyaw1': 'Brendan Amoyaw'})
srpe['Name'] = srpe['Name'].replace({'Moody Mohamud': 'Moody Muhammed', 'Thomas Matsell': 'Thomas Mattsel'})

# Select relevant columns
imu_data = imu_data[['Name', 'Date', 'Footnote', 'Impact Load Total (L+R)', 'Average Intensity (L and R)']]
force_plate = force_plate[['Name', 'ExternalId', 'Test Type', 'Date', 'Time', 'BW [KG]', 'Reps', 'Tags', 'Additional Load [kg]', 'Jump Height (Imp-Mom) in Inches [in] ',
'Eccentric Mean Braking Force [N] ', 'Eccentric Braking RFD / BM [N/s/kg] ', 'Eccentric Braking Impulse [N s] ', 'Force at Zero Velocity / BM [N/kg] ', 'Concentric Mean Force / BM [N/kg] ',
'Concentric RFD / BM [N/s/kg] ', 'Concentric Impulse (Abs) / BM [N s] ', 'Concentric Mean Force [N] ']]
srpe = srpe[['Name', 'Date', 'SRPE']]

# Standardize date formats across datasets
imu_data['Date'] = pd.to_datetime(imu_data['Date']).dt.strftime('%Y-%m-%d')
force_plate['Date'] = pd.to_datetime(force_plate['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
srpe['Date'] = pd.to_datetime(srpe['Date'], errors='coerce').dt.strftime('%Y-%m-%d')

# Merge datasets
complete_data = imu_data.merge(force_plate, on=['Name', 'Date'], how='outer').merge(srpe, on=['Name', 'Date'], how='outer')
all_footnote_data = complete_data[complete_data['Footnote'] == 'All']

# Filter data for 'Pre-Practice' and 'Post-Practice'
pre_data = all_footnote_data[all_footnote_data['Tags'] == 'Pre-Practice']
post_data = all_footnote_data[all_footnote_data['Tags'] == 'Post-Practice']

# Get the column names that are metrics
metric_columns = [col for col in post_data.columns if col not in ['Name', 'Date', 'Tags']]

# Modify the metric column names to add 'Post-Practice'
for metric in metric_columns:
    new_metric_name = f'{metric}_Post-Practice'
    post_data = post_data.rename(columns={metric: new_metric_name})

# Drop unnecessary columns if they exist
columns_to_remove = ['Footnote_Post-Practice', 'Period Start Time (24H)_Post-Practice', 'Impact Load Total (L+R)_Post-Practice', 'Impact Load Total per Minute (L and R)_Post-Practice', 'Average Intensity (L and R)_Post-Practice', 'SRPE_Post-Practice']
post_data = post_data.drop(columns=[col for col in columns_to_remove if col in post_data.columns], errors='ignore')

# Merge the pre and post data
pre_post_data = pre_data.merge(post_data, on=['Name', 'Date'], suffixes=('_Pre', '_Post'), how='inner')

# Define the mapping of names to positions
positions = {'Brendan Amoyaw': 'Forward','Cashius McNeilly': 'Guard','Daniel Graham': 'Guard','Elijah Bethune': 'Guard','Jeremiah Francis': 'Guard','Kazim Raza': 'Guard','Matthew Groe': 'Guard','Mike Demagus': 'Guard','Moody Muhammed': 'Forward','Parker Davis': 'Guard','Riaz Saliu': 'Forward','Sebastian Di Manno': 'Guard','Stevan Japundzic': 'Forward','Thomas Mattsel': 'Forward'}

# Add 'Position' and 'PlayerID' columns
pre_post_data['Position'] = pre_post_data['Name'].map(positions)
unique_names = pre_post_data['Name'].unique()
player_ids = {name: idx for idx, name in enumerate(unique_names, 1)}
pre_post_data['PlayerID'] = pre_post_data['Name'].map(player_ids)

# Create a new DataFrame without the 'Name' column
final_data_set = pre_post_data.drop(columns=['Name'])

# Columns to keep as identifiers
id_vars = ['PlayerID', 'Date', 'Footnote']

# Variables measured both pre and post
value_vars = ['Jump Height (Imp-Mom) in Inches [in] ',
'Eccentric Mean Braking Force [N] ',
'Eccentric Braking RFD / BM [N/s/kg] ',
'Eccentric Braking Impulse [N s] ',
'Force at Zero Velocity / BM [N/kg] ',
'Concentric Mean Force / BM [N/kg] ', 'Concentric RFD / BM [N/s/kg] ',
'Concentric Impulse (Abs) / BM [N s] ', 'Concentric Mean Force [N] ']

# Melting the dataframe
linear_mixed_model = pd.melt(final_data_set, id_vars=id_vars, value_vars=value_vars + [f"{var}_Post-Practice" for var in value_vars], var_name='Measurement_Type', value_name='Value')

# Create a new 'Time' variable to indicate pre- or post-practice
linear_mixed_model['Time'] = linear_mixed_model['Measurement_Type'].apply(lambda x: 'Post' if 'Post-Practice' in x else 'Pre')

# Adjust 'Measurement_Type' to have consistent naming for pre and post
linear_mixed_model['Measurement_Type'] = linear_mixed_model['Measurement_Type'].str.replace('_Post-Practice', '')

# Ensure 'Position' is included
linear_mixed_model = linear_mixed_model.merge(pre_post_data[['PlayerID', 'Position']], on='PlayerID', how='left')

# Remove duplicates
linear_mixed_model = linear_mixed_model.drop_duplicates()

# Statistical outlier detection and removal using IQR
def remove_outliers_iqr(df, value_vars):
    for var in value_vars:
        for time in ['Pre', 'Post']:
            subset = df[(df['Measurement_Type'] == var) & (df['Time'] == time)]
            Q1 = subset['Value'].quantile(0.25)
            Q3 = subset['Value'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identify outliers
            outliers = df[(df['Measurement_Type'] == var) & (df['Time'] == time) & ((df['Value'] < lower_bound) | (df['Value'] > upper_bound))]
            
            # Remove outliers
            df = df[~((df['Measurement_Type'] == var) & (df['Time'] == time) & ((df['Value'] < lower_bound) | (df['Value'] > upper_bound)))]
        
    return df

# Remove outliers from the data
linear_mixed_model_clean = remove_outliers_iqr(linear_mixed_model, value_vars)

# Descriptive statistics for the cleaned data (means and standard deviations only)
descriptive_stats = linear_mixed_model_clean.groupby(['Position', 'Measurement_Type', 'Time'])['Value'].agg(['mean', 'std']).unstack()

# Reorder columns to have 'Pre' before 'Post'
ordered_columns = [('mean', 'Pre'), ('mean', 'Post'), ('std', 'Pre'), ('std', 'Post')]
descriptive_stats = descriptive_stats[ordered_columns]

# Adding Sample Size and Confidence Intervals to Descriptive Stats
descriptive_stats['n_Pre'] = linear_mixed_model_clean[linear_mixed_model_clean['Time'] == 'Pre'].groupby(['Position', 'Measurement_Type'])['Value'].count().values
descriptive_stats['n_Post'] = linear_mixed_model_clean[linear_mixed_model_clean['Time'] == 'Post'].groupby(['Position', 'Measurement_Type'])['Value'].count().values

pre_means = linear_mixed_model_clean[linear_mixed_model_clean['Time'] == 'Pre'].groupby(['Position', 'Measurement_Type'])['Value'].apply(confidence_interval).apply(pd.Series)
post_means = linear_mixed_model_clean[linear_mixed_model_clean['Time'] == 'Post'].groupby(['Position', 'Measurement_Type'])['Value'].apply(confidence_interval).apply(pd.Series)

descriptive_stats['CI_Lower_Pre'] = pre_means[0].values
descriptive_stats['CI_Upper_Pre'] = pre_means[1].values
descriptive_stats['CI_Lower_Post'] = post_means[0].values
descriptive_stats['CI_Upper_Post'] = post_means[1].values

# Adding Median and IQR
descriptive_stats['Median_Pre'] = linear_mixed_model_clean[linear_mixed_model_clean['Time'] == 'Pre'].groupby(['Position', 'Measurement_Type'])['Value'].median().values
descriptive_stats['Median_Post'] = linear_mixed_model_clean[linear_mixed_model_clean['Time'] == 'Post'].groupby(['Position', 'Measurement_Type'])['Value'].median().values
descriptive_stats['IQR_Pre'] = linear_mixed_model_clean[linear_mixed_model_clean['Time'] == 'Pre'].groupby(['Position', 'Measurement_Type'])['Value'].apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25)).values
descriptive_stats['IQR_Post'] = linear_mixed_model_clean[linear_mixed_model_clean['Time'] == 'Post'].groupby(['Position', 'Measurement_Type'])['Value'].apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25)).values

# Adding p-values from Statistical Tests
p_values_t = []
p_values_w = []

for position in ['Forward', 'Guard']:
    for var in value_vars:
        subset = linear_mixed_model_clean[(linear_mixed_model_clean['Position'] == position) & (linear_mixed_model_clean['Measurement_Type'] == var)]
        pre_values = subset[subset['Time'] == 'Pre']['Value'].dropna()
        post_values = subset[subset['Time'] == 'Post']['Value'].dropna()
        
        # Ensure equal lengths by dropping NaNs
        min_len = min(len(pre_values), len(post_values))
        pre_values = pre_values.iloc[:min_len]
        post_values = post_values.iloc[:min_len]
        
        if len(pre_values) > 0 and len(post_values) > 0:
            t_stat, p_value_t = ttest_rel(pre_values, post_values)
            w_stat, p_value_w = wilcoxon(pre_values, post_values)
        else:
            p_value_t, p_value_w = np.nan, np.nan
        
        p_values_t.append(p_value_t)
        p_values_w.append(p_value_w)

descriptive_stats['p-value (t-test)'] = p_values_t
descriptive_stats['p-value (Wilcoxon)'] = p_values_w

# Calculate Cohen's d for each position and measurement type
effect_sizes = []
swc_values = []
for position in ['Forward', 'Guard']:
    for var in value_vars:
        subset = linear_mixed_model_clean[(linear_mixed_model_clean['Position'] == position) & (linear_mixed_model_clean['Measurement_Type'] == var)]
        subset_mean = subset.groupby(['PlayerID', 'Time'])['Value'].mean().reset_index()
        pivot_data = subset_mean.pivot(index='PlayerID', columns='Time', values='Value').reset_index()
        
        # Ensure equal lengths by dropping NaNs
        pivot_data = pivot_data.dropna()
        
        cohen_d_value = cohen_d(pivot_data['Pre'], pivot_data['Post'])
        effect_sizes.append(cohen_d_value)
        swc_values.append(swc(np.std(pivot_data['Pre'], ddof=1)))

descriptive_stats['Effect Size (Cohen\'s d)'] = effect_sizes
descriptive_stats['Smallest Worthwhile Change (SWC)'] = swc_values

# Save to Excel file
descriptive_stats.to_excel(output_excel_path)
print(f"Descriptive statistics with additional information have been saved to '{output_excel_path}'.")
