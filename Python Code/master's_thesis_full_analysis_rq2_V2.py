# -*- coding: utf-8 -*-
"""
Created on Mon May 20 09:25:16 2024

@author: Palan
"""

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro,norm
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import mixedlm

# Define file paths for three different data sources
imu_data_raw = r"C:\Users\Palan\OneDrive\Documents\School\McMaster\1.Master's Thesis\Thesis_Data_Analysis\I.M.U data\StepSessionSummaryExport"
force_plate = r"C:\Users\Palan\OneDrive\Documents\School\McMaster\1.Master's Thesis\Thesis_Data_Analysis\Forceplate Data\forcedecks-test-export-full_v1.xlsx"
srpe = r"C:\Users\Palan\OneDrive\Documents\School\McMaster\1.Master's Thesis\Thesis_Data_Analysis\SRPE\McMaster MBB 2023 - Sessional RPE (Study Fall-2023) (Responses) - Form Responses 1.xlsx"

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
srpe['Name'] = srpe['Name'].replace({'Moody Mohamud': 'Moody Muhammed','Thomas Matsell': 'Thomas Mattsel'})

# Select the columns for each DataFrame
imu_data = imu_data[['Name', 'Date', 'Footnote', 'Period Start Time (24H)', 'Impact Load Total (L+R)', 'Impact Load Total per Minute (L and R)', 'Average Intensity (L and R)']]
force_plate = force_plate[['Name', 'Date', 'Tags', 'RSI-modified (Imp-Mom) [m/s]', 'Eccentric Duration [ms]', 'Jump Height (Imp-Mom) in Inches [in]', 'Eccentric Braking RFD / BM [N/s/kg]', 'Force at Zero Velocity / BM [N/kg]']] 
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

# Drop unnecessary columns
columns_to_remove = ['Footnote_Post-Practice', 'Period Start Time (24H)_Post-Practice', 'Impact Load Total (L+R)_Post-Practice', 'Impact Load Total per Minute (L and R)_Post-Practice', 'Average Intensity (L and R)_Post-Practice', 'SRPE_Post-Practice']
post_data.drop(columns=columns_to_remove, inplace=True)

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

# Calculate the pre-to-post differences for all metrics
metrics = ['RSI-modified (Imp-Mom) [m/s]', 'Eccentric Duration [ms]', 'Jump Height (Imp-Mom) in Inches [in]', 
           'Eccentric Braking RFD / BM [N/s/kg]', 'Force at Zero Velocity / BM [N/kg]', 
          ]

# Add a suffix to the columns for clarity
for metric in metrics:
    final_data_set[f'{metric}_Difference'] = final_data_set[f'{metric}_Post-Practice'] - final_data_set[f'{metric}']


# Calculate the Percentiles for Binning
impact_load_33rd = final_data_set['Impact Load Total (L+R)'].quantile(0.33)
impact_load_66th = final_data_set['Impact Load Total (L+R)'].quantile(0.66)

average_intensity_33rd = final_data_set['Average Intensity (L and R)'].quantile(0.33)
average_intensity_66th = final_data_set['Average Intensity (L and R)'].quantile(0.66)

# Create Bins for Each Metric
def create_bins(value, low, high):
    if value <= low:
        return 'Low'
    elif value <= high:
        return 'Medium'
    else:
        return 'High'

final_data_set['Impact Load Bin'] = final_data_set['Impact Load Total (L+R)'].apply(create_bins, args=(impact_load_33rd, impact_load_66th))
final_data_set['Average Intensity Bin'] = final_data_set['Average Intensity (L and R)'].apply(create_bins, args=(average_intensity_33rd, average_intensity_66th))

# Combine the Bins (Optional)
def combine_bins(row):
    if row['Impact Load Bin'] == 'High' or row['Average Intensity Bin'] == 'High':
        return 'High'
    elif row['Impact Load Bin'] == 'Medium' or row['Average Intensity Bin'] == 'Medium':
        return 'Medium'
    else:
        return 'Low'

final_data_set['Overall Practice Volume'] = final_data_set.apply(combine_bins, axis=1)

# Assuming final_data_set is your DataFrame with the practice volumes
impact_load_mean = final_data_set['Impact Load Total (L+R)'].mean()
impact_load_std = final_data_set['Impact Load Total (L+R)'].std()

average_intensity_mean = final_data_set['Average Intensity (L and R)'].mean()
average_intensity_std = final_data_set['Average Intensity (L and R)'].std()

# Generate data for the normal distribution curve
x_impact_load = np.linspace(impact_load_mean - 3*impact_load_std, impact_load_mean + 3*impact_load_std, 1000)
y_impact_load = norm.pdf(x_impact_load, impact_load_mean, impact_load_std)

x_average_intensity = np.linspace(average_intensity_mean - 3*average_intensity_std, average_intensity_mean + 3*average_intensity_std, 1000)
y_average_intensity = norm.pdf(x_average_intensity, average_intensity_mean, average_intensity_std)


# Plot for Impact Load
plt.figure(figsize=(10, 6))
sns.histplot(final_data_set['Impact Load Total (L+R)'], kde=False, bins=30, color='skyblue', stat='density')
plt.plot(x_impact_load, y_impact_load, color='red')
plt.axvline(x=impact_load_mean, color='black', linestyle='--')
plt.fill_between(x_impact_load, y_impact_load, where=(x_impact_load <= impact_load_33rd), color='green', alpha=0.3, label='Low')
plt.fill_between(x_impact_load, y_impact_load, where=(x_impact_load > impact_load_33rd) & (x_impact_load <= impact_load_66th), color='yellow', alpha=0.3, label='Medium')
plt.fill_between(x_impact_load, y_impact_load, where=(x_impact_load > impact_load_66th), color='red', alpha=0.3, label='High')
plt.title('Impact Load Distribution')
plt.xlabel('Impact Load Total (L+R)')
plt.ylabel('Density')
plt.legend()
plt.show()

# Plot for Average Intensity
plt.figure(figsize=(10, 6))
sns.histplot(final_data_set['Average Intensity (L and R)'], kde=False, bins=30, color='skyblue', stat='density')
plt.plot(x_average_intensity, y_average_intensity, color='red')
plt.axvline(x=average_intensity_mean, color='black', linestyle='--')
plt.fill_between(x_average_intensity, y_average_intensity, where=(x_average_intensity <= average_intensity_33rd), color='green', alpha=0.3, label='Low')
plt.fill_between(x_average_intensity, y_average_intensity, where=(x_average_intensity > average_intensity_33rd) & (x_average_intensity <= average_intensity_66th), color='yellow', alpha=0.3, label='Medium')
plt.fill_between(x_average_intensity, y_average_intensity, where=(x_average_intensity > average_intensity_66th), color='red', alpha=0.3, label='High')
plt.title('Average Intensity Distribution')
plt.xlabel('Average Intensity (L and R)')
plt.ylabel('Density')
plt.legend()
plt.show()

# Filter for numeric columns and include the 'Date' column
numeric_columns = final_data_set.select_dtypes(include=[np.number]).columns.tolist()
columns_to_include = ['Date'] + numeric_columns
grouped_data = final_data_set[columns_to_include].groupby('Date').mean().reset_index()

# Function to create the dual-axis plots
def dual_axis_plot(x, y1, y2_pre, y2_post, y1_label, y2_label_pre, y2_label_post, title):
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Bar plot for the primary y-axis
    sns.barplot(x=x, y=y1, ax=ax1, color='skyblue')
    ax1.set_xlabel('Date')
    ax1.set_ylabel(y1_label)
    ax1.set_title(title)
    ax1.tick_params(axis='x', rotation=45)

    # Create a secondary y-axis
    ax2 = ax1.twinx()
    
    # Line plot for the secondary y-axis
    sns.lineplot(x=x, y=y2_pre, ax=ax2, color='green', marker='o', label=y2_label_pre)
    sns.lineplot(x=x, y=y2_post, ax=ax2, color='red', marker='o', label=y2_label_post)
    ax2.set_ylabel('Metrics')
    ax2.legend(loc='upper left')

    plt.show()

# Data for Impact Load graph
impact_load = grouped_data['Impact Load Total (L+R)']
rsi_pre = grouped_data['RSI-modified (Imp-Mom) [m/s]']
rsi_post = grouped_data['RSI-modified (Imp-Mom) [m/s]_Post-Practice']
ecc_duration_pre = grouped_data['Eccentric Duration [ms]']
ecc_duration_post = grouped_data['Eccentric Duration [ms]_Post-Practice']
jump_height_pre = grouped_data['Jump Height (Imp-Mom) in Inches [in]']
jump_height_post = grouped_data['Jump Height (Imp-Mom) in Inches [in]_Post-Practice']
ecc_braking_pre = grouped_data['Eccentric Braking RFD / BM [N/s/kg]']
ecc_braking_post = grouped_data['Eccentric Braking RFD / BM [N/s/kg]_Post-Practice']
force_zero_pre = grouped_data['Force at Zero Velocity / BM [N/kg]']
force_zero_post = grouped_data['Force at Zero Velocity / BM [N/kg]_Post-Practice']

# Plot for Impact Load
dual_axis_plot(grouped_data['Date'], impact_load, rsi_pre, rsi_post, 
               'Impact Load Total (L+R)', 'RSI-modified Pre', 'RSI-modified Post', 'Impact Load and RSI-modified')

# Data for Average Intensity graph
average_intensity = grouped_data['Average Intensity (L and R)']

# Plot for Average Intensity
dual_axis_plot(grouped_data['Date'], average_intensity, rsi_pre, rsi_post, 
               'Average Intensity (L and R)', 'RSI-modified Pre', 'RSI-modified Post', 'Average Intensity and RSI-modified')

# Additional line plots for Eccentric Duration, Jump Height, Eccentric Braking, and Force at Zero Velocity
metrics_pre_post = [
    ('Eccentric Duration', ecc_duration_pre, ecc_duration_post),
    ('Jump Height', jump_height_pre, jump_height_post),
    ('Eccentric Braking', ecc_braking_pre, ecc_braking_post),
    ('Force at Zero Velocity', force_zero_pre, force_zero_post)
]

for metric, pre, post in metrics_pre_post:
    dual_axis_plot(grouped_data['Date'], impact_load, pre, post, 
                   'Impact Load Total (L+R)', f'{metric} Pre', f'{metric} Post', f'Impact Load and {metric}')
    
    dual_axis_plot(grouped_data['Date'], average_intensity, pre, post, 
                   'Average Intensity (L and R)', f'{metric} Pre', f'{metric} Post', f'Average Intensity and {metric}')
    
    
