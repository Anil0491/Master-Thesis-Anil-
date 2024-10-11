# -*- coding: utf-8 -*-
"""
Created on Tue May 21 08:18:24 2024

@author: Palan
"""
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro
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
srpe['Name'] = srpe['Name'].replace({'Moody Mohamud': 'Moody Muhammed', 'Thomas Matsell': 'Thomas Mattsel'})

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

# Merge the pre and post data
pre_post_data = pre_data.merge(post_data, on=['Name', 'Date'], suffixes=('_Pre', ''), how='inner')

# Define the mapping of names to positions
positions = {'Brendan Amoyaw': 'Forward', 'Cashius McNeilly': 'Guard', 'Daniel Graham': 'Guard', 'Elijah Bethune': 'Guard', 'Jeremiah Francis': 'Guard', 'Kazim Raza': 'Guard', 'Matthew Groe': 'Guard', 'Mike Demagus': 'Guard', 'Moody Muhammed': 'Forward', 'Parker Davis': 'Guard', 'Riaz Saliu': 'Forward', 'Sebastian Di Manno': 'Guard', 'Stevan Japundzic': 'Forward', 'Thomas Mattsel': 'Forward'}

# Add 'Position' and 'PlayerID' columns
pre_post_data['Position'] = pre_post_data['Name'].map(positions)
unique_names = pre_post_data['Name'].unique()
player_ids = {name: idx for idx, name in enumerate(unique_names, 1)}
pre_post_data['PlayerID'] = pre_post_data['Name'].map(player_ids)

# Create a new DataFrame without the 'Name' column
final_data_set = pre_post_data.drop(columns=['Name'])

# Columns to keep as identifiers
id_vars = ['PlayerID', 'Date', 'Position']

# Variables measured both pre and post
value_vars = ['Impact Load Total (L+R)', 'Average Intensity (L and R)', 'RSI-modified (Imp-Mom) [m/s]', 'Eccentric Duration [ms]', 'Jump Height (Imp-Mom) in Inches [in]', 'Eccentric Braking RFD / BM [N/s/kg]', 'Force at Zero Velocity / BM [N/kg]']

# Melting the dataframe
linear_mixed_model = pd.melt(final_data_set, id_vars=id_vars, value_vars=value_vars + [f'{var}_Post-Practice' for var in value_vars], var_name='Measurement_Type', value_name='Value')

# Create a new 'Time' variable to indicate pre- or post-practice
linear_mixed_model['Time'] = linear_mixed_model['Measurement_Type'].apply(lambda x: 'Post' if 'Post-Practice' in x else 'Pre')

# Adjust 'Measurement_Type' to have consistent naming for pre and post
linear_mixed_model['Measurement_Type'] = linear_mixed_model['Measurement_Type'].str.replace('_Pre', '').str.replace('_Post-Practice', '')

# Ensure 'Position' is included
linear_mixed_model = linear_mixed_model.merge(pre_post_data[['PlayerID', 'Position']], on='PlayerID', how='left')

# Remove duplicates
linear_mixed_model = linear_mixed_model.drop_duplicates()

# Visual inspection using boxplots
for measurement in value_vars:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=linear_mixed_model[linear_mixed_model['Measurement_Type'] == measurement], x='Time', y='Value')
    plt.title(f'Boxplot of {measurement}')
    plt.show()
    
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
 
# Normality checks on the cleaned data with individual graphs for each metric
for var in value_vars:
    plt.figure(figsize=(10, 6))
    
    # Determine x-axis limit based on data distribution
    xlim_pre = linear_mixed_model_clean[(linear_mixed_model_clean['Measurement_Type'] == var) & (linear_mixed_model_clean['Time'] == 'Pre')]['Value'].quantile(0.99)
    xlim_post = linear_mixed_model_clean[(linear_mixed_model_clean['Measurement_Type'] == var) & (linear_mixed_model_clean['Time'] == 'Post')]['Value'].quantile(0.99)
    xlim = max(xlim_pre, xlim_post)
    
    # Plot pre-practice data
    sns.histplot(data=linear_mixed_model_clean[(linear_mixed_model_clean['Measurement_Type'] == var) & (linear_mixed_model_clean['Time'] == 'Pre')], x='Value', color='blue', label='Pre-Practice', kde=True)
    # Plot post-practice data
    sns.histplot(data=linear_mixed_model_clean[(linear_mixed_model_clean['Measurement_Type'] == var) & (linear_mixed_model_clean['Time'] == 'Post')], x='Value', color='red', label='Post-Practice', kde=True)
    
    # Set the title and labels
    plt.title(f'Distribution of {var} Pre and Post Practice')
    plt.xlabel(f'{var} [units]')
    plt.ylabel('Frequency')
    plt.xlim(0, xlim)
    
    # Add the legend
    plt.legend()
    
    # Show the plot
    plt.show()

# Initialize an empty dictionary to store the Shapiro-Wilk results
normality_results = {}

# Loop through each measurement type and perform the Shapiro-Wilk test
for var in value_vars:
    # Extract pre and post data for the variable
    pre_data = linear_mixed_model_clean[(linear_mixed_model_clean['Measurement_Type'] == var) & (linear_mixed_model_clean['Time'] == 'Pre')]['Value']
    post_data = linear_mixed_model_clean[(linear_mixed_model_clean['Measurement_Type'] == var) & (linear_mixed_model_clean['Time'] == 'Post')]['Value']
    
    # Drop missing values as Shapiro-Wilk cannot handle them
    clean_pre = pre_data.dropna()
    clean_post = post_data.dropna()
    
    # Perform Shapiro-Wilk test for normality
    stat_pre, p_value_pre = shapiro(clean_pre)
    stat_post, p_value_post = shapiro(clean_post)
    
    # Store the results
    normality_results[var + ' Pre-Practice'] = (stat_pre, p_value_pre)
    normality_results[var + ' Post-Practice'] = (stat_post, p_value_post)

# Print the results of normality tests
for test_name, result in normality_results.items():
    stat, p_value = result
    print(f"{test_name}: Shapiro-Wilk Statistic={stat:.3f}, p-value={p_value:.3f}")
    if p_value > 0.05:
        print("    -> The data appears to be normally distributed.")
    else:
        print("    -> The data does not appear to be normally distributed.")   

# Linear Mixed Model Analysis
lmm_results = {}

for metric in value_vars:
    # Calculate the change in metric from pre to post
    linear_mixed_model_clean[f'{metric}_Change'] = linear_mixed_model_clean.apply(
        lambda row: row['Value'] if row['Time'] == 'Post' else -row['Value'], axis=1
    )
    
    # Aggregate to calculate the actual change per player per date
    metric_change = linear_mixed_model_clean.groupby(['PlayerID', 'Date', 'Measurement_Type'])['Value'].sum().reset_index()
    
    # Filter the data to include only the changes for the current metric
    metric_change = metric_change[metric_change['Measurement_Type'] == metric]
    
    # Merge the changes with the original data to include impact load and average intensity
    merged_data = metric_change.merge(pre_post_data[['PlayerID', 'Date', 'Impact_Load_Total_L_R', 'Average_Intensity_L_R']], on=['PlayerID', 'Date'])
    
    # Fit the linear mixed model
    model = mixedlm(f"Value ~ Impact_Load_Total_L_R + Average_Intensity_L_R", merged_data, groups=merged_data['PlayerID'])
    result = model.fit()
    lmm_results[metric] = result.summary()
    
    print(f"Linear Mixed Model Results for {metric} Change")
    print(result.summary())
    print("\n")

# Display the LMM results
for metric, summary in lmm_results.items():
    print(f"\nLMM Results for {metric} Change:\n")
    print(summary)

# Visualize the effect of impact load and average intensity on metric changes
for metric in value_vars:
    # Merge data for visualization
    merged_data = linear_mixed_model_clean[linear_mixed_model_clean['Measurement_Type'] == metric].copy()
    merged_data['Value_Change'] = merged_data.apply(lambda row: row['Value'] if row['Time'] == 'Post' else -row['Value'], axis=1)
    merged_data = merged_data.groupby(['PlayerID', 'Date'])['Value_Change'].sum().reset_index()
    merged_data = merged_data.merge(pre_post_data[['PlayerID', 'Date', 'Impact_Load_Total_L_R', 'Average_Intensity_L_R']], on=['PlayerID', 'Date'])

    plt.figure(figsize=(14, 7))
    sns.regplot(x='Impact_Load_Total_L_R', y='Value_Change', data=merged_data, label='Impact Load Total (L+R)', color='blue')
    sns.regplot(x='Average_Intensity_L_R', y='Value_Change', data=merged_data, label='Average Intensity (L and R)', color='green')
    plt.title(f'Effect of Impact Load and Average Intensity on {metric} Changes')
    plt.xlabel('Load/Intensity')
    plt.ylabel(f'{metric} Change')
    plt.legend()
    plt.show()
 