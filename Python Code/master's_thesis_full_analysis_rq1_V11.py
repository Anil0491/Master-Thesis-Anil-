# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 18:28:42 2024

@author: Palan
"""

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from statsmodels.formula.api import mixedlm

# Define file paths for three different data sources
imu_data_raw = r"C:\Users\Palan\OneDrive\Documents\School\McMaster\1.Master's Thesis\Thesis_Data_Analysis\I.M.U data\StepSessionSummaryExport"
force_plate = r"C:\Users\Palan\OneDrive\Documents\School\McMaster\1.Master's Thesis\Thesis_Data_Analysis\raw_data\forcedecks-test-export-09_07_2024.xlsx"
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

# Select relevant columns
imu_data = imu_data[['Name', 'Date', 'Footnote', 'Impact Load Total (L+R)', 'Average Intensity (L and R)']]

force_plate = force_plate[['Name', 'ExternalId', 'Test Type', 'Date', 'Time', 'BW [KG]', 'Reps',
       'Tags', 'Additional Load [kg]', 'Jump Height (Imp-Mom) in Inches [in] ',
       'RSI-modified (Imp-Mom) [m/s] ', 'Eccentric Braking RFD / BM [N/s/kg] ',
       'Eccentric Braking Impulse [N s] ',
       'Force at Zero Velocity / BM [N/kg] ',
       'Concentric Impulse (Abs) / BM [N s] ']]
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
'RSI-modified (Imp-Mom) [m/s] ', 'Eccentric Braking RFD / BM [N/s/kg] ',
'Eccentric Braking Impulse [N s] ',
'Force at Zero Velocity / BM [N/kg] ',
'Concentric Impulse (Abs) / BM [N s] ']

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

# Function to calculate Cohen's d
def cohen_d(x, y):
    diff = x - y
    return np.mean(diff) / np.std(diff, ddof=1)

# Boxplots for each measurement
for measurement in value_vars:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=linear_mixed_model[linear_mixed_model['Measurement_Type'] == measurement], x='Time', y='Value')
    plt.title(f'Boxplot of {measurement}')
    plt.show()

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

# Enhanced paired plot to show statistically significant drop-off
for var in value_vars:
    # Subset data for the specific variable
    subset = linear_mixed_model_clean[linear_mixed_model_clean['Measurement_Type'] == var]
    
    # Handle duplicates by taking the mean
    subset_mean = subset.groupby(['PlayerID', 'Time'])['Value'].mean().reset_index()
    
    # Pivot the data to get Pre and Post values in separate columns
    pivot_data = subset_mean.pivot(index='PlayerID', columns='Time', values='Value').reset_index()
    
    # Calculate Cohen's d for the effect size
    cohen_d_value = cohen_d(pivot_data['Pre'], pivot_data['Post'])
    
    # Melt the pivoted data to long format for seaborn plotting
    paired_data = pivot_data.melt(id_vars='PlayerID', value_vars=['Pre', 'Post'], var_name='Time', value_name='Value')
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=paired_data, x='Time', y='Value', hue='PlayerID', marker='o', legend=False, alpha=0.5, palette='viridis')
    sns.pointplot(data=paired_data, x='Time', y='Value', ci='sd', markers='D', color='black')
    plt.title(f'Paired Changes in {var} from Pre to Post (Cohen\'s d = {cohen_d_value:.2f})')
    plt.xlabel('Time')
    plt.ylabel(var)
    plt.show()

# Descriptive statistics for the cleaned data (means and standard deviations only)
descriptive_stats = linear_mixed_model_clean.groupby(['Position', 'Measurement_Type', 'Time'])['Value'].agg(['mean', 'std']).unstack()

# Reorder columns to have 'Pre' before 'Post'
ordered_columns = [('mean', 'Pre'), ('mean', 'Post'), ('std', 'Pre'), ('std', 'Post')]
descriptive_stats = descriptive_stats[ordered_columns]

# Function to plot the descriptive statistics as a table
def plot_descriptive_stats_table(stats, title):
    fig, ax = plt.subplots(figsize=(12, 8))  # set size frame
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=np.round(stats.values, 2), colLabels=stats.columns, rowLabels=stats.index, loc='center')
    plt.title(title)
    plt.show()

# Plot the descriptive statistics table separately for forwards and guards
for position in ['Forward', 'Guard']:
    stats = descriptive_stats.loc[position]
    plot_descriptive_stats_table(stats, f'Descriptive Statistics for {position}s (Means and Standard Deviations)')

# Calculate Cohen's d for each position and measurement type
effect_sizes = []
for position in ['Forward', 'Guard']:
    for var in value_vars:
        subset = linear_mixed_model_clean[(linear_mixed_model_clean['Position'] == position) & (linear_mixed_model_clean['Measurement_Type'] == var)]
        subset_mean = subset.groupby(['PlayerID', 'Time'])['Value'].mean().reset_index()
        pivot_data = subset_mean.pivot(index='PlayerID', columns='Time', values='Value').reset_index()
        cohen_d_value = cohen_d(pivot_data['Pre'], pivot_data['Post'])
        effect_sizes.append((position, var, cohen_d_value))

effect_sizes_df = pd.DataFrame(effect_sizes, columns=['Position', 'Measurement_Type', 'Cohen\'s d']).set_index(['Position', 'Measurement_Type'])

# Function to plot the effect sizes as a table
def plot_effect_sizes_table(effect_sizes, title):
    fig, ax = plt.subplots(figsize=(12, 8))  # set size frame
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=np.round(effect_sizes.values, 2), colLabels=effect_sizes.columns, rowLabels=effect_sizes.index, loc='center')
    plt.title(title)
    plt.show()

# Plot the effect sizes table separately for forwards and guards
for position in ['Forward', 'Guard']:
    effect_sizes_pos = effect_sizes_df.loc[position]
    plot_effect_sizes_table(effect_sizes_pos, f'Cohen\'s d for {position}s')

# Results of linear mixed models for overall and by position
for var in value_vars:
    data_subset = linear_mixed_model_clean[linear_mixed_model_clean['Measurement_Type'] == var]
    model = mixedlm("Value ~ Time", data_subset, groups=data_subset["PlayerID"])
    result = model.fit()
    print(f"Results for {var}:")
    print(result.summary())
    time_coefficient_name = result.params.index[result.params.index.str.contains("Time")][0]
    if result.pvalues[time_coefficient_name] < 0.05:
        print(f"    -> The change in {var} from Pre to Post is significant.")
    else:
        print(f"    -> The change in {var} from Pre to Post is not significant.")
    print("\n")

for position in ['Forward', 'Guard']:
    print(f"Processing for {position}s:\n")
    
    # Subset the data by position
    position_data = linear_mixed_model_clean[linear_mixed_model_clean['Position'] == position]
    
    # Fit the linear mixed model on the cleaned data for each variable
    for var in value_vars:
        data_subset = position_data[position_data['Measurement_Type'] == var]
        
        # Fit the linear mixed model for pre-post difference
        model = mixedlm("Value ~ Time", data_subset, groups=data_subset["PlayerID"], re_formula="~Time")
        result = model.fit()
        print(f"Results for {var} ({position}s):")
        print(result.summary())
        time_coefficient_name = result.params.index[result.params.index.str.contains("Time")][0]
        if result.pvalues[time_coefficient_name] < 0.05:
            print(f"    -> The change in {var} from Pre to Post for {position}s is significant.")
        else:
            print(f"    -> The change in {var} from Pre to Post for {position}s is not significant.")
        print("\n")
        
        # Plot the results
        subset_mean = data_subset.groupby(['PlayerID', 'Time'])['Value'].mean().reset_index()
        pivot_data = subset_mean.pivot(index='PlayerID', columns='Time', values='Value').reset_index()
        paired_data = pivot_data.melt(id_vars='PlayerID', value_vars=['Pre', 'Post'], var_name='Time', value_name='Value')
        cohen_d_value = cohen_d(pivot_data['Pre'], pivot_data['Post'])

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=paired_data, x='Time', y='Value', hue='PlayerID', marker='o', legend=False, alpha=0.5, palette='viridis')
        sns.pointplot(data=paired_data, x='Time', y='Value', ci='sd', markers='D', color='black')
        plt.title(f'Paired Changes in {var} from Pre to Post ({position}s, Cohen\'s d = {cohen_d_value:.2f})')
        plt.xlabel('Time')
        plt.ylabel(var)
        plt.show()

# Print confirmation
print("All figures have been displayed.")