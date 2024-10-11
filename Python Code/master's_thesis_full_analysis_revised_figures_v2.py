# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 06:52:54 2024

@author: Palan
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import mixedlm

# Define file paths
base_path = r"C:\Users\Palan\OneDrive\Documents\School\McMaster\1.Master's Thesis\Thesis_Data_Analysis"
imu_data_raw = os.path.join(base_path, "I.M.U data", "StepSessionSummaryExport")
force_plate_path = os.path.join(base_path, "raw_data", "forcedecks-test-export-12_07_2024.xlsx")
srpe_path = os.path.join(base_path, "SRPE", "McMaster MBB 2023 - Sessional RPE (Study Fall-2023) (Responses) - Form Responses 1.xlsx")

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
force_plate = force_plate[['Name', 'Date', 'Tags', 'Jump Height (Imp-Mom) in Inches [in] ',
                           'RSI-modified (Imp-Mom) [m/s] ', 'Eccentric Braking RFD / BM [N/s/kg] ',
                           'Eccentric Braking Impulse [N s] ', 'Force at Zero Velocity / BM [N/kg] ',
                           'Concentric Impulse (Abs) / BM [N s] ', 'Concentric Impulse [N s] ',
                           'Concentric RFD / BM [N/s/kg] ']]
srpe = srpe[['Name', 'Date', 'SRPE']]

# Standardize date formats
imu_data['Date'] = pd.to_datetime(imu_data['Date']).dt.strftime('%Y-%m-%d')
force_plate['Date'] = pd.to_datetime(force_plate['Date'], dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%d')
srpe['Date'] = pd.to_datetime(srpe['Date'], errors='coerce').dt.strftime('%Y-%m-%d')

# Merge datasets
complete_data = imu_data.merge(force_plate, on=['Name', 'Date'], how='outer').merge(srpe, on=['Name', 'Date'], how='outer')
all_footnote_data = complete_data[complete_data['Footnote'] == 'All']

# Filter and rename columns
pre_data = all_footnote_data[all_footnote_data['Tags'] == 'Pre-Practice']
post_data = all_footnote_data[all_footnote_data['Tags'] == 'Post-Practice']
metric_columns = [col for col in post_data.columns if col not in ['Name', 'Date', 'Tags', 'Footnote']]
post_data = post_data.rename(columns={metric: f'{metric}_Post-Practice' for metric in metric_columns})
pre_post_data = pre_data.merge(post_data, on=['Name', 'Date'], suffixes=('_Pre', '_Post'), how='inner')

# Map names to positions
positions = {'Brendan Amoyaw': 'Forward', 'Cashius McNeilly': 'Guard', 'Daniel Graham': 'Guard', 'Elijah Bethune': 'Guard', 'Jeremiah Francis': 'Guard',
             'Kazim Raza': 'Guard', 'Matthew Groe': 'Guard', 'Mike Demagus': 'Guard', 'Moody Muhammed': 'Forward', 'Parker Davis': 'Guard',
             'Riaz Saliu': 'Forward', 'Sebastian Di Manno': 'Guard', 'Stevan Japundzic': 'Forward', 'Thomas Mattsel': 'Forward'}

pre_post_data['Position'] = pre_post_data['Name'].map(positions)
unique_names = pre_post_data['Name'].unique()
player_ids = {name: idx for idx, name in enumerate(unique_names, 1)}
pre_post_data['PlayerID'] = pre_post_data['Name'].map(player_ids)

# Function to remove outliers using the IQR method for a specific column 
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    # Define outlier condition based on the specific column
    outlier_condition = (df[column] >= (Q1 - 1.5 * IQR)) & (df[column] <= (Q3 + 1.5 * IQR))
    return df[outlier_condition]

# Apply the function to remove outliers based on 'Jump Height (Imp-Mom) in Inches [in] and calculate descrriptive stats'
pre_post_data = remove_outliers(pre_post_data, 'Jump Height (Imp-Mom) in Inches [in] ')
descriptive_stats_by_position = pre_post_data.groupby('Position').describe()
pre_post_data.to_excel('pre_post_data.xlsx', index=False)

# Calculate the change in each metric from pre to post practice
for metric in metric_columns:
    pre_post_data[f'{metric}_Change'] = pre_post_data[f'{metric}_Post-Practice'] - pre_post_data[f'{metric}']

# Calculate the SEM for each metric on each date
sem_change_data = pre_post_data.groupby('Date').agg({f'{metric}_Change': lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) for metric in metric_columns}).reset_index()

# Rename columns to indicate SEM
sem_change_data.rename(columns={f'{metric}_Change': f'{metric}_SEM' for metric in metric_columns}, inplace=True)

# Calculate the average change for each date
avg_change_data = pre_post_data.groupby('Date').agg({f'{metric}_Change': 'mean' for metric in metric_columns}).reset_index()

# Merge SEM data with avg_change_data
avg_change_data = avg_change_data.merge(sem_change_data, on='Date', how='left')

# Merge with the average practice volume (Impact Load and Intensity)
practice_means = pre_post_data.groupby('Date').agg({'Impact Load Total (L+R)': 'mean', 'Average Intensity (L and R)': 'mean'}).reset_index()
avg_change_data = avg_change_data.merge(practice_means, on='Date', how='left')
avg_change_data.rename(columns={'Impact Load Total (L+R)': 'Avg_Impact_Load', 'Average Intensity (L and R)': 'Avg_Average_Intensity'}, inplace=True)

# Reshape data to long format for drop-off data
long_format_change_data = pd.melt(avg_change_data, 
                                   id_vars=['Date', 'Avg_Impact_Load', 'Avg_Average_Intensity'],
                                   value_vars=[f'{metric}_Change' for metric in metric_columns],
                                   var_name='Metric', value_name='Change')

# Extract metric names without '_Change' suffix
long_format_change_data['Metric'] = long_format_change_data['Metric'].str.replace('_Change', '')

# Save DataFrame to an Excel file
output_file_path = 'metrics_data.xlsx'
long_format_change_data.to_excel(output_file_path, index=False)

def fit_mixed_model(df, y_var, fixed_effects):
    model = mixedlm(f"{y_var} ~ {' + '.join(fixed_effects)}", df, groups=df['Date'])
    result = model.fit()
    return result, result.summary()

def create_predictions(df, fixed_effect, fixed_effect_name, model_result):
    intercept = model_result.params['Intercept']
    slope = model_result.params[fixed_effect]
    df['Predicted'] = intercept + slope * df[fixed_effect_name]
    return df
    
def plot_metric_dropoffs(metric, predictions_load, predictions_intensity, avg_change_data):
    # Scatter plot with regression line for Impact Load vs Metric Drop-Off
    plt.figure(figsize=(14, 9))

    # Assign a different color to each data point
    colors = sns.color_palette("husl", len(predictions_load))

    # Define the SEM column name
    sem_col = f'{metric}_SEM'

    # Apply jittering to the x-values to reduce overlap
    jittered_x = predictions_load['Avg_Impact_Load'] + np.random.normal(0, 0.5, size=len(predictions_load))

    for i, (x, y, yerr) in enumerate(zip(jittered_x, predictions_load['Change'], avg_change_data[sem_col])):
        plt.errorbar(x, y, yerr=yerr, fmt='o', color=colors[i], ecolor='lightgray', elinewidth=2, capsize=4, alpha=0.7)

    sns.lineplot(x='Avg_Impact_Load', y='Predicted', data=predictions_load, color='red', linewidth=2)

    # Fit linear regression model to display equation and R-squared value
    X = predictions_load['Avg_Impact_Load'].values.reshape(-1, 1)
    y = predictions_load['Change'].values
    model = LinearRegression().fit(X, y)
    r_squared = model.score(X, y)

    # Annotate with regression equation and R-squared
    plt.annotate(f'y = {model.coef_[0]:.6f}x + {model.intercept_:.6f}\nR² = {r_squared:.3f}', 
                 xy=(0.01, 0.90), xycoords='axes fraction', fontsize=12, color='black',horizontalalignment='left', 
                 bbox=dict(facecolor='white', alpha=0.5))

    # Set larger buffer for x-axis limits to ensure all points are visible
    plt.xlim(predictions_load['Avg_Impact_Load'].min() - 10000, predictions_load['Avg_Impact_Load'].max() + 10000)
    plt.ylim(predictions_load['Change'].min() - 0.1 * abs(predictions_load['Change'].min()), 
             predictions_load['Change'].max() + 0.1 * abs(predictions_load['Change'].max()))

    plt.title(f'Impact Load vs Change in {metric.strip()}')
    plt.xlabel('Average Impact Load')
    plt.ylabel(f'Drop-Off in {metric.strip()}')
    plt.show()

    # Scatter plot with regression line for Average Intensity vs Metric Drop-Off
    plt.figure(figsize=(14,9))

    # Apply jittering to the x-values to reduce overlap
    jittered_x_intensity = predictions_intensity['Avg_Average_Intensity'] + np.random.normal(0, 0.5, size=len(predictions_intensity))

    for i, (x, y, yerr) in enumerate(zip(jittered_x_intensity, predictions_intensity['Change'], avg_change_data[sem_col])):
        plt.errorbar(x, y, yerr=yerr, fmt='o', color=colors[i], ecolor='lightgray', elinewidth=2, capsize=4, alpha=0.7)

    sns.lineplot(x='Avg_Average_Intensity', y='Predicted', data=predictions_intensity, color='red', linewidth=2)

    # Fit linear regression model to display equation and R-squared value
    X = predictions_intensity['Avg_Average_Intensity'].values.reshape(-1, 1)
    y = predictions_intensity['Change'].values
    model = LinearRegression().fit(X, y)
    r_squared = model.score(X, y)

    # Annotate with regression equation and R-squared
    plt.annotate(f'y = {model.coef_[0]:.6f}x + {model.intercept_:.6f}\nR² = {r_squared:.3f}', 
                 xy=(0.01, 0.90), xycoords='axes fraction', fontsize=12, color='black', horizontalalignment='left',
                 bbox=dict(facecolor='white', alpha=0.5))

    # Set larger buffer for x-axis limits to ensure all points are visible
    plt.xlim(predictions_intensity['Avg_Average_Intensity'].min() - 1, predictions_intensity['Avg_Average_Intensity'].max() + 1)
    plt.ylim(predictions_intensity['Change'].min() - 0.1 * abs(predictions_intensity['Change'].min()), 
             predictions_intensity['Change'].max() + 0.1 * abs(predictions_intensity['Change'].max()))

    plt.title(f'Average Intensity vs Change in {metric.strip()}')
    plt.xlabel('Average Intensity')
    plt.ylabel(f'Drop-Off in {metric.strip()}')
    plt.show()

# Initialize dictionaries to store results for each metric
impact_load_results = {}
average_intensity_results = {}

# Perform the analysis for each metric
for metric in metric_columns:
    y_var = f'{metric}_Change'
    
    # Analysis with Avg_Impact_Load
    result_load, summary_load = fit_mixed_model(long_format_change_data[long_format_change_data['Metric'] == metric], 'Change', ['Avg_Impact_Load'])
    impact_load_results[metric] = result_load
    
    # Analysis with Avg_Average_Intensity
    result_intensity, summary_intensity = fit_mixed_model(long_format_change_data[long_format_change_data['Metric'] == metric], 'Change', ['Avg_Average_Intensity'])
    average_intensity_results[metric] = result_intensity

# Iterate through each metric and generate the plots
for metric in metric_columns:
    # Create predictions for Impact Load
    predictions_load = create_predictions(
        long_format_change_data[long_format_change_data['Metric'] == metric],
        'Avg_Impact_Load', 'Avg_Impact_Load', impact_load_results[metric]
    )
    
    # Create predictions for Average Intensity
    predictions_intensity = create_predictions(
        long_format_change_data[long_format_change_data['Metric'] == metric],
        'Avg_Average_Intensity', 'Avg_Average_Intensity', average_intensity_results[metric]
    )
    
    # Plot the metric drop-offs
    plot_metric_dropoffs(metric, predictions_load, predictions_intensity, avg_change_data)

# Function to sanitize file names
def sanitize_filename(filename):
    return "".join([c for c in filename if c.isalnum() or c in (' ', '.', '_')]).rstrip()

# Function to save model summaries to text files
def save_model_summaries_to_file(results, analysis_type, output_path):
    file_path = os.path.join(output_path, f"{sanitize_filename(analysis_type)}_Mixed_Model_Summary.txt")
    with open(file_path, "w") as f:
        for metric, result in results.items():
            f.write(f"Metric: {metric}\n")
            f.write(result.summary().as_text())
            f.write("\n\n")

# Specify the output path where the text files will be saved
output_path = r"C:\Users\Palan\OneDrive\Documents\School\McMaster\1.Master's Thesis\Thesis_Data_Analysis"

# Save summaries for both Impact Load and Average Intensity analyses
save_model_summaries_to_file(impact_load_results, "Impact_Load", output_path)
save_model_summaries_to_file(average_intensity_results, "Average_Intensity", output_path)

print("Mixed-effects model summaries have been saved to text files.")

# Function to extract model coefficients and standard errors into a table
def model_to_table(result, metric_name):
    params = result.params
    standard_errors = result.bse
    table_data = {
        'Metric': [metric_name] * len(params),
        'Coefficient': params.index,
        'Estimate': params.values,
        'Std. Error': standard_errors.values,
        'z-value': result.tvalues.values,
        'p-value': result.pvalues.values
    }
    return pd.DataFrame(table_data)

# Combine all results into one dataframe for both impact load and average intensity
impact_load_tables = [model_to_table(result, metric) for metric, result in impact_load_results.items()]
average_intensity_tables = [model_to_table(result, metric) for metric, result in average_intensity_results.items()]

# Concatenate tables for easy display
impact_load_df = pd.concat(impact_load_tables, ignore_index=True)
average_intensity_df = pd.concat(average_intensity_tables, ignore_index=True)

# Filter out the metrics that need to be removed
metrics_to_remove = ['Impact Load Total (L+R)', 'Average Intensity (L and R)', 'SRPE']

# Filter the DataFrames
filtered_impact_load_df = impact_load_df[~impact_load_df['Metric'].isin(metrics_to_remove)]
filtered_average_intensity_df = average_intensity_df[~average_intensity_df['Metric'].isin(metrics_to_remove)]

with pd.ExcelWriter('filtered_metrics.xlsx') as writer: filtered_impact_load_df.to_excel(writer, sheet_name='Impact Load'); filtered_average_intensity_df.to_excel(writer, sheet_name='Average Intensity')

# Round all numeric values to 3 decimal places
#filtered_impact_load_df = filtered_impact_load_df.round(3)
#filtered_average_intensity_df = filtered_average_intensity_df.round(3)

