# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 16:54:47 2024

@author: Palan
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 08:19:36 2024

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

# Calculate the change in each metric from pre to post practice
for metric in metric_columns:
    pre_post_data[f'{metric}_Change'] = pre_post_data[f'{metric}_Post-Practice'] - pre_post_data[f'{metric}']

# Function to remove outliers using the IQR method
def remove_outliers(df, metric_columns):
    Q1 = df[metric_columns].quantile(0.25)
    Q3 = df[metric_columns].quantile(0.75)
    IQR = Q3 - Q1
    # Define outlier condition
    outlier_condition = ~((df[metric_columns] < (Q1 - 1.5 * IQR)) | (df[metric_columns] > (Q3 + 1.5 * IQR))).any(axis=1)
    return df[outlier_condition]

# Remove outliers from pre_post_data
pre_post_data = remove_outliers(pre_post_data, metric_columns)
# Calculate the average change for each date
avg_change_data = pre_post_data.groupby('Date').agg({f'{metric}_Change': 'mean' for metric in metric_columns}).reset_index()

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

def fit_mixed_model(df, y_var, fixed_effects):
    model = mixedlm(f"{y_var} ~ {' + '.join(fixed_effects)}", df, groups=df['Date'])
    result = model.fit()
    return result, result.summary()

def create_predictions(df, fixed_effect, fixed_effect_name, model_result):
    intercept = model_result.params['Intercept']
    slope = model_result.params[fixed_effect]
    df['Predicted'] = intercept + slope * df[fixed_effect_name]
    return df

def plot_metric_dropoffs(metric, predictions_load, predictions_intensity):
    # Scatter plot with regression line for Impact Load vs Metric Drop-Off
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='Avg_Impact_Load', y='Change', data=predictions_load, hue='Date', palette='viridis', s=100, alpha=0.6)
    sns.lineplot(x='Avg_Impact_Load', y='Predicted', data=predictions_load, color='red', linewidth=2)

    # Fit linear regression model to display equation and R-squared value
    X = predictions_load['Avg_Impact_Load'].values.reshape(-1, 1)
    y = predictions_load['Change'].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    r_squared = model.score(X, y)

    # Plot confidence interval band
    ci = 1.96 * np.std(y - y_pred) / np.mean(y)
    plt.fill_between(predictions_load['Avg_Impact_Load'], y_pred - ci, y_pred + ci, color='red', alpha=0.1)

    # Annotate with regression equation and R-squared
    plt.annotate(f'y = {model.coef_[0]:.6f}x + {model.intercept_:.6f}\nR² = {r_squared:.3f}', 
                 xy=(0.05, 0.90), xycoords='axes fraction', fontsize=12, color='red',
                 bbox=dict(facecolor='white', alpha=0.5))


   # Adjust x and y axis limits to the data range for the current metric
    plt.xlim(predictions_load['Avg_Impact_Load'].min(), predictions_load['Avg_Impact_Load'].max())
    plt.ylim(predictions_load['Change'].min() - 0.1 * abs(predictions_load['Change'].min()), 
             predictions_load['Change'].max() + 0.1 * abs(predictions_load['Change'].max()))



    plt.title(f'Impact Load vs Change in {metric.strip()}')
    plt.xlabel('Average Impact Load')
    plt.ylabel(f'Drop-Off in {metric.strip()}')
    plt.legend().remove()
    plt.show()

    # Scatter plot with regression line for Average Intensity vs Metric Drop-Off
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='Avg_Average_Intensity', y='Change', data=predictions_intensity, hue='Date', palette='viridis', s=100, alpha=0.6)
    sns.lineplot(x='Avg_Average_Intensity', y='Predicted', data=predictions_intensity, color='red', linewidth=2)

    # Fit linear regression model to display equation and R-squared value
    X = predictions_intensity['Avg_Average_Intensity'].values.reshape(-1, 1)
    y = predictions_intensity['Change'].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    r_squared = model.score(X, y)

    # Plot confidence interval band
    ci = 1.96 * np.std(y - y_pred) / np.mean(y)
    plt.fill_between(predictions_intensity['Avg_Average_Intensity'], y_pred - ci, y_pred + ci, color='red', alpha=0.1)

    # Annotate with regression equation and R-squared
    plt.annotate(f'y = {model.coef_[0]:.6f}x + {model.intercept_:.6f}\nR² = {r_squared:.3f}', 
                 xy=(0.05, 0.90), xycoords='axes fraction', fontsize=12, color='red',
                 bbox=dict(facecolor='white', alpha=0.5))
    
# Adjust x and y axis limits to the data range for the current metric
    plt.xlim(predictions_intensity['Avg_Average_Intensity'].min(), predictions_intensity['Avg_Average_Intensity'].max())
    plt.ylim(predictions_intensity['Change'].min() - 0.1 * abs(predictions_intensity['Change'].min()), 
             predictions_intensity['Change'].max() + 0.1 * abs(predictions_intensity['Change'].max()))

    plt.title(f'Average Intensity vs Change in {metric.strip()}')
    plt.xlabel('Average Intensity')
    plt.ylabel(f'Drop-Off in {metric.strip()}')
    plt.legend().remove()
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
    plot_metric_dropoffs(metric, predictions_load, predictions_intensity)


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