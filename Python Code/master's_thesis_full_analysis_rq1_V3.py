# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:44:04 2024

@author: Palan
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import shapiro, wilcoxon,ttest_rel
import matplotlib.pyplot as plt


# Define file paths for three different data sources: IMU data (in CSV format), Force Plate data (in Excel format), and SRPE data (in Excel format).
imu_data_raw = r"C:\Users\Palan\OneDrive\Documents\School\McMaster\1.Master's Thesis\Thesis_Data_Analysis\I.M.U data\StepSessionSummaryExport"
force_plate = r"C:\Users\Palan\OneDrive\Documents\School\McMaster\1.Master's Thesis\Thesis_Data_Analysis\Forceplate Data\forcedecks-test-export-full_v1.xlsx"
srpe = r"C:\Users\Palan\OneDrive\Documents\School\McMaster\1.Master's Thesis\Thesis_Data_Analysis\SRPE\McMaster MBB 2023 - Sessional RPE (Study Fall-2023) (Responses) - Form Responses 1.xlsx"

# Initialize an empty DataFrame to store IMU data
imu_data = pd.DataFrame()
# These lines start a nested loop to traverse subfolders and date-specific subfolders within the directory containing IMU data.
for subfolder in os.listdir(imu_data_raw):
    subfolder_path = os.path.join(imu_data_raw, subfolder)
    if os.path.isdir(subfolder_path):
        # Loop over date-specific subfolders
        for date_folder in os.listdir(subfolder_path):
            date_folder_path = os.path.join(subfolder_path, date_folder)
            if os.path.isdir(date_folder_path):
                # Within the nested loops, it walks through files within the date-specific subfolders, specifically looking for CSV files. For each CSV file, it reads the data into a temporary DataFrame (temp_imu_table) and concatenates this data into the imu_data DataFrame.
                for root, dirs, files in os.walk(date_folder_path):
                    for file in files:
                        if file.endswith(".csv"):
                            imu_sheet_path = os.path.join(root, file)
                            temp_imu_table = pd.read_csv(imu_sheet_path)
                            imu_data = pd.concat([imu_data, temp_imu_table], ignore_index=True)
                            
                            # Read the Excel files for Force Plate and SRPE Data
force_plate = pd.read_excel(force_plate)
srpe = pd.read_excel(srpe)

#These lines combine the "First Name" and "Last Name" columns in the imu_data DataFrame into a new "Name" column. It also renames the "Date (YYYY-MM-DD)" column to simply "Date."
imu_data['Name'] = imu_data['First Name'] + ' ' + imu_data['Last Name']
imu_data.rename(columns={'Date (YYYY-MM-DD)': 'Date'}, inplace=True)

# Replace specific names in the 'Name' column
imu_data['Name'] = imu_data['Name'].replace({'Kazim Raza1': 'Kazim Raza', 'Brendan Amoyaw1': 'Brendan Amoyaw'})
srpe['Name'] = srpe['Name'].replace({'Moody Mohamud': 'Moody Muhammed','Thomas Matsell': 'Thomas Mattsel'})

# Select the columns for each DataFrame
imu_data= imu_data[['Name', 'Date', 'Footnote','Period Start Time (24H)', 'Impact Load Total (L+R)','Impact Load Total per Minute (L and R)','Average Intensity (L and R)']]
force_plate= force_plate[['Name','Date','Tags','RSI-modified (Imp-Mom) [m/s]','Eccentric Duration [ms]','Jump Height (Imp-Mom) in Inches [in]','Eccentric Braking RFD / BM [N/s/kg]','Force at Zero Velocity / BM [N/kg]']] 
srpe = srpe[['Name', 'Date', 'SRPE']]

# Standardize date formats across datasets
imu_data['Date'] = pd.to_datetime(imu_data['Date']).dt.strftime('%Y-%m-%d')
force_plate['Date'] = pd.to_datetime(force_plate['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
srpe['Date'] = pd.to_datetime(srpe['Date'], errors='coerce').dt.strftime('%Y-%m-%d')

# Select the columns for each DataFrame
imu_data = imu_data[['Name', 'Date', 'Footnote', 'Period Start Time (24H)', 'Impact Load Total (L+R)', 'Impact Load Total per Minute (L and R)', 'Average Intensity (L and R)']]
force_plate = force_plate[['Name', 'Date', 'Tags', 'RSI-modified (Imp-Mom) [m/s]', 'Eccentric Duration [ms]', 'Jump Height (Imp-Mom) in Inches [in]', 'Eccentric Braking RFD / BM [N/s/kg]', 'Force at Zero Velocity / BM [N/kg]']]
srpe = srpe[['Name', 'Date', 'SRPE']]

#Full data set
complete_data = imu_data.merge(force_plate, on=['Name', 'Date'], how='outer').merge(srpe, on=['Name', 'Date'], how='outer')
all_footnote_data = complete_data[complete_data['Footnote'] == 'All']

# Filter data for 'Pre-Practice' and 'Post-Practice'
pre_data =all_footnote_data [all_footnote_data['Tags'] == 'Pre-Practice']
post_data = all_footnote_data[all_footnote_data['Tags'] == 'Post-Practice']

# Get the column names that are metrics
metric_columns = [col for col in post_data.columns if col not in ['Name', 'Date', 'Tags']]

# Modify the metric column names to add 'Post-Practice'
for metric in metric_columns:
    # Create new column names by appending 'Post-Practice'
    new_metric_name = f'{metric}_Post-Practice'
    # Rename the columns
    post_data = post_data.rename(columns={metric: new_metric_name})
    
    columns_to_remove = [
    'Footnote_Post-Practice',
    'Period Start Time (24H)_Post-Practice',
    'Impact Load Total (L+R)_Post-Practice',
    'Impact Load Total per Minute (L and R)_Post-Practice',
    'Average Intensity (L and R)_Post-Practice',
    'SRPE_Post-Practice'
]

# Drop the columns from the DataFrame
post_data.drop(columns=columns_to_remove, inplace=True)
# Merge the two dataframes based on 'Name' and 'Date' to get rows with both Pre-Practice and Post-Practice
pre_post_data = pre_data.merge(post_data, on=['Name', 'Date'], suffixes=('_Pre', '_Post'), how='inner')

# Define the mapping of names to positions
positions = {
    'Brendan Amoyaw': 'Forward','Cashius McNeilly': 'Guard','Daniel Graham': 'Guard','Elijah Bethune': 'Guard','Jeremiah Francis': 'Guard','Kazim Raza': 'Guard',
    'Matthew Groe': 'Guard','Mike Demagus': 'Guard','Moody Muhammed': 'Forward','Parker Davis': 'Guard','Riaz Saliu': 'Forward','Sebastian Di Manno': 'Guard','Stevan Japundzic': 'Forward',
  'Thomas Mattsel': 'Forward'
}
# Create the 'Position' column based on the 'Name' column
pre_post_data['Position'] = pre_post_data['Name'].map(positions)

# Create a new DataFrame without the 'Name' column
final_data_set = pre_post_data.drop(columns=['Name'])
#final_data_set.to_excel('final_data.xlsx', index=False)


# Statstical Analysis #### look into welch's t-test and Wilcoxon signed-rank test

#A1.Acute Changes – In CMJ metrics pre to post practice: 
    #(RSI-modified (IMP-Mom), Eccentric Duration, Force at Zero Velocity / BM [N/kg], 
    #Jump Height (Imp-Mom) in Inches [in], Eccentric Braking RFD [N/s]/ / BM [N/kg])

# Function to calculate Typical Error of Measurement (TEM)
def calculate_tem(pre_data, post_data):
    differences = post_data - pre_data
    tem = np.sqrt(np.sum(np.square(differences - np.mean(differences))) / (2 * len(differences)))
    return tem

# Function to calculate Smallest Worthwhile Change (SWC)
def calculate_swc(pre_data, coefficient=0.2):
    swc = coefficient * np.std(pre_data, ddof=1)
    return swc

# Function to perform Shapiro-Wilk test for normality
def check_normality(data):
    stat, p = shapiro(data)
    return stat, p, 'normal' if p > 0.05 else 'not normal'

# Function to perform Wilcoxon signed-rank test
def perform_wilcoxon(pre_data, post_data):
    if len(pre_data) > 0 and len(post_data) > 0:
        stat, p = wilcoxon(pre_data, post_data)
    else:
        p = np.nan  # Handling cases with insufficient data
    return p

# Function to perform paired t-test
def perform_ttest(pre_data, post_data):
    if len(pre_data) > 0 and len(post_data) > 0:
        stat, p = ttest_rel(pre_data, post_data)
    else:
        p = np.nan  # Handling cases with insufficient data
    return p

# Function to determine significance
def determine_significance(p_value):
    return "statistically significant" if p_value < 0.05 else "not statistically significant"

# Function to extract descriptive statistics
def descriptive_stats(data):
    stats = {
        'mean': np.mean(data),
        'median': np.median(data),
        'std_dev': np.std(data, ddof=1),
        'min': np.min(data),
        'max': np.max(data)
    }
    return stats

# Function to run the analysis for each metric and group
def analyze_metrics(grouped_data, metrics):
    results = {}
    for position, data in grouped_data:
        print(f"\nAnalyzing position: {position}")
        results[position] = {}
        for pre_metric, post_metric in metrics:
            pre_data = data[pre_metric].dropna()
            post_data = data[post_metric].dropna()

            # Normality Tests
            pre_norm_stat, pre_norm_p, pre_status = check_normality(pre_data)
            post_norm_stat, post_norm_p, post_status = check_normality(post_data)

            # Calculate Tests Regardless of Normality
            wilcox_p = perform_wilcoxon(pre_data, post_data)
            ttest_p = perform_ttest(pre_data, post_data)

            # Calculate TEM and SWC
            tem = calculate_tem(pre_data, post_data)
            swc = calculate_swc(pre_data)

            # Descriptive Statistics
            pre_stats = descriptive_stats(pre_data)
            post_stats = descriptive_stats(post_data)

            # Store Results
            results[position][(pre_metric, post_metric)] = {
                'pre_normality': pre_status,
                'post_normality': post_status,
                'wilcoxon_p': wilcox_p,
                'ttest_p': ttest_p,
                'tem': tem,
                'swc': swc
            }

            # Print Results
            print(f"{pre_metric} to {post_metric}:")
            print(f"  Pre Normality: {pre_status} (p = {pre_norm_p:.3f})")
            print(f"  Post Normality: {post_status} (p = {post_norm_p:.3f})")
            print(f"  Wilcoxon p-value: {wilcox_p:.3f} ({determine_significance(wilcox_p)})")
            print(f"  T-test p-value: {ttest_p:.3f} ({determine_significance(ttest_p)})")
            print(f"  Pre Descriptive Stats: Mean = {pre_stats['mean']:.3f}, Median = {pre_stats['median']:.3f}, Std Dev = {pre_stats['std_dev']:.3f}, Min = {pre_stats['min']:.3f}, Max = {pre_stats['max']:.3f}")
            print(f"  Post Descriptive Stats: Mean = {post_stats['mean']:.3f}, Median = {post_stats['median']:.3f}, Std Dev = {post_stats['std_dev']:.3f}, Min = {post_stats['min']:.3f}, Max = {post_stats['max']:.3f}")
            print(f"  TEM: {tem:.3f}, SWC: {swc:.3f}")

    return results

# Define metrics for comparison
metrics = [
    ("RSI-modified (Imp-Mom) [m/s]", "RSI-modified (Imp-Mom) [m/s]_Post-Practice"),
    ("Eccentric Duration [ms]", "Eccentric Duration [ms]_Post-Practice"),
    ("Jump Height (Imp-Mom) in Inches [in]", "Jump Height (Imp-Mom) in Inches [in]_Post-Practice"),
    ("Eccentric Braking RFD / BM [N/s/kg]", "Eccentric Braking RFD / BM [N/s/kg]_Post-Practice"),
    ("Force at Zero Velocity / BM [N/kg]", "Force at Zero Velocity / BM [N/kg]_Post-Practice")
]

# Example of running analysis
# Assuming 'final_data_set' is your DataFrame grouped by 'Position'
grouped_data = final_data_set.groupby('Position')
results = analyze_metrics(grouped_data, metrics)

##Visuals##

def calculate_metrics_data(data, metrics):
    results = {}
    for pre_metric, post_metric in metrics:
        pre_data = data[pre_metric].dropna()
        post_data = data[post_metric].dropna()
        pre_mean = pre_data.mean()
        post_mean = post_data.mean()
        mean_diff = post_mean - pre_mean
        p_value = perform_wilcoxon(pre_data, post_data)
        results[(pre_metric, post_metric)] = (pre_mean, post_mean, mean_diff, p_value)
    return results

# Calculate metrics for each group

# Filter data for 'Forwards' and 'Guards'
forwards_data = final_data_set[final_data_set['Position'] == 'Forward']
guards_data = final_data_set[final_data_set['Position'] == 'Guard']

# Metrics to plot
metrics_forwards = [
    ("Eccentric Braking RFD / BM [N/s/kg]", "Eccentric Braking RFD / BM [N/s/kg]_Post-Practice"),
    ("Force at Zero Velocity / BM [N/kg]", "Force at Zero Velocity / BM [N/kg]_Post-Practice")
]

metrics_guards = [
    ("RSI-modified (Imp-Mom) [m/s]", "RSI-modified (Imp-Mom) [m/s]_Post-Practice"),
    ("Eccentric Duration [ms]", "Eccentric Duration [ms]_Post-Practice"),
    ("Eccentric Braking RFD / BM [N/s/kg]", "Eccentric Braking RFD / BM [N/s/kg]_Post-Practice"),
    ("Force at Zero Velocity / BM [N/kg]", "Force at Zero Velocity / BM [N/kg]_Post-Practice")
]


metrics_data_forwards = calculate_metrics_data(forwards_data, metrics_forwards)
metrics_data_guards = calculate_metrics_data(guards_data, metrics_guards)



def plot_individual_metric_comparison(results, title_base, colors=('skyblue', 'salmon')):
    for metric, data in results.items():
        plt.clf()  # Clear the current figure before creating a new one.
        fig, ax = plt.subplots(figsize=(10, 10))  # Ensuring a fresh figure is created each time

        pre_mean, post_mean, mean_diff, p_value = data
        metric_name = metric[0].split(" [")[0]  # Simplify the metric name for display

        x = np.arange(1)  # Only one set of bars per metric
        width = 0.4  # Adjusted width for better visibility

        # Plotting the bars with adjusted width and separation
        ax.bar(x - width/1.5, pre_mean, width, label='Pre', color=colors[0])
        ax.bar(x + width/1.5, post_mean, width, label='Post', color=colors[1])

        ax.set_ylabel('Values')
        ax.set_title(f'{title_base}: {metric_name}')
        ax.set_xticks(x)
        ax.set_xticklabels([metric_name])
        ax.legend()

        # Adding significance label
        significance = "*" if p_value < 0.05 else "ns"
        y_height = max(pre_mean, post_mean)
        ax.text(x, y_height + 0.01 * y_height, significance, ha='center', va='bottom')

        plt.show()

# Define colors for the bars
colors_forwards = ('maroon', 'black')
colors_guards = ('maroon', 'black')

# Execute plotting for Forwards and Guards using the new function
plot_individual_metric_comparison(metrics_data_forwards, 'Forwards Metric Comparison', colors_forwards)
plot_individual_metric_comparison(metrics_data_guards, 'Guards Metric Comparison', colors_guards)


