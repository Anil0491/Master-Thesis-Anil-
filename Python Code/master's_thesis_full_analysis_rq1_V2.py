# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:44:04 2024

@author: Palan
"""

import pandas as pd
import os
from scipy.stats import shapiro
import scipy.stats as stats

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


metrics = [
    ("RSI-modified (Imp-Mom) [m/s]", "RSI-modified (Imp-Mom) [m/s]_Post-Practice"),
    ("Eccentric Duration [ms]", "Eccentric Duration [ms]_Post-Practice"),
    ("Jump Height (Imp-Mom) in Inches [in]", "Jump Height (Imp-Mom) in Inches [in]_Post-Practice"),
    ("Eccentric Braking RFD / BM [N/s/kg]", "Eccentric Braking RFD / BM [N/s/kg]_Post-Practice"),
    ("Force at Zero Velocity / BM [N/kg]", "Force at Zero Velocity / BM [N/kg]_Post-Practice")
]

# Group the data by 'Position'
grouped = final_data_set.groupby('Position')

# Iterate over each position and perform the analyses
for position, data in grouped:
    print(f'\nPosition: {position}\n')
    print("--------------------------------------------------")

    # Perform Shapiro-Wilk test and provide descriptive statistics for each metric
    for pre_metric, post_metric in metrics:
        # Prepare data by dropping NaN values for both pre and post
        pre_values = data[pre_metric].dropna()
        post_values = data[post_metric].dropna()

        # Shapiro-Wilk Test for Pre-Practice
        pre_stat, pre_p = shapiro(pre_values)
        print(f'Pre-Practice {pre_metric}: Statistic = {pre_stat:.3f}, P-value = {pre_p:.3f}')
        if pre_p < 0.05:
            print("    -> The distribution is not normal.")
        else:
            print("    -> The distribution is normal.")

        # Descriptive Statistics for Pre-Practice
        pre_desc_stats = pre_values.describe()
        print(f'Pre-Practice Descriptive Statistics for {pre_metric}:')
        for stat, value in pre_desc_stats.items():
            print(f"    {stat}: {value:.2f}")

        # Shapiro-Wilk Test for Post-Practice
        post_stat, post_p = shapiro(post_values)
        print(f'Post-Practice {post_metric}: Statistic = {post_stat:.3f}, P-value = {post_p:.3f}')
        if post_p < 0.05:
            print("    -> The distribution is not normal.")
        else:
            print("    -> The distribution is normal.")

        # Descriptive Statistics for Post-Practice
        post_desc_stats = post_values.describe()
        print(f'Post-Practice Descriptive Statistics for {post_metric}:')
        for stat, value in post_desc_stats.items():
            print(f"    {stat}: {value:.2f}")
        print("--------------------------------------------------")
        
        # Calculate mean differences, smallest worthwhile change and perform Wilcoxon signed-rank test
        # Function to perform analysis
def analyze_group(data_set, group_name, metrics):
    print(f"\nAnalysis for {group_name}")
    print("-" * 50)  # Adding a separator for better readability
    for pre_metric, post_metric in metrics:
        pre_data = data_set[pre_metric]
        post_data = data_set[post_metric]
        differences = post_data - pre_data
        # Handling cases where data might have non-paired or NaN values
        valid_indices = pre_data.notna() & post_data.notna()
        if valid_indices.sum() > 0:
            stat, p_value = stats.wilcoxon(pre_data[valid_indices], post_data[valid_indices])
            swc = 0.2 * pre_data.std()
            print(f"{pre_metric} to {post_metric}: Mean Difference = {differences.mean():.3f}, SWC = {swc:.3f}, p-value = {p_value:.3f}")
            if p_value < 0.05:
                print("    -> The test result is significant.")
            else:
                print("    -> The test result is not significant.")
        else:
            print(f"No valid data to compare for {pre_metric} to {post_metric}")


metrics = [
    ("RSI-modified (Imp-Mom) [m/s]", "RSI-modified (Imp-Mom) [m/s]_Post-Practice"),
    ("Eccentric Duration [ms]", "Eccentric Duration [ms]_Post-Practice"),
    ("Jump Height (Imp-Mom) in Inches [in]", "Jump Height (Imp-Mom) in Inches [in]_Post-Practice"),
    ("Eccentric Braking RFD / BM [N/s/kg]", "Eccentric Braking RFD / BM [N/s/kg]_Post-Practice"),
    ("Force at Zero Velocity / BM [N/kg]", "Force at Zero Velocity / BM [N/kg]_Post-Practice")
]

# Separate data by position
final_data_set_forwards = final_data_set[final_data_set['Position'] == 'Forward']
final_data_set_guards = final_data_set[final_data_set['Position'] == 'Guard']

# Perform analysis for Forwards & Guards
analyze_group(final_data_set_forwards, "Forwards", metrics)
analyze_group(final_data_set_guards, "Guards", metrics)

