# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 11:37:39 2024

@author: Palan
"""

import pandas as pd
import os
from scipy.stats import shapiro
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel


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

# Write the final DataFrame to an Excel file
final_data_set.to_excel('final_data.xlsx', index=False)


####A1.Acute Changes – In CMJ metrics pre to post practice: 
    #(RSI-modified (IMP-Mom), Eccentric Duration, Force at Zero Velocity / BM [N/kg], 
    #Jump Height (Imp-Mom) in Inches [in], Eccentric Braking RFD [N/s]/ / BM [N/kg])

# Define variables for pre-practice
pre_practice_variables = [
    'Impact Load Total (L+R)',
    'Average Intensity (L and R)',
    'RSI-modified (Imp-Mom) [m/s]',
    'Eccentric Duration [ms]',
    'Jump Height (Imp-Mom) in Inches [in]',
    'Eccentric Braking RFD / BM [N/s/kg]',
    'Force at Zero Velocity / BM [N/kg]',
    'SRPE'
]

# Define corresponding post-practice variables
post_practice_variables = [
    'RSI-modified (Imp-Mom) [m/s]_Post-Practice',
    'Eccentric Duration [ms]_Post-Practice',
    'Jump Height (Imp-Mom) in Inches [in]_Post-Practice',
    'Eccentric Braking RFD / BM [N/s/kg]_Post-Practice',
    'Force at Zero Velocity / BM [N/kg]_Post-Practice'
]

# Calculate descriptive statistics for pre-practice and post-practice
pre_practice_stats = final_data_set[pre_practice_variables].describe()
post_practice_stats = final_data_set[post_practice_variables].describe()

# Calculate SWC and TEM for each variable with both pre- and post-practice data
results = {}
for var in pre_practice_variables:
    post_var = var + '_Post-Practice'
    if post_var in final_data_set.columns:
        # Calculate SWC and TEM
        sd_between = final_data_set[var].std()
        swc = 0.2 * sd_between
        differences = final_data_set[var] - final_data_set[post_var]
        tem = differences.std() / np.sqrt(2)
        
        # Store results
        results[var] = {'SWC': swc, 'TEM': tem, 'SD_Between': sd_between}

# Print results
for var, metrics in results.items():
    print(f"{var}: SWC = {metrics['SWC']:.2f}, TEM = {metrics['TEM']:.2f}, SD_Between = {metrics['SD_Between']:.2f}")

print("Pre-Practice Descriptive Statistics:")
print(pre_practice_stats)
print("\nPost-Practice Descriptive Statistics:")
print(post_practice_stats)


import pandas as pd

# Assuming final_data_set is already loaded as a DataFrame

# Set the 'Date' column as datetime type if it isn't already
final_data_set['Date'] = pd.to_datetime(final_data_set['Date'])

# Sorting data by Date to ensure plots are in chronological order
final_data_set.sort_values('Date', inplace=True)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import norm

# Assuming final_data_set is already loaded as a DataFrame

# Calculate overall means for the entire dataset
overall_mean_impact_load = final_data_set['Impact Load Total (L+R)'].mean()
overall_mean_intensity = final_data_set['Average Intensity (L and R)'].mean()

# Group data by 'Date' and calculate standard deviation for each group
std_dev_data = final_data_set.groupby('Date').agg({
    'Impact Load Total (L+R)': 'std',
    'Average Intensity (L and R)': 'std'
}).reset_index()

# Calculate deviations from the overall mean for standard deviations
std_dev_data['Deviation from Mean (Impact Load)'] = std_dev_data['Impact Load Total (L+R)'] - overall_mean_impact_load
std_dev_data['Deviation from Mean (Intensity)'] = std_dev_data['Average Intensity (L and R)'] - overall_mean_intensity

# Plotting Deviation from Mean for Impact Load Total (L+R)
plt.figure(figsize=(12, 6))
sns.distplot(std_dev_data['Deviation from Mean (Impact Load)'], fit=norm, kde=True, color='blue')
plt.title('Distribution of Deviation from Mean for Impact Load Total (L+R)')
plt.xlabel('Deviation from Mean (SD)')
plt.ylabel('Density')

# Show the first plot
plt.show()

# Plotting Deviation from Mean for Average Intensity (L and R)
plt.figure(figsize=(12, 6))
sns.distplot(std_dev_data['Deviation from Mean (Intensity)'], fit=norm, kde=True, color='red')
plt.title('Distribution of Deviation from Mean for Average Intensity (L and R)')
plt.xlabel('Deviation from Mean (SD)')
plt.ylabel('Density')

# Show the second plot
plt.show()
