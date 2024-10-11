# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 21:38:46 2024

@author: Palan
"""
##B1.	 Acute Associations – Is there a significant association between practice workload, 
#measured by IMUs (external load) and/or RPE (internal load), and changes in CMJ metrics: 
    #(RSI-modified (IMP-Mom), Eccentric Duration, Force at Zero Velocity / BM [N/kg], Jump Height (Imp-Mom)
    #in Inches [in], Eccentric Braking RFD [N/s]/ / BM [N/kg]. 
    
    ##This code will look to build a linear mixed model analysis
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro

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

# Write the final DataFrame to an Excel file
#pre_post_data.to_excel('linear_mixed_model_data.xlsx', index=False)

# Columns to keep as identifiers
id_vars = ['Name', 'Date', 'Footnote']

# Variables measured both pre and post
value_vars = [
    'RSI-modified (Imp-Mom) [m/s]', 
    'Eccentric Duration [ms]', 
    'Jump Height (Imp-Mom) in Inches [in]', 
    'Eccentric Braking RFD / BM [N/s/kg]', 
    'Force at Zero Velocity / BM [N/kg]'
]

# Melting the dataframe
data_long = pd.melt(pre_post_data, id_vars=id_vars, 
                    value_vars=value_vars + [f"{var}_Post-Practice" for var in value_vars],
                    var_name='Measurement_Type', value_name='Value')

# Create a new 'Time' variable to indicate pre- or post-practice
data_long['Time'] = data_long['Measurement_Type'].apply(lambda x: 'Post' if 'Post-Practice' in x else 'Pre')

# Adjust 'Measurement_Type' to have consistent naming for pre and post
data_long['Measurement_Type'] = data_long['Measurement_Type'].str.replace('_Post-Practice', '')

# Display the first few rows of the reshaped dataframe
data_long.head()


##Checks for Normality Test 

n_rows = len(value_vars)

# Set up the matplotlib figure
fig, axes = plt.subplots(n_rows, 1, figsize=(20, 5 * n_rows))

# Iterate over each variable to create a subplot for each
for i, var in enumerate(value_vars):
    # Plot pre-practice data
    sns.histplot(data=pre_post_data, x=var, color='blue', label='Pre-Practice', kde=True, ax=axes[i])
    # Plot post-practice data
    sns.histplot(data=pre_post_data, x=f'{var}_Post-Practice', color='red', label='Post-Practice', kde=True, ax=axes[i])
    
    # Set the title and labels
    axes[i].set_title(f'Distribution of {var} Pre and Post Practice')
    axes[i].set_xlabel(f'{var} [units]')
    axes[i].set_ylabel('Frequency')
    
    # Add the legend
    axes[i].legend()

# Adjust the layout
plt.tight_layout()
plt.show()

# Initialize an empty dictionary to store the Shapiro-Wilk results
normality_results = {}

# Loop through each measurement type and perform the Shapiro-Wilk test
for var in value_vars:
    # Extract pre and post data for the variable
    pre_data = data_long[(data_long['Measurement_Type'] == var) & (data_long['Time'] == 'Pre')]['Value']
    post_data = data_long[(data_long['Measurement_Type'] == var) & (data_long['Time'] == 'Post')]['Value']
    
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