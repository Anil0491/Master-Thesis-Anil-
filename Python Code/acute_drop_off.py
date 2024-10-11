# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 07:17:07 2023

@author: Palan
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 17:46:14 2023

@author: Palan
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from IPython.display import clear_output
import seaborn as sns

# Define file paths for three different data sources: IMU data (in CSV format), Force Plate data (in Excel format), and SRPE data (in Excel format).
imu_data_raw = r'C:\Users\Palan\OneDrive\Documents\Work\McMaster Performance\Sport Science\Data Management\2023-24\Raw_Data\StepSessionSummaryExport'
force_plate = r'C:\Users\Palan\OneDrive\Documents\Work\McMaster Performance\Sport Science\Data Management\2023-24\Raw_Data\pre_post_masters_data.xlsx'
srpe = r'C:\Users\Palan\OneDrive\Documents\Work\McMaster Performance\Sport Science\Data Management\RPE Export\McMaster MBB 2023 - Sessional RPE (Study Fall-2023) (Responses).xlsx'


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
imu_data= imu_data[['Name', 'Date', 'Footnote','Period Start Time (24H)', 'Impact Load Total (L+R)','Impact Load Total per Minute (L and R)','Average Intensity (L and R)','Max Bone Stimulus (L)','Max Bone Stimulus (R)','Impact Asymmetry']]
force_plate= force_plate[['Name','Date','Tags','RSI-modified [m/s] ','Eccentric Duration [ms] ','Jump Height (Imp-Mom) [cm] ','Concentric RFD / BM [N/s/kg] ','Eccentric Braking RFD [N/s] ','Concentric Peak Force / BM [N/kg] ','Eccentric Peak Force / BM [N/kg] ', 'Eccentric Braking Impulse [N s] ','Concentric Impulse [N s] ']] 
srpe = srpe[['Name', 'Date', 'SRPE']]

# Convert the "Date" column to datetime for all DataFrames
imu_data['Date'] = pd.to_datetime(imu_data['Date'])
force_plate['Date'] = pd.to_datetime(force_plate['Date'])
srpe['Date'] = pd.to_datetime(srpe['Date'])

# Ensure consistent date format in all DataFrames
imu_data['Date'] = imu_data['Date'].dt.strftime('%Y-%m-%d')
force_plate['Date'] = force_plate['Date'].dt.strftime('%Y-%m-%d')
srpe['Date'] = srpe['Date'].dt.strftime('%Y-%m-%d')


#Full data set
complete_data = imu_data.merge(force_plate, on=['Name', 'Date'], how='outer').merge(srpe, on=['Name', 'Date'], how='outer')

all_footnote_data = complete_data[complete_data['Footnote'] == 'All']


#all_footnote_data.to_excel('sample_data.xlsx', index=False)


# Filter data for 'Pre-Practice'
pre_data =all_footnote_data [all_footnote_data['Tags'] == 'Pre-Practice']
# Filter data for 'Post-Practice'
post_data = all_footnote_data[all_footnote_data['Tags'] == 'Post-Practice']

# Get the column names that are metrics
metric_columns = [col for col in post_data.columns if col not in ['Name', 'Date', 'Tags']]

# Modify the metric column names to add 'Post-Practice'
for metric in metric_columns:
    # Create new column names by appending 'Post-Practice'
    new_metric_name = f'{metric}_Post-Practice'
    # Rename the columns
    post_data = post_data.rename(columns={metric: new_metric_name})
    
    
#columns_to_remove = [
    'Footnote_Post-Practice',
    'Period Start Time (24H)_Post-Practice',
    'Impact Load Total (L+R)_Post-Practice',
    'Impact Load Total per Minute (L and R)_Post-Practice',
    'Average Intensity (L and R)_Post-Practice',
    'Max Bone Stimulus (L)_Post-Practice',
    'Max Bone Stimulus (R)_Post-Practice'
#]

# Drop the columns from the DataFrame
#post_data.drop(columns=columns_to_remove, inplace=True)
# Merge the two dataframes based on 'Name' and 'Date' to get rows with both Pre-Practice and Post-Practice
pre_post_data = pre_data.merge(post_data, on=['Name', 'Date'], suffixes=('_Pre', '_Post'), how='inner')
#print(pre_post_data.columns)
pre_post_data.to_excel('pre_post.xlsx', index=True)
#print(pre_post_data.columns)


############ DATA VISUIAL#############

pre_post_data.columns = pre_post_data.columns.str.strip()

# Get the unique player names from the 'Name' column
unique_names = pre_post_data['Name'].unique()

metric_names = [
    'Eccentric Duration [ms]','RSI-modified [m/s]','Jump Height (Imp-Mom) [cm]','Concentric RFD / BM [N/s/kg]',
    # Add other metric names here
]

# Iterate through each player
for name in unique_names:
    player_data = pre_post_data[pre_post_data['Name'] == name]

    # Create a separate line graph for each metric
    for metric in metric_names:
        plt.close('all')
        plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

        # Create column names for Pre-Practice and Post-Practice data
        pre_col = metric
        post_col = f'{metric} _Post-Practice'

        # Plot the data for the current metric
        plt.plot(player_data['Date'], player_data[pre_col], label=f'{metric} (Pre-Practice)', marker='o', linestyle='-')
        plt.plot(player_data['Date'], player_data[post_col], label=f'{metric} (Post-Practice)', marker='o', linestyle='-')

        # Customize the graph
        plt.title(f'{name} - {metric} Comparison')
        plt.xlabel('Date')
        plt.ylabel(metric)
        plt.legend()

        # Show the graph
        plt.grid(True)
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.show()
        plt.close()


####### SCATTER PLOTS FOR DROPS VS IMPACT LOAD AND AVG INTENSITY 

# Calculate drop-off for specific columns
dropoff_data = pre_post_data[['Name']].copy()  # Create a copy of the DataFrame

# Include the specified Post-Practice metrics
specified_metrics = ['RSI-modified [m/s]',
                     'Eccentric Duration [ms]',
                     'Jump Height (Imp-Mom) [cm]',
                     'Concentric RFD / BM [N/s/kg]',
                     ]

# Iterate through each specified Post-Practice metric
for metric in specified_metrics:
    pre_col = metric
    post_col = f'{metric} _Post-Practice'  # No need for "_Post-Practice" in the column name
    
    # Calculate the drop-off as the difference between Post-Practice and Pre-Practice values
    dropoff_data[f'{metric}_Dropoff'] = pre_post_data[pre_col] - pre_post_data[post_col]

# Include the additional columns in the drop-off DataFrame
additional_columns = ['Impact Load Total (L+R)', 'Impact Load Total per Minute (L and R)',
                      'Average Intensity (L and R)', 'Max Bone Stimulus (L)',
                      'Max Bone Stimulus (R)', 'Impact Asymmetry','SRPE']

for column in additional_columns:
    dropoff_data[column] = pre_post_data[column]

# Print the drop-off data
#print(dropoff_data)

# Define the columns to be plotted against 'Impact Load Total (L+R)' and 'Average Intensity (L and R)'
columns_to_plot = ['RSI-modified [m/s]_Dropoff',
                   'Eccentric Duration [ms]_Dropoff',
                   'Jump Height (Imp-Mom) [cm]_Dropoff',
                   'Concentric RFD / BM [N/s/kg]_Dropoff',
                   ]

# Create scatter plots for each combination
for column in columns_to_plot:
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    
    # Scatter plot for 'Impact Load Total (L+R)' and the specified drop-off metric
    plt.scatter(dropoff_data['Impact Load Total (L+R)'], dropoff_data[column], label=column)
    
    # Customize the graph
    plt.title(f'Scatter Plot: Impact Load Total (L+R) vs. {column}')
    plt.xlabel('Impact Load Total (L+R)')
    plt.ylabel(column)
    plt.legend()
    
    # Show the graph
    plt.grid(True)
    plt.show()

# Create scatter plots for 'Average Intensity (L and R)' and the drop-off metrics
for column in columns_to_plot:
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    
    # Scatter plot for 'Average Intensity (L and R)' and the specified drop-off metric
    plt.scatter(dropoff_data['Average Intensity (L and R)'], dropoff_data[column], label=column)
    
    # Customize the graph
    plt.title(f'Scatter Plot: Average Intensity (L and R) vs. {column}')
    plt.xlabel('Average Intensity (L and R)')
    plt.ylabel(column)
    plt.legend()
    
    # Show the graph
    plt.grid(True)
    plt.show()


#########################ALL PRE_POST JUMP CORRELATION ANALYSIS############################

# Define the drop-off metrics to correlate with 'Impact Load Total (L+R)'
dropoff_metrics = ['RSI-modified [m/s]_Dropoff',
                   'Eccentric Duration [ms]_Dropoff',
                   'Jump Height (Imp-Mom) [cm]_Dropoff',
                   'Concentric RFD / BM [N/s/kg]_Dropoff']

# Perform correlation analysis for each metric
for metric in dropoff_metrics:
    # Create a subset of the data with 'Impact Load Total (L+R)' and the current drop-off metric
    correlation_data = dropoff_data[['Impact Load Total (L+R)', metric]]

    # Calculate the Pearson correlation coefficient
    correlation = correlation_data.corr().iloc[0, 1]

    # Print the correlation result for the current metric
    print(f'Correlation between Impact Load Total (L+R) and {metric}: {correlation}')


# Define the drop-off metrics to correlate with 'Average Intensity'
dropoff_metrics = ['RSI-modified [m/s]_Dropoff',
                   'Eccentric Duration [ms]_Dropoff',
                   'Jump Height (Imp-Mom) [cm]_Dropoff',
                   'Concentric RFD / BM [N/s/kg]_Dropoff']

# Perform correlation analysis for each metric
for metric in dropoff_metrics:
    # Create a subset of the data with 'Average Intensity (L and R)' and the current drop-off metric
    correlation_data = dropoff_data[['Average Intensity (L and R)', metric]]

    # Calculate the Pearson correlation coefficient
    correlation = correlation_data.corr().iloc[0, 1]

    # Print the correlation result for the current metric
    print(f'Correlation between Average Intensity (L and R) and {metric}: {correlation}')
    
    
    # Define the drop-off metrics to correlate with 'Average Intensity'
    dropoff_metrics = ['RSI-modified [m/s]_Dropoff',
                       'Eccentric Duration [ms]_Dropoff',
                       'Jump Height (Imp-Mom) [cm]_Dropoff',
                       'Concentric RFD / BM [N/s/kg]_Dropoff']
    
 # Perform correlation analysis for each metric
for metric in dropoff_metrics:
    # Create a subset of the data with 'Average Intensity (L and R)' and the current drop-off metric
    correlation_data = dropoff_data[['SRPE', metric]]

    # Calculate the Pearson correlation coefficient
    correlation = correlation_data.corr().iloc[0, 1]

    # Print the correlation result for the current metric
    print(f'SRPE and {metric}: {correlation}')
    
    
    # Define the drop-off metrics to correlate with 'Average Intensity'
    dropoff_metrics = ['RSI-modified [m/s]_Dropoff',
                       'Eccentric Duration [ms]_Dropoff',
                       'Jump Height (Imp-Mom) [cm]_Dropoff',
                       'Concentric RFD / BM [N/s/kg]_Dropoff']


########INDIVIDUAL PLAYER ANALYSIS###################

# Define the drop-off metrics and target variable
dropoff_metrics = ['RSI-modified [m/s]_Dropoff',
                   'Eccentric Duration [ms]_Dropoff',
                   'Jump Height (Imp-Mom) [cm]_Dropoff',
                   'Concentric RFD / BM [N/s/kg]_Dropoff']
target_variable = 'Impact Load Total (L+R)'

# Group the data by 'Name'
grouped_data = dropoff_data.groupby('Name')

# Create an empty list to store correlation results
correlation_results = []

# Perform correlation analysis for each individual and store the results
for name, group in grouped_data:
    individual_correlations = {}
    
    # Calculate and store the correlation for each drop-off metric
    for metric in dropoff_metrics:
        correlation_data = group[[target_variable, metric]]
        correlation = correlation_data.corr().iloc[0, 1]
        individual_correlations[metric] = correlation
    
    # Store the individual's correlation results
    correlation_results.append({'Name': name, **individual_correlations})

# Create a DataFrame from the correlation results
correlation_df = pd.DataFrame(correlation_results)
