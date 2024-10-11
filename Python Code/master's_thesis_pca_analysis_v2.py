# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 07:07:53 2024

@author: Palan
"""

import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define base path
base_path = r"C:\Users\Palan\OneDrive\Documents\School\McMaster\1.Master's Thesis\Thesis_Data_Analysis"

# Correct file paths
force_plate_path = os.path.join(base_path, "raw_data", "pca_model_forceplate_metrics.xlsx")
srpe_path = os.path.join(base_path, "SRPE", "McMaster MBB 2023 - Sessional RPE (Study Fall-2023) (Responses) - Form Responses 1.xlsx")
imu_data_raw = os.path.join(base_path, "I.M.U data", "StepSessionSummaryExport")

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

# Select relevant columns from IMU data
imu_data = imu_data[['First Name', 'Last Name', 'Date (YYYY-MM-DD)', 'Footnote', 'Impact Load Total (L+R)', 'Average Intensity (L and R)']]
imu_data['Name'] = imu_data['First Name'] + ' ' + imu_data['Last Name']
imu_data.rename(columns={'Date (YYYY-MM-DD)': 'Date'}, inplace=True)

# Load Force Plate and SRPE data
force_plate = pd.read_excel(force_plate_path)
srpe = pd.read_excel(srpe_path)

# Replace specific names in SRPE data (if needed)
name_replacements = {'Kazim Raza1': 'Kazim Raza', 'Brendan Amoyaw1': 'Brendan Amoyaw', 
                     'Moody Mohamud': 'Moody Muhammed', 'Thomas Matsell': 'Thomas Mattsel'}
srpe['Name'].replace(name_replacements, inplace=True)

# Strip leading or trailing spaces from column names in force plate data
force_plate.columns = force_plate.columns.str.strip()

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
columns_to_remove = ['Footnote_Post-Practice', 'Impact Load Total per Minute (L and R)_Post-Practice']
post_data = post_data.drop(columns=[col for col in columns_to_remove if col in post_data.columns], errors='ignore')
pre_post_data = pre_data.merge(post_data, on=['Name', 'Date'], suffixes=('_Pre', '_Post'), how='inner')

# Map names to positions
positions = {'Brendan Amoyaw': 'Forward', 'Cashius McNeilly': 'Guard', 'Daniel Graham': 'Guard', 'Elijah Bethune': 'Guard', 'Jeremiah Francis': 'Guard',
             'Kazim Raza': 'Guard', 'Matthew Groe': 'Guard', 'Mike Demagus': 'Guard', 'Moody Muhammed': 'Forward', 'Parker Davis': 'Guard',
             'Riaz Saliu': 'Forward', 'Sebastian Di Manno': 'Guard', 'Stevan Japundzic': 'Forward', 'Thomas Mattsel': 'Forward'}

pre_post_data['Position'] = pre_post_data['Name'].map(positions)
unique_names = pre_post_data['Name'].unique()
player_ids = {name: idx for idx, name in enumerate(unique_names, 1)}
pre_post_data['PlayerID'] = pre_post_data['Name'].map(player_ids)

# Load the Excel file
df = pd.read_excel(force_plate_path, sheet_name='pca_model_forceplate_metrics')

# Strip leading and trailing spaces from all column names to ensure consistency
df.columns = df.columns.str.strip()

# Drop "Name", "Date", and "Tags" columns (with jump height removed)
df_clean = df.drop(columns=['Name', 'Date', 'Tags', 'Jump Height (Imp-Mom) in Inches [in]'])

# Impute missing values with the mean of each column
imputer = SimpleImputer(strategy='mean')
df_imputed = imputer.fit_transform(df_clean)

# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_imputed)

# Perform PCA to retain components that explain 95% of the variance
pca = PCA(n_components=0.95)
pca_result = pca.fit_transform(df_scaled)

# View explained variance
explained_variance = pca.explained_variance_ratio_
print("Explained variance by each component:", explained_variance)

# Plot PCA results
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1])
plt.title('PCA of Force Plate Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Display component loadings
pca_components = pd.DataFrame(pca.components_, columns=df_clean.columns)
print(pca_components)

# Plot cumulative explained variance to visualize how many components are needed
plt.figure(figsize=(8, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Components')
plt.show()

# 3D Plot of the first three principal components
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2])
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.title('PCA 3D Plot')
plt.show()

# Add jump height back to the PCA result dataframe
pca_result_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
pca_result_df['Jump Height'] = df['Jump Height (Imp-Mom) in Inches [in]']  # Make sure the column is referenced correctly

# Calculate correlations between the principal components and jump height
correlations = pca_result_df.corr()
jump_height_corr = correlations['Jump Height']

print(jump_height_corr)

# Check loadings for PC2
pc2_loadings = pca_components.iloc[1]  # Second row corresponds to PC2
print(pc2_loadings)


# Separate the pre-practice and post-practice data based on the "Tags" column
pre_data = df[df['Tags'] == 'Pre-Practice']
post_data = df[df['Tags'] == 'Post-Practice']

# Strip spaces from column names in case there are any inconsistencies
pre_data.columns = pre_data.columns.str.strip()
post_data.columns = post_data.columns.str.strip()

# Drop "Name", "Date", and "Tags" from both datasets and ensure Jump Height is excluded from the input for PCA
pre_data_clean = pre_data.drop(columns=['Name', 'Date', 'Tags', 'Jump Height (Imp-Mom) in Inches [in]'])
post_data_clean = post_data.drop(columns=['Name', 'Date', 'Tags', 'Jump Height (Imp-Mom) in Inches [in]'])

# Impute missing values for both datasets
imputer = SimpleImputer(strategy='mean')
pre_data_imputed = imputer.fit_transform(pre_data_clean)
post_data_imputed = imputer.fit_transform(post_data_clean)

# Standardize the data for both datasets
scaler = StandardScaler()
pre_data_scaled = scaler.fit_transform(pre_data_imputed)
post_data_scaled = scaler.fit_transform(post_data_imputed)

# Perform PCA for pre-practice data
pca_pre = PCA(n_components=0.95)  # Retain enough components to explain 95% of the variance
pca_pre_result = pca_pre.fit_transform(pre_data_scaled)

# Perform PCA for post-practice data
pca_post = PCA(n_components=0.95)  # Retain enough components to explain 95% of the variance
pca_post_result = pca_post.fit_transform(post_data_scaled)

# Create DataFrames for the PCA results (pre-practice)
pca_pre_result_df = pd.DataFrame(pca_pre_result, columns=[f'PC{i+1}' for i in range(pca_pre_result.shape[1])])
pca_pre_result_df['Jump Height'] = pre_data['Jump Height (Imp-Mom) in Inches [in]'].values

# Create DataFrames for the PCA results (post-practice)
pca_post_result_df = pd.DataFrame(pca_post_result, columns=[f'PC{i+1}' for i in range(pca_post_result.shape[1])])
pca_post_result_df['Jump Height'] = post_data['Jump Height (Imp-Mom) in Inches [in]'].values

# Calculate correlations between the principal components and jump height (pre-practice)
correlations_pre = pca_pre_result_df.corr()
jump_height_corr_pre = correlations_pre['Jump Height']

# Calculate correlations between the principal components and jump height (post-practice)
correlations_post = pca_post_result_df.corr()
jump_height_corr_post = correlations_post['Jump Height']

# Print the correlations for both pre and post-practice
print("Pre-Practice Correlations with Jump Height:")
print(jump_height_corr_pre)

print("Post-Practice Correlations with Jump Height:")
print(jump_height_corr_post)

# Optional: Visualize PCA loadings for both pre- and post-practice
print("Pre-Practice PCA Loadings:")
print(pd.DataFrame(pca_pre.components_, columns=pre_data_clean.columns))

print("Post-Practice PCA Loadings:")
print(pd.DataFrame(pca_post.components_, columns=post_data_clean.columns))

# Loadings for the key components pre- and post-practice
pc2_pre_loadings = pca_pre.components_[1]  # PC2 pre-practice
pc1_post_loadings = pca_post.components_[0]  # PC1 post-practice

print("PC2 Pre-Practice Loadings:")
print(pd.DataFrame(pc2_pre_loadings, index=pre_data_clean.columns))

print("PC1 Post-Practice Loadings:")
print(pd.DataFrame(pc1_post_loadings, index=post_data_clean.columns))

