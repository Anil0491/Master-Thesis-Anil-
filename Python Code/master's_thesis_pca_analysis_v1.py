# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 10:22:11 2024

@author: Palan
"""

import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

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

# Step 1: Identify numeric columns
numeric_columns = pre_post_data.select_dtypes(include=[np.number]).columns
non_numeric_columns = pre_post_data.select_dtypes(exclude=[np.number]).columns

# Step 2: Apply imputation only to numeric columns
pre_post_data_numeric = pre_post_data[numeric_columns]
imputer = SimpleImputer(strategy='mean')

# Perform imputation on numeric data only
pre_post_data_numeric_filled = pd.DataFrame(imputer.fit_transform(pre_post_data_numeric), 
                                            columns=pre_post_data_numeric.columns)

# Step 3: Merge back the non-numeric data
pre_post_data_non_numeric = pre_post_data[non_numeric_columns].reset_index(drop=True)
pre_post_data_filled = pd.concat([pre_post_data_non_numeric, pre_post_data_numeric_filled], axis=1)

# Step 4: Select only the numeric columns (imputed) for PCA
X = pre_post_data_filled[numeric_columns]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Apply PCA
pca = PCA()
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)

# Step 6: Visualize the PCA results

# Scree plot to visualize the explained variance
plt.figure(figsize=(8, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot')
plt.grid(True)
plt.show()

# Updated PCA biplot with adjusted label placement
plt.figure(figsize=(12, 8))  # Increase figure size
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)

# Add player names as labels with bounding boxes for better readability
for i, name in enumerate(pre_post_data_filled['Name']):
    plt.text(X_pca[i, 0], X_pca[i, 1], name, fontsize=8, alpha=0.75, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

# Add arrows for each feature (change metrics)
for i, feature in enumerate(numeric_columns):
    plt.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], color='r', alpha=0.5)
    plt.text(pca.components_[0, i] * 1.2, pca.components_[1, i] * 1.2, feature, color='g', ha='center', va='center')

plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Biplot with Player Names (Improved)')
plt.grid(True)
plt.show()


# Explained variance ratios
explained_variance_df = pd.DataFrame({
    'Component': np.arange(1, len(pca.explained_variance_ratio_) + 1),
    'Explained Variance Ratio': pca.explained_variance_ratio_
})

# Replace ace_tools with standard Pandas display for explained variance ratios
explained_variance_df = pd.DataFrame({
    'Component': np.arange(1, len(pca.explained_variance_ratio_) + 1),
    'Explained Variance Ratio': pca.explained_variance_ratio_
})

# Display the explained variance ratios using Pandas
print(explained_variance_df)

# Number of top components to consider based on scree plot
num_top_components = 10  # You can adjust this based on your analysis

# Get the component loadings (PCA components)
loadings = pd.DataFrame(pca.components_[:num_top_components, :], columns=numeric_columns)

# Display the top components and their corresponding metrics
for i in range(num_top_components):
    print(f"\nPrincipal Component {i+1}")
    sorted_loadings = loadings.iloc[i].abs().sort_values(ascending=False)
    print("Top contributing metrics:")
    print(sorted_loadings.head(10))  # Display the top 10 contributing metrics for each component

