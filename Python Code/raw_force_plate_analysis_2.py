# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 10:21:27 2024

@author: Palan
"""

import pandas as pd
import numpy as np

# Load the Excel file into a DataFrame
force_plate_pre = r"C:\Users\Palan\OneDrive\Documents\School\McMaster\1.Master's Thesis\Thesis_Data_Analysis\Forceplate Data\raw_data\elijah_bethune_pre_practice_trial_1.xlsx"
force_plate_pre = pd.read_excel(force_plate_pre)
def calculate_body_mass_velocity_and_initiation_time(df):
    if 'Time' not in df.columns:
        print("Time column is missing.")
        return None, None, None
    
    df['Total Force'] = df['Z Left'] + df['Z Right']
    
    exceed_threshold_indices = df.index[df['Total Force'] > 300].tolist()
    if not exceed_threshold_indices:
        print("Threshold not exceeded in the dataset.")
        return None, None, None
    
    start_index = exceed_threshold_indices[0]
    
    df['Average Force'] = df['Total Force'].rolling(window=1000, min_periods=1).mean()
    df['Std Dev'] = df['Total Force'].rolling(window=1000, min_periods=1).std()
    
    stable_window_index = df.loc[start_index:]['Std Dev'].idxmin()
    
    mass = df.loc[stable_window_index, 'Average Force'] / 9.81
    
    df['Acceleration'] = (df['Total Force'] - mass * 9.81) / mass
    
    velocities = np.zeros(len(df))
    time_diffs = np.diff(df['Time'].values)
    
    for i in range(stable_window_index, len(df) - 1):
        velocities[i + 1] = velocities[i] + df['Acceleration'].iloc[i] * time_diffs[i - stable_window_index]
    
    df['Velocity'] = velocities
    
    mean_force_stance = df.loc[:stable_window_index, 'Total Force'].mean()
    sd_force_stance = df.loc[:stable_window_index, 'Total Force'].std()
    
    upper_threshold = mean_force_stance + 3 * sd_force_stance
    lower_threshold = mean_force_stance - 3 * sd_force_stance
    
    initiation_indices = df.index[(df['Total Force'] > upper_threshold) | (df['Total Force'] < lower_threshold) & (df.index > stable_window_index)].tolist()
    initiation_time = df.loc[initiation_indices[0], 'Time'] if initiation_indices else None
    
    return mass, df, stable_window_index, initiation_time

def calculate_unweighting_phase(df, body_mass, stable_window_index):
    body_weight_n = body_mass * 9.81
    
    # Adjust this line if the initiation time needs to be used instead of the stable window index for determining the start of unweighting
    unweighting_start_indices = df.index[(df['Total Force'] < body_weight_n) & (df.index > stable_window_index)].tolist()
    
    if not unweighting_start_indices:
        print("No unweighting phase found.")
        return None, None, None, None
    
    unweighting_start_index = unweighting_start_indices[0]
    unweighting_end_indices = df.index[(df['Total Force'] >= body_weight_n) & (df.index > unweighting_start_index)].tolist()
    
    if not unweighting_end_indices:
        print("Force does not return to BW after unweighting phase start.")
        return None, None, None, None
    
    unweighting_end_index = unweighting_end_indices[0]
    auc = np.trapz(df.loc[unweighting_start_index:unweighting_end_index, 'Total Force'], df.loc[unweighting_start_index:unweighting_end_index, 'Time'])
    peak_neg_com_velocity_time = df.loc[unweighting_end_index, 'Time']
    
    return unweighting_start_index, unweighting_end_index, auc, peak_neg_com_velocity_time

# Assuming 'force_plate_pre' DataFrame is already defined and loaded
body_mass, updated_df, stable_window_index, initiation_time = calculate_body_mass_velocity_and_initiation_time(force_plate_pre)
if body_mass and updated_df is not None:
    print(f"Initiation Time: {initiation_time} seconds")
    unweighting_start_index, unweighting_end_index, auc, peak_neg_com_velocity_time = calculate_unweighting_phase(updated_df, body_mass, stable_window_index)
    if unweighting_start_index is not None:
        print(f"Unweighting Phase Start Index: {unweighting_start_index}, End Index: {unweighting_end_index}")
        print(f"Area under the curve (AUC) during unweighting phase: {auc}")
        print(f"Time of peak negative COM velocity: {peak_neg_com_velocity_time} seconds")
    else:
        print("Unweighting phase calculation failed.")
else:
    print("Initial calculations failed.")
