# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 16:07:57 2024

@author: Palan
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


force_plate_pre = r"C:\Users\Palan\OneDrive\Documents\School\McMaster\1.Master's Thesis\Thesis_Data_Analysis\Forceplate Data\raw_data\elijah_bethune_pre_practice_trial_1.xlsx"
#force_plate_post = r"C:\Users\Palan\OneDrive\Documents\School\McMaster\1.Master's Thesis\Thesis_Data_Analysis\Forceplate Data\raw_data\elijah_bethune_post_practice_raw.xlsx"

# Load data frames
force_plate_pre = pd.read_excel(force_plate_pre)
#force_plate_post = pd.read_excel(force_plate_post)

# Calculate total force by summing left and right forces
force_plate_pre['Total Force'] = force_plate_pre['Z Left'] + force_plate_pre['Z Right']
#force_plate_post['Total Force'] = force_plate_post['Z Left'] + force_plate_post['Z Right']


def calculate_body_mass_and_velocity(df):
    # Find index where force exceeds the threshold
    exceed_threshold_indices = df.index[df['Total Force'] > 300].tolist()
    
    if not exceed_threshold_indices:
        print("Threshold not exceeded in the dataset.")
        return None, None

    start_index = exceed_threshold_indices[0]
    
    # Calculate average force and standard deviation for each 1-second window AFTER the threshold is exceeded
    df['Average Force'] = df['Total Force'].rolling(window=1000, min_periods=1).mean()
    df['Std Dev'] = df['Total Force'].rolling(window=1000, min_periods=1).std()

    # Find the window with the lowest standard deviation (most stable) after the threshold is exceeded
    # This ensures the selection of a period with minimal movement for accurate body mass calculation
    stable_window_index = df.loc[start_index:]['Std Dev'].idxmin()
    stable_window = df.loc[stable_window_index]

    # Calculate body mass using F = m * g, where F is the force, g is the gravitational acceleration (9.81 m/s^2)
    mass = stable_window['Average Force'] / 9.81

    # Proceed with acceleration and velocity calculations after body mass calculation
    # Initialize acceleration for the whole dataset but start calculations from the stable window
    df['Acceleration'] = (df['Total Force'] - mass * 9.81) / mass

    # To calculate velocity, start from the stable window index where body mass is calculated
    # Initialize velocity with zeros and calculate only after the stable window
    velocities = np.zeros(len(df))
    time_diffs = np.diff(df['Time'].values)  # Assuming time is continuous and evenly sampled
    
    for i in range(stable_window_index, len(df) - 1):  # Start from stable window to end of dataframe
        velocities[i + 1] = velocities[i] + df['Acceleration'].iloc[i] * time_diffs[i - stable_window_index]

    df['Velocity'] = velocities

    return mass, df, stable_window_index
# Usage
body_mass, updated_df, stable_window_index = calculate_body_mass_and_velocity(force_plate_pre)

# Use the function and unpack the returned values
body_mass, updated_df, stable_window_index = calculate_body_mass_and_velocity(force_plate_pre)

# Body Weight in Newtons
body_weight_n = body_mass * 9.81  # Convert mass to weight

# Calculate the mean and SD of the force during the stable period
mean_force = updated_df.loc[:stable_window_index, 'Total Force'].mean()
sd_force = updated_df.loc[:stable_window_index, 'Total Force'].std()

# Define thresholds for movement initiation
upper_threshold = mean_force + 5 * sd_force
lower_threshold = mean_force - 5 * sd_force

# Identify the initiation of movement
# This searches for the first instance where force exceeds the defined thresholds after the stable window
movement_start_indices = updated_df.index[(updated_df['Total Force'] > upper_threshold) | (updated_df['Total Force'] < lower_threshold) & (updated_df.index > stable_window_index)]

if not movement_start_indices.empty:
    movement_start_index = movement_start_indices[0]
    movement_start_time = updated_df.loc[movement_start_index, 'Time']
    print(f"Movement initiation detected at index {movement_start_index}, time {movement_start_time:.2f}s")
else:
    print("No clear movement initiation detected based on the criteria.")


# Rest of your code ...

# Calculate the residual force after calculating body mass
updated_df['Residual Force'] = updated_df['Total Force'] - body_weight_n

# Calculate the mean and SD of the force during the stable period to determine the movement initiation
mean_force = updated_df.loc[:stable_window_index, 'Total Force'].mean()
sd_force = updated_df.loc[:stable_window_index, 'Total Force'].std()

# Define thresholds for movement initiation
upper_threshold = mean_force + 5 * sd_force
lower_threshold = mean_force - 5 * sd_force

# Identify the initiation of movement
movement_start_indices = updated_df.index[
    ((updated_df['Total Force'] > upper_threshold) | (updated_df['Total Force'] < lower_threshold)) &
    (updated_df.index > stable_window_index)
].tolist()

if movement_start_indices:
    movement_start_index = movement_start_indices[0]
    movement_start_time = updated_df.loc[movement_start_index, 'Time']
    print(f"Movement initiation detected at index {movement_start_index}, time {movement_start_time:.2f}s")

    # Now calculate the mean and SD of the residual force after movement initiation for takeoff detection
    mean_residual_force_post_movement = updated_df.loc[movement_start_index:, 'Residual Force'].mean()
    sd_residual_force_post_movement = updated_df.loc[movement_start_index:, 'Residual Force'].std()

    takeoff_threshold = mean_residual_force_post_movement + 5 * sd_residual_force_post_movement

    # Identify the takeoff initiation point
    takeoff_start_indices = updated_df.index[
        (updated_df['Residual Force'] > takeoff_threshold) &
        (updated_df.index > movement_start_index)
    ].tolist()

    # If there is a takeoff point, identify the first instance after movement initiation
    if takeoff_start_indices:
        takeoff_start_index = takeoff_start_indices[0]
        takeoff_start_time = updated_df.loc[takeoff_start_index, 'Time']
        # Visualization with Takeoff Point
        plt.figure(figsize=(15, 7))
        plt.plot(updated_df['Time'], updated_df['Total Force'], label='Total Force')
        plt.axhline(y=body_weight_n, color='r', linestyle='--', label='Body Weight in Newtons')
        plt.axvline(x=takeoff_start_time, color='g', linestyle='--', label='Takeoff Point')
        plt.title('Force Analysis with Takeoff Point')
        plt.xlabel('Time (s)')
        plt.ylabel('Force (N)')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("No clear takeoff initiation detected based on the criteria.")
else:
    print("No clear movement initiation detected based on the criteria.")

























 











