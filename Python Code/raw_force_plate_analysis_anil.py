# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 18:58:14 2024

@author: Palan
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Excel file into a DataFrame
data = pd.read_excel(r"C:\Users\Palan\OneDrive\Documents\School\McMaster\1.Master's Thesis\Thesis_Data_Analysis\raw_data\Anil Palanisamy-Countermovement Jump-2023.09.29-13.35.28-Trial2.xlsx")

# Sum forces from all sensors on the left and right sides to get total force
data['Total_Force'] = data[['Fz1 Left', 'Fz2 Left', 'Fz3 Left', 'Fz4 Left',
                            'Fz1 Right', 'Fz2 Right', 'Fz3 Right', 'Fz4 Right']].sum(axis=1)

# Calculate Body Weight (BW)
def calculate_body_weight(force_data):
    return force_data[:1000].mean()  # Assuming data is sampled at 1000 Hz, first second = first 1000 data points

# Detect onset of movement
def detect_movement_onset(force_data, body_weight):
    threshold = 5 * np.std(force_data[:1000])  # SD of BW during weighing phase
    onset_index = np.where(force_data < body_weight - threshold)[0][0] - 30  # 30 ms prior to force reduction
    return max(onset_index, 0)

# Detect take-off point where force drops below 30N
def detect_takeoff(force_data):
    takeoff_index = np.where(force_data < 30)[0][0]
    return takeoff_index

# Calculate body weight, movement onset, and take-off point
body_weight = calculate_body_weight(data['Total_Force'])
movement_onset = detect_movement_onset(data['Total_Force'], body_weight)
takeoff_point = detect_takeoff(data['Total_Force'])

# Calculate Acceleration: a = F/m
mass = body_weight / 9.81  # Assuming body weight in Newtons and g = 9.81 m/s²
data['Acceleration'] = (data['Total_Force'] - body_weight) / mass  # Net force divided by mass

# Calculate Velocity: Integrate Acceleration over time
data['Velocity'] = np.cumsum(data['Acceleration'] * np.diff(data['Time'], prepend=0))

# Plotting Force-Time and Velocity-Time
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Force on primary y-axis
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Force (N)', color='black')
ax1.plot(data['Time'], data['Total_Force'], label='Total Force', color='black')
#ax1.axvline(x=data['Time'].iloc[movement_onset], color='r', linestyle='--', label='Movement Onset')
ax1.tick_params(axis='y', labelcolor='black')

# Plot Velocity on secondary y-axis
ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Velocity (m/s)', color='maroon')
ax2.plot(data['Time'], data['Velocity'], label='Velocity', color='maroon')
ax2.tick_params(axis='y', labelcolor='black')

# Shade the Eccentric Phase between movement onset and take-off
#ax2.fill_between(data['Time'], data['Velocity'], where=(data['Velocity'] < 0) & (data.index >= movement_onset) & (data.index <= takeoff_point),
 #                color='gray', alpha=0.3, label='Eccentric Phase')

# Title and Labels
#plt.title('Force-Time and Velocity-Time Wave with Movement Onset and Eccentric Phase')
fig.tight_layout()  # Adjust layout to make room for the secondary axis

# Show the plot
plt.show()

