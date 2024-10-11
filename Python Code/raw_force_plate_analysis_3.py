# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 16:24:52 2024

@author: Palan
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load the Excel and CSV files into DataFrames
force_plate_pre = pd.read_excel(r"C:\Users\Palan\OneDrive\Documents\School\McMaster\1.Master's Thesis\Thesis_Data_Analysis\Sebastian Di Manno-Countermovement Jump-2023.10.03-09.30.26-Trial3(Pre-test).xlsx")
force_plate_post = pd.read_excel(r"C:\Users\Palan\OneDrive\Documents\School\McMaster\1.Master's Thesis\Thesis_Data_Analysis\Sebastian Di Manno-Countermovement Jump-2023.10.03-11.20.19-Trial3 (post-test).xlsx")

# Extract the relevant columns (adjust these column names according to your data structure)
pre_time = pd.to_numeric(force_plate_pre['Time'], errors='coerce')
pre_force_left = pd.to_numeric(force_plate_pre['Z Left'], errors='coerce')
pre_force_right = pd.to_numeric(force_plate_pre['Z Right'], errors='coerce')

post_time = pd.to_numeric(force_plate_post['Time'], errors='coerce')
post_force_left = pd.to_numeric(force_plate_post['Z Left'], errors='coerce')
post_force_right = pd.to_numeric(force_plate_post['Z Right'], errors='coerce')

# Combine the left and right foot forces
pre_total_force = pre_force_left + pre_force_right
post_total_force = post_force_left + post_force_right

# Normalize time to start from 0 seconds
pre_time_normalized = pre_time - pre_time.iloc[0]
post_time_normalized = post_time - post_time.iloc[0]

# Plot the normalized total force waveforms with customizable colors
pre_color = 'blue'  # Change this to your desired color for pre-test
post_color = 'red'  # Change this to your desired color for post-test

plt.figure(figsize=(12, 6))

# Pre-test total force waveform
plt.plot(pre_time_normalized, pre_total_force, label='Pre-test Total Force', color=pre_color)

# Post-test total force waveform
plt.plot(post_time_normalized, post_total_force, label='Post-test Total Force', color=post_color)

plt.xlabel('Time (s)')
plt.ylabel('Total Force (N)')
plt.title('Total Force-Time Comparison (Normalized): Pre-Test vs Post-Test')
plt.legend()
plt.grid(True)
plt.show()



# Find the minimum force value during the eccentric phase (pre-test)
min_force_pre_idx = pre_total_force.idxmin()
min_force_pre_time = pre_time_normalized[min_force_pre_idx]
min_force_pre_value = pre_total_force[min_force_pre_idx]

# Determine the end of the eccentric phase (pre-test)
# Assuming the end of the eccentric phase is where the force begins to increase rapidly
eccentric_end_pre_idx = np.where(np.diff(pre_total_force) > 0)[0][-1]
eccentric_end_pre_time = pre_time_normalized[eccentric_end_pre_idx]
eccentric_end_pre_value = pre_total_force[eccentric_end_pre_idx]

# Find the minimum force value during the eccentric phase (post-test)
min_force_post_idx = post_total_force.idxmin()
min_force_post_time = post_time_normalized[min_force_post_idx]
min_force_post_value = post_total_force[min_force_post_idx]

# Determine the end of the eccentric phase (post-test)
# Assuming the end of the eccentric phase is where the force begins to increase rapidly
eccentric_end_post_idx = np.where(np.diff(post_total_force) > 0)[0][-1]
eccentric_end_post_time = post_time_normalized[eccentric_end_post_idx]
eccentric_end_post_value = post_total_force[eccentric_end_post_idx]

# Display the results
print(f"Pre-Test: Minimum Force: {min_force_pre_value} N at {min_force_pre_time:.2f} s, End of Eccentric Phase: {eccentric_end_pre_value} N at {eccentric_end_pre_time:.2f} s")
print(f"Post-Test: Minimum Force: {min_force_post_value} N at {min_force_post_time:.2f} s, End of Eccentric Phase: {eccentric_end_post_value} N at {eccentric_end_post_time:.2f} s")
