# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 08:19:36 2024

@author: Palan
"""
import pandas as pd
import numpy as np
import os
from scipy.stats import shapiro, wilcoxon, zscore
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
import seaborn as sns
from sklearn.linear_model import LinearRegression


# Define file paths
base_path = r"C:\Users\Palan\OneDrive\Documents\School\McMaster\1.Master's Thesis\Thesis_Data_Analysis"
imu_data_raw = os.path.join(base_path, "I.M.U data", "StepSessionSummaryExport")
force_plate_path = os.path.join(base_path, "raw_data", "forcedecks-test-export-12_07_2024.xlsx")
srpe_path = os.path.join(base_path, "SRPE", "McMaster MBB 2023 - Sessional RPE (Study Fall-2023) (Responses) - Form Responses 1.xlsx")
output_path = os.path.join(base_path, "raw_data")

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

# Load Force Plate and SRPE data
force_plate = pd.read_excel(force_plate_path)
srpe = pd.read_excel(srpe_path)

# Combine "First Name" and "Last Name" in IMU data
imu_data['Name'] = imu_data['First Name'] + ' ' + imu_data['Last Name']
imu_data.rename(columns={'Date (YYYY-MM-DD)': 'Date'}, inplace=True)

# Replace specific names
name_replacements = {'Kazim Raza1': 'Kazim Raza', 'Brendan Amoyaw1': 'Brendan Amoyaw', 'Moody Mohamud': 'Moody Muhammed', 'Thomas Matsell': 'Thomas Mattsel'}
imu_data['Name'].replace(name_replacements, inplace=True)
srpe['Name'].replace(name_replacements, inplace=True)

# Select relevant columns
imu_data = imu_data[['Name', 'Date', 'Footnote', 'Impact Load Total (L+R)', 'Average Intensity (L and R)']]
force_plate = force_plate[['Name', 'Date', 'Tags', 'Jump Height (Imp-Mom) in Inches [in] ',
                           'RSI-modified (Imp-Mom) [m/s] ', 'Eccentric Braking RFD / BM [N/s/kg] ',
                           'Eccentric Braking Impulse [N s] ', 'Force at Zero Velocity / BM [N/kg] ',
                           'Concentric Impulse (Abs) / BM [N s] ', 'Concentric Impulse [N s] ',
                           'Concentric RFD / BM [N/s/kg] ']]
srpe = srpe[['Name', 'Date', 'SRPE']]

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

# Prepare long format dataset
long_format_data = pd.melt(pre_post_data, id_vars=['PlayerID', 'Date', 'Position', 'Name', 'Impact Load Total (L+R)', 'Average Intensity (L and R)'],
                           value_vars=metric_columns + [f'{metric}_Post-Practice' for metric in metric_columns if f'{metric}_Post-Practice' in pre_post_data.columns],
                           var_name='Metric', value_name='Value')

# Extract pre/post information
long_format_data['Time'] = np.where(long_format_data['Metric'].str.endswith('_Post-Practice'), 'Post', 'Pre')
long_format_data['Metric'] = long_format_data['Metric'].str.replace('_Post-Practice', '')

# Ensure 'Time' column is ordered correctly
long_format_data['Time'] = pd.Categorical(long_format_data['Time'], categories=['Pre', 'Post'], ordered=True)

# Drop unnecessary columns and reorder
final_long_format_data = long_format_data[['PlayerID', 'Date', 'Position', 'Metric', 'Value', 'Time','Impact Load Total (L+R)', 'Average Intensity (L and R)']]

# Check the structure of the final long format dataset
print(final_long_format_data.head())

# Save the final dataset for further analysis
final_long_format_data.to_csv(os.path.join(output_path, 'long_format_data.csv'), index=False)

# Function to calculate modified z-scores
def modified_zscore(series):
    median_y = np.median(series)
    median_absolute_deviation_y = np.median(np.abs(series - median_y))
    modified_z_scores = 0.6745 * (series - median_y) / median_absolute_deviation_y
    return modified_z_scores

# Statistical outlier detection and removal using Modified Z-Score
def remove_outliers_modified_zscore(df, value_vars):
    outliers_list = []
    for var in value_vars:
        for time in ['Pre', 'Post']:
            subset = df.loc[(df['Metric'] == var) & (df['Time'] == time)]
            subset.loc[:, 'Modified_Z-Score'] = modified_zscore(subset['Value'])
            
            # Identify outliers
            outliers = subset.loc[subset['Modified_Z-Score'].abs() > 3.5]
            outliers_list.append(outliers)
            
            # Remove outliers
            df = df[~((df['Metric'] == var) & (df['Time'] == time) & (df['Value'].isin(outliers['Value'])))]
    
    # Concatenate all outliers into a single DataFrame if any outliers found
    if outliers_list:
        outliers_df = pd.concat(outliers_list, ignore_index=True)
    else:
        outliers_df = pd.DataFrame()
    return df, outliers_df

# Apply the outlier removal function to the dataset
value_vars = list(set(final_long_format_data['Metric']))
final_long_format_data, outliers_df = remove_outliers_modified_zscore(final_long_format_data, value_vars)


# Save the cleaned dataset for further analysis
#final_long_format_data.to_csv(os.path.join(output_path, 'cleaned_long_format_data.csv'), index=False)
# Save the outliers for reference
#outliers_df.to_csv(os.path.join(output_path, 'outliers_data.csv'), index=False)

# Function to compute descriptive statistics
def compute_descriptive_stats(df, group_by_vars, value_var):
    descriptive_stats = df.groupby(group_by_vars)[value_var].describe().unstack()
    return descriptive_stats

# Compute descriptive statistics for the full dataset
group_by_vars = ['Metric', 'Time']
value_var = 'Value'
descriptive_stats_full = compute_descriptive_stats(final_long_format_data, group_by_vars, value_var)

# Compute descriptive statistics by position
group_by_vars_position = ['Position', 'Metric', 'Time']
descriptive_stats_by_position = compute_descriptive_stats(final_long_format_data, group_by_vars_position, value_var)

# Compute the mean Impact Load and Average Intensity for each practice session
practice_means = final_long_format_data.groupby(['Date', 'Position'])[['Impact Load Total (L+R)', 'Average Intensity (L and R)']].mean().reset_index()

# Calculate z-scores for each practice session
practice_means['Impact Load Z-Score'] = zscore(practice_means['Impact Load Total (L+R)'])
practice_means['Average Intensity Z-Score'] = zscore(practice_means['Average Intensity (L and R)'])

# Determine high, medium, and low zones for practice volume
def categorize_z_score(z_score):
    if z_score >= 1:
        return 'High'
    elif z_score <= -1:
        return 'Low'
    else:
        return 'Medium'

practice_means['Impact Load Zone'] = practice_means['Impact Load Z-Score'].apply(categorize_z_score)
practice_means['Average Intensity Zone'] = practice_means['Average Intensity Z-Score'].apply(categorize_z_score)

# Plot the data with improved clarity
plt.figure(figsize=(14, 8))

# Impact Load Z-Score over time
plt.subplot(2, 1, 1)
for position in practice_means['Position'].unique():
    subset = practice_means[practice_means['Position'] == position]
    plt.plot(subset['Date'], subset['Impact Load Z-Score'], marker='o', linestyle='-', label=position)
plt.axhline(y=1, color='r', linestyle='--', label='High Zone Threshold')
plt.axhline(y=-1, color='b', linestyle='--', label='Low Zone Threshold')
plt.title('Impact Load Z-Score Over Time')
plt.xlabel('Date')
plt.ylabel('Impact Load Z-Score')
plt.legend()

# Average Intensity Z-Score over time
plt.subplot(2, 1, 2)
for position in practice_means['Position'].unique():
    subset = practice_means[practice_means['Position'] == position]
    plt.plot(subset['Date'], subset['Average Intensity Z-Score'], marker='o', linestyle='-', label=position)
plt.axhline(y=1, color='r', linestyle='--', label='High Zone Threshold')
plt.axhline(y=-1, color='b', linestyle='--', label='Low Zone Threshold')
plt.title('Average Intensity Z-Score Over Time')
plt.xlabel('Date')
plt.ylabel('Average Intensity Z-Score')
plt.legend()

plt.tight_layout()
plt.show()


# Function to compute Shapiro-Wilk test results
def compute_shapiro_wilk(df, group_by_vars, value_var):
    shapiro_results = []
    for name, group in df.groupby(group_by_vars):
        stat, p_value = shapiro(group[value_var])
        normality = 'Normally Distributed' if p_value > 0.05 else 'Not Normally Distributed'
        shapiro_results.append((*name, stat, p_value, normality))
    shapiro_df = pd.DataFrame(shapiro_results, columns=group_by_vars + ['Shapiro-Wilk Statistic', 'p-value', 'Normality'])
    return shapiro_df

# Compute Shapiro-Wilk test results for the full dataset
shapiro_wilk_full = compute_shapiro_wilk(final_long_format_data, group_by_vars, value_var)

# Compute Shapiro-Wilk test results by position
shapiro_wilk_by_position = compute_shapiro_wilk(final_long_format_data, group_by_vars_position, value_var)

# Define force plate metrics for Wilcoxon test
force_plate_metrics = ['Jump Height (Imp-Mom) in Inches [in] ', 'RSI-modified (Imp-Mom) [m/s] ', 'Eccentric Braking RFD / BM [N/s/kg] ',
                       'Eccentric Braking Impulse [N s] ', 'Force at Zero Velocity / BM [N/kg] ', 'Concentric Impulse (Abs) / BM [N s] ',
                       'Concentric Impulse [N s] ', 'Concentric RFD / BM [N/s/kg] ']

# Function to compute Wilcoxon signed-rank test results
def compute_wilcoxon(df, metrics):
    wilcoxon_results = []
    for metric in metrics:
        pre_values = df[f'{metric}']
        post_values = df[f'{metric}_Post-Practice']
        if len(pre_values) == len(post_values) and len(pre_values) > 0:
            differences = post_values - pre_values
            if not all(differences == 0):
                stat, p_value = wilcoxon(pre_values, post_values)
                significance = 'Yes' if p_value < 0.05 else 'No'
                wilcoxon_results.append((metric, stat, p_value, significance))
            else:
                wilcoxon_results.append((metric, np.nan, np.nan, 'No'))
        else:
            wilcoxon_results.append((metric, np.nan, np.nan, 'No'))
    wilcoxon_df = pd.DataFrame(wilcoxon_results, columns=['Metric', 'Wilcoxon Statistic', 'p-value', 'Significance'])
    return wilcoxon_df

# Compute Wilcoxon signed-rank test results for the entire dataset
wilcoxon_results_full = compute_wilcoxon(pre_post_data, force_plate_metrics)

# Compute Wilcoxon signed-rank test results by position
wilcoxon_results_by_position = []
for position in pre_post_data['Position'].unique():
    df_position = pre_post_data[pre_post_data['Position'] == position]
    wilcoxon_results_position = compute_wilcoxon(df_position, force_plate_metrics)
    wilcoxon_results_position['Position'] = position
    wilcoxon_results_by_position.append(wilcoxon_results_position)

wilcoxon_results_by_position = pd.concat(wilcoxon_results_by_position, ignore_index=True)

# Function to fit and plot mixed-effects model for each metric
def fit_and_plot_mixed_model(df, metric):
    df_metric = df[df['Metric'] == metric]
    
    # Check if there is enough variability
    if df_metric['Value'].std() == 0:
        print(f"No variability in data for {metric}. Skipping...")
        return None
    
    # Fit the model
    try:
        model = mixedlm("Value ~ Time", df_metric, groups=df_metric["PlayerID"], re_formula="~Time")
        result = model.fit()
        
        print(f"Mixed Linear Model Regression Results for {metric}")
        print(result.summary())
        
        # Extract fixed effects and random effects
        fixed_effects = result.fe_params
        random_effects = result.random_effects
        
       # Plotting
        plt.figure(figsize=(10, 6))
        for player_id, random_effect in random_effects.items():
            intercept = random_effect['Group']
            slope = random_effect.get('Time[T.Post]', 0)  # Use .get to avoid KeyError
            x = np.array([0, 1])
            y = fixed_effects['Intercept'] + intercept + x * (fixed_effects['Time[T.Post]'] + slope)
            plt.plot(x, y, marker='o')
        
        plt.xlabel('Time')
        plt.ylabel(metric)
        plt.xticks([0, 1], ['Pre', 'Post'])
        plt.title(f'Mixed-Effects Model for {metric}')
        plt.show()
        
        # Plotting the random effects
        intercepts = [re['Group'] for re in random_effects.values()]
        slopes = [re.get('Time[T.Post]', 0) for re in random_effects.values()]
        player_ids = list(random_effects.keys())
        
        plt.figure(figsize=(14, 7))
        
        plt.subplot(1, 2, 1)
        sns.barplot(x=player_ids, y=intercepts)
        plt.xlabel('Player ID')
        plt.ylabel('Random Intercepts')
        plt.title(f'Random Intercepts for Each Player ({metric})')
        plt.xticks(rotation=90)
        
        plt.subplot(1, 2, 2)
        sns.barplot(x=player_ids, y=slopes)
        plt.xlabel('Player ID')
        plt.ylabel('Random Slopes')
        plt.title(f'Random Slopes for Each Player ({metric})')
        plt.xticks(rotation=90)
        
        plt.tight_layout()
        plt.show()
        
     
    
    except Exception as e:
        print(f"An error occurred while fitting the model for {metric}: {e}")
        return None
    
# Initialize a dictionary to store results for each metric
mixed_model_results = {}

# Fit and plot the model for each metric
for metric in force_plate_metrics:
    result = fit_and_plot_mixed_model(final_long_format_data, metric)
    if result is not None:
        mixed_model_results[metric] = result

# Function to fit and plot mixed-effects model for each metric by position
def fit_and_plot_mixed_model_by_position(df, metric):
    results_by_position = {}
    
    for position in df['Position'].unique():
        df_position = df[df['Position'] == position]
        df_metric = df_position[df_position['Metric'] == metric]
        
        # Check if there is enough variability
        if df_metric['Value'].std() == 0:
            print(f"No variability in data for {metric} for position {position}. Skipping...")
            continue
        
        # Fit the model
        try:
            model = mixedlm("Value ~ Time", df_metric, groups=df_metric["PlayerID"], re_formula="~Time")
            result = model.fit()
            
            print(f"Mixed Linear Model Regression Results for {metric} - Position: {position}")
            print(result.summary())
            
            # Extract fixed effects and random effects
            fixed_effects = result.fe_params
            random_effects = result.random_effects
            
            # Plotting
            plt.figure(figsize=(10, 6))
            for player_id, random_effect in random_effects.items():
                intercept = random_effect['Group']
                slope = random_effect.get('Time[T.Post]', 0)  # Use .get to avoid KeyError
                x = np.array([0, 1])
                y = fixed_effects['Intercept'] + intercept + x * (fixed_effects['Time[T.Post]'] + slope)
                plt.plot(x, y, marker='o')
            
            plt.xlabel('Time')
            plt.ylabel(metric)
            plt.xticks([0, 1], ['Pre', 'Post'])
            plt.title(f'Mixed-Effects Model for {metric} - Position: {position}')
            plt.show()
            
            # Plotting the random effects
            intercepts = [re['Group'] for re in random_effects.values()]
            slopes = [re.get('Time[T.Post]', 0) for re in random_effects.values()]
            player_ids = list(random_effects.keys())
            
            plt.figure(figsize=(14, 7))
            
            plt.subplot(1, 2, 1)
            sns.barplot(x=player_ids, y=intercepts)
            plt.xlabel('Player ID')
            plt.ylabel('Random Intercepts')
            plt.title(f'Random Intercepts for Each Player ({metric}) - Position: {position}')
            plt.xticks(rotation=90)
            
            plt.subplot(1, 2, 2)
            sns.barplot(x=player_ids, y=slopes)
            plt.xlabel('Player ID')
            plt.ylabel('Random Slopes')
            plt.title(f'Random Slopes for Each Player ({metric}) - Position: {position}')
            plt.xticks(rotation=90)
            
            plt.tight_layout()
            plt.show()
            
            results_by_position[position] = result
        
        except Exception as e:
            print(f"An error occurred while fitting the model for {metric} - Position: {position}: {e}")
            continue
    
    return results_by_position

# Initialize a dictionary to store results for each metric by position
mixed_model_results_by_position = {}

# Fit and plot the model for each metric by position
for metric in force_plate_metrics:
    result_by_position = fit_and_plot_mixed_model_by_position(final_long_format_data, metric)
    if result_by_position:
        mixed_model_results_by_position[metric] = result_by_position

# Calculate drop-off for each metric
drop_off_data = pre_post_data.copy()

for metric in metric_columns:
    drop_off_data[f'{metric}_DropOff'] = drop_off_data[f'{metric}'] - drop_off_data[f'{metric}_Post-Practice']

# Calculate the average Impact Load and Average Intensity for each player
drop_off_data['Avg_Impact_Load'] = drop_off_data.groupby('PlayerID')['Impact Load Total (L+R)'].transform('mean')
drop_off_data['Avg_Average_Intensity'] = drop_off_data.groupby('PlayerID')['Average Intensity (L and R)'].transform('mean')

# Select relevant columns for drop-off data
drop_off_data = drop_off_data[['Name', 'Date', 'Position', 'PlayerID', 'Impact Load Total (L+R)', 'Average Intensity (L and R)', 'Avg_Impact_Load', 'Avg_Average_Intensity'] +
                              [f'{metric}_DropOff' for metric in metric_columns]]

# Calculate the change in each metric from pre to post practice
for metric in metric_columns:
    pre_post_data[f'{metric}_Change'] = pre_post_data[f'{metric}_Post-Practice'] - pre_post_data[f'{metric}']

# Calculate the average change for each player
avg_change_data = pre_post_data.groupby('PlayerID').agg({f'{metric}_Change': 'mean' for metric in metric_columns}).reset_index()

# Merge with the player info and average practice volume
avg_change_data = avg_change_data.merge(pre_post_data[['PlayerID', 'Name', 'Position']].drop_duplicates(), on='PlayerID')
avg_change_data = avg_change_data.merge(drop_off_data[['PlayerID', 'Avg_Impact_Load', 'Avg_Average_Intensity']].drop_duplicates(), on='PlayerID')

# Reshape data to long format
long_format_change_data = pd.melt(avg_change_data, 
                                  id_vars=['PlayerID', 'Name', 'Position', 'Avg_Impact_Load', 'Avg_Average_Intensity'],
                                  value_vars=[f'{metric}_Change' for metric in metric_columns],
                                  var_name='Metric', value_name='Change')

# Extract metric names without '_Change' suffix
long_format_change_data['Metric'] = long_format_change_data['Metric'].str.replace('_Change', '')

# Function to fit and summarize a mixed-effects model
def fit_mixed_model(df, y_var, fixed_effects, group_var):
    model = mixedlm(f"{y_var} ~ {' + '.join(fixed_effects)}", df, groups=df[group_var])
    result = model.fit()
    return result, result.summary()

# Function to create predictions based on the model result
def create_predictions(df, fixed_effect, fixed_effect_name, model_result):
    intercept = model_result.params['Intercept']
    slope = model_result.params[fixed_effect]
    df['Predicted'] = intercept + slope * df[fixed_effect_name]
    return df

# Function to plot scatter plots with regression lines for given metric
def plot_metric_changes(metric, predictions_load, predictions_intensity):
    # Scatter plot with regression line for Impact Load vs Metric Change
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='Avg_Impact_Load', y='Change', data=predictions_load, hue='PlayerID', palette='viridis', s=100, alpha=0.6)
    sns.lineplot(x='Avg_Impact_Load', y='Predicted', data=predictions_load, color='red', linewidth=2)

    # Fit linear regression model to display equation and R-squared value
    X = predictions_load['Avg_Impact_Load'].values.reshape(-1, 1)
    y = predictions_load['Change'].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    r_squared = model.score(X, y)

    # Plot confidence interval band
    ci = 1.96 * np.std(y - y_pred) / np.mean(y)
    plt.fill_between(predictions_load['Avg_Impact_Load'], y_pred - ci, y_pred + ci, color='red', alpha=0.1)

    # Annotate with regression equation and R-squared
    plt.annotate(f'y = {model.coef_[0]:.6f}x + {model.intercept_:.6f}\nR² = {r_squared:.3f}', 
                 xy=(0.05, 0.90), xycoords='axes fraction', fontsize=12, color='red',
                 bbox=dict(facecolor='white', alpha=0.5))
    # Scale the x-axis to the specific values of Avg_Impact_Load
    plt.xlim(predictions_load['Avg_Impact_Load'].min(), predictions_load['Avg_Impact_Load'].max())

    plt.title(f'Impact Load vs {metric.strip()} Change')
    plt.xlabel('Average Impact Load')
    plt.ylabel(f'Change in {metric.strip()}')
    plt.legend().remove()
    plt.show()
    
    # Scatter plot with regression line for Average Intensity vs Metric Change
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='Avg_Average_Intensity', y='Change', data=predictions_intensity, hue='PlayerID', palette='viridis', s=100, alpha=0.6)
    sns.lineplot(x='Avg_Average_Intensity', y='Predicted', data=predictions_intensity, color='red', linewidth=2)

    # Fit linear regression model to display equation and R-squared value
    X = predictions_intensity['Avg_Average_Intensity'].values.reshape(-1, 1)
    y = predictions_intensity['Change'].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    r_squared = model.score(X, y)

    # Plot confidence interval band
    ci = 1.96 * np.std(y - y_pred) / np.mean(y)
    plt.fill_between(predictions_intensity['Avg_Average_Intensity'], y_pred - ci, y_pred + ci, color='red', alpha=0.1)

    # Annotate with regression equation and R-squared
    plt.annotate(f'y = {model.coef_[0]:.6f}x + {model.intercept_:.6f}\nR² = {r_squared:.3f}', 
                 xy=(0.05, 0.90), xycoords='axes fraction', fontsize=12, color='red',
                 bbox=dict(facecolor='white', alpha=0.5))
    
    # Scale the x-axis to the specific values of Avg_Average_Intensity
    plt.xlim(predictions_intensity['Avg_Average_Intensity'].min(), predictions_intensity['Avg_Average_Intensity'].max())

    plt.title(f'Average Intensity vs {metric.strip()} Change')
    plt.xlabel('Average Intensity')
    plt.ylabel(f'Change in {metric.strip()}')
    plt.legend().remove()
    plt.show()

# Initialize dictionaries to store results for each metric
impact_load_results = {}
average_intensity_results = {}

# Perform the analysis for each metric
for metric in force_plate_metrics:
    y_var = f'{metric}_Change'
    
    # Analysis with Avg_Impact_Load
    result_load, summary_load = fit_mixed_model(long_format_change_data[long_format_change_data['Metric'] == metric], 'Change', ['Avg_Impact_Load'], 'PlayerID')
    impact_load_results[metric] = result_load
    
    # Analysis with Avg_Average_Intensity
    result_intensity, summary_intensity = fit_mixed_model(long_format_change_data[long_format_change_data['Metric'] == metric], 'Change', ['Avg_Average_Intensity'], 'PlayerID')
    average_intensity_results[metric] = result_intensity

# Function to sanitize file names
def sanitize_filename(filename):
    return "".join([c for c in filename if c.isalnum() or c in (' ', '.', '_')]).rstrip()

# Save summaries to text files with sanitized filenames
def save_model_summaries(results, analysis_type, output_path):
    with open(os.path.join(output_path, f"{analysis_type}_Summary.txt"), "w") as f:
        for metric, result in results.items():
            f.write(f"Metric: {metric}\n")
            f.write(result.summary().as_text())
            f.write("\n\n")

# Save summaries for each type of analysis
save_model_summaries(mixed_model_results, "position", output_path)
save_model_summaries(impact_load_results, "Impact_Load", output_path)
save_model_summaries(average_intensity_results, "Average_Intensity", output_path)

# Iterate through each metric and generate the plots
for metric in force_plate_metrics:
    # Create predictions for Impact Load
    predictions_load = create_predictions(
        long_format_change_data[long_format_change_data['Metric'] == metric],
        'Avg_Impact_Load', 'Avg_Impact_Load', impact_load_results[metric]
    )
    
    # Create predictions for Average Intensity
    predictions_intensity = create_predictions(
        long_format_change_data[long_format_change_data['Metric'] == metric],
        'Avg_Average_Intensity', 'Avg_Average_Intensity', average_intensity_results[metric]
    )
    
    # Plot the metric changes
    plot_metric_changes(metric, predictions_load, predictions_intensity)

# Save the descriptive statistics and test results to Excel
save_path = os.path.join(output_path, "descriptive_stats_and_tests.xlsx")
with pd.ExcelWriter(save_path) as writer:
    descriptive_stats_full.to_excel(writer, sheet_name='Full Dataset Descriptive Stats')
    descriptive_stats_by_position.to_excel(writer, sheet_name='By Position Descriptive Stats')
    shapiro_wilk_full.to_excel(writer, sheet_name='Full Dataset Shapiro-Wilk')
    shapiro_wilk_by_position.to_excel(writer, sheet_name='By Position Shapiro-Wilk')
    wilcoxon_results_full.to_excel(writer, sheet_name='Wilcoxon Signed-Rank Test Full')
    wilcoxon_results_by_position.to_excel(writer, sheet_name='Wilcoxon Signed-Rank Test Position')
