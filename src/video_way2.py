# NOTE: Only the input/output file paths below were updated (project.xlsx and
# the SPI_* files now live under ../data/raw and ../data/processed, matching
# this repository's folder layout). No other logic was changed from the
# original script. This is the final, consolidated script for Parts 1 and 2
# of the report -- it produced every file currently in data/processed/.

import pandas as pd
import numpy as np
import openpyxl
import scipy.stats as st
import statsmodels.api as sm
import math

def SPI(ds, thresh):
    ds_ma = ds.rolling(thresh, center=False).mean()
    ds_ma[ds_ma <= 0] = np.nan
    ds_In = np.log(ds_ma)
    ds_In[np.isinf(ds_In)] = np.nan
    ds_mu = np.nanmean(ds_ma)
    ds_sum = np.nansum(ds_In)
    n = len(ds_In[thresh-1:])
    A = np.log(ds_mu) - (ds_sum / n)
    alpha = (1 / (4 * A)) * (1 + (1 + ((4 * A) / 3))**0.5)
    beta = ds_mu / alpha
    gamma = st.gamma.cdf(ds_ma, a=alpha, scale=beta)
    norm_spi = st.norm.ppf(gamma, loc=0, scale=1)
    return norm_spi

def generate_all_patterns(pattern_length):
    num_patterns = 2**pattern_length
    all_patterns = []
    for i in range(num_patterns):
        pattern = []
        for j in range(pattern_length):
            pattern.append((i >> (pattern_length - 1 - j)) & 1)
        all_patterns.append(pattern)
    return all_patterns

def count_pattern_occurrences(data, pattern):
    pattern_count = 0
    pattern_with_rain_count = 0
    pattern_length = len(pattern)
    for i in range(len(data) - pattern_length):
        if data[i:i + pattern_length] == pattern:
            pattern_count += 1
            if data[i + pattern_length] == 1:
                pattern_with_rain_count += 1
    return pattern_count, pattern_with_rain_count

def calculate_probabilities(data, pattern_length):
    all_patterns = generate_all_patterns(pattern_length)
    pattern_probabilities = {}
    for pattern in all_patterns:
        pattern_count, rain_count = count_pattern_occurrences(data, pattern)
        total_occurrences = pattern_count
        probability_of_rain = rain_count / total_occurrences if total_occurrences > 0 else 0
        probability_of_no_rain = 1 - probability_of_rain
        pattern_probabilities[tuple(pattern)] = (probability_of_no_rain, probability_of_rain)
    return pattern_probabilities

def calculate_log_likelihood(pattern_probabilities, data, pattern_length):
    log_likelihood = 0
    for pattern, (no_rain_prob, rain_prob) in pattern_probabilities.items():
        pattern_count, rain_count = count_pattern_occurrences(data, list(pattern))
        no_rain_count = pattern_count - rain_count

        if no_rain_prob > 0 and no_rain_count > 0:
            log_likelihood += no_rain_count * np.log(no_rain_prob)
        if rain_prob > 0 and rain_count > 0:
            log_likelihood += rain_count * np.log(rain_prob)

    return log_likelihood

def calculate_transition_matrix(data, pattern_length, label):
    pattern_probabilities = calculate_probabilities(data, pattern_length)
    patterns = generate_all_patterns(pattern_length)
    transition_matrix = []
    for pattern in patterns:
        no_rain_prob, rain_prob = pattern_probabilities[tuple(pattern)]
        transition_matrix.append([no_rain_prob, rain_prob])
    transition_df = pd.DataFrame(transition_matrix, index=[tuple(p) for p in patterns], columns=['P(0)', 'P(1)'])

    # Calculate AIC and BIC
    n = len(data)
    k = (2**pattern_length) * (2**pattern_length) - (2**pattern_length)
    log_likelihood = calculate_log_likelihood(pattern_probabilities, data, pattern_length)
    aic = 2 * k - 2 * log_likelihood
    bic = np.log(n) * k - 2 * log_likelihood

    print(f"\n{label} Transition Matrix for Pattern Length {pattern_length}:")
    print(transition_df)
    print(f"AIC: {aic:.2f}")
    print(f"BIC: {bic:.2f}")

    return transition_df, aic, bic

# Load data and prepare dataframes (unchanged)
file_path = '../data/raw/project.xlsx'
workbook = openpyxl.load_workbook(file_path)
sheet = workbook.active

data = []
for row_number in range(2, sheet.max_row + 1):
    date_cell = sheet.cell(row=row_number, column=1).value
    rain_cell = sheet.cell(row=row_number, column=6).value
    if date_cell and rain_cell:
        date = pd.to_datetime(date_cell)
        rain_value = float(rain_cell.replace(' mm', ''))
        data.append({'Date': date, 'Rain': rain_value})

df = pd.DataFrame(data)
df.set_index('Date', inplace=True)

yearly_mean_rain = df.resample('Y').mean()
spi_values_yearly = SPI(yearly_mean_rain['Rain'], 1)
spi_df_yearly = pd.DataFrame(spi_values_yearly, index=yearly_mean_rain.index, columns=['SPI'])
spi_df_yearly['SPI_binary'] = np.where(spi_df_yearly['SPI'] <= 0, 0, 1)
SPI_binary_yearly = spi_df_yearly['SPI_binary'].tolist()

monthly_mean_rain = df.resample('M').mean()
spi_values_monthly = SPI(monthly_mean_rain['Rain'], 1)
spi_df_monthly = pd.DataFrame(spi_values_monthly, index=monthly_mean_rain.index, columns=['SPI'])
spi_df_monthly['SPI_binary'] = np.where(spi_df_monthly['SPI'] <= 0, 0, 1)
SPI_binary_monthly = spi_df_monthly['SPI_binary'].tolist()
spi_df_monthly['Mean_Rainfall'] = monthly_mean_rain['Rain'].values

# Calculate and print transition matrices with AIC and BIC
for pattern_length in [1, 2, 3, 4, 5]:
    _, aic_yearly, bic_yearly = calculate_transition_matrix(SPI_binary_yearly, pattern_length, 'yearly')
    _, aic_monthly, bic_monthly = calculate_transition_matrix(SPI_binary_monthly, pattern_length, 'monthly')

spi_df_yearly.reset_index(inplace=True)
spi_df_yearly['Date'] = spi_df_yearly['Date'].dt.strftime('%m/%d/%Y')

spi_df_monthly.reset_index(inplace=True)
spi_df_monthly['Date'] = spi_df_monthly['Date'].dt.strftime('%m/%d/%Y')

# Define the categorization function
def categorize_mean_rainfall(mean_rainfall):
    if 0 <= mean_rainfall < 1:
        return 0
    elif 1 <= mean_rainfall < 2:
        return 1
    elif 2 <= mean_rainfall < 3:
        return 2
    elif 3 <= mean_rainfall < 4:
        return 3
    elif 4 <= mean_rainfall < 5:
        return 4
    elif 5 <= mean_rainfall < 6:
        return 5
    elif 6 <= mean_rainfall < 7:
        return 6
    elif 7 <= mean_rainfall < 8:
        return 7
    elif 8 <= mean_rainfall < 9:
        return 8
    elif 9 <= mean_rainfall < 10:
        return 9
    else:
        return np.nan

spi_df_yearly['Mean_Rainfall'] = yearly_mean_rain['Rain'].values
spi_df_yearly['Mean_Rainfall_category'] = spi_df_yearly['Mean_Rainfall'].apply(categorize_mean_rainfall)

with pd.ExcelWriter('../data/processed/SPI_data_yearly_with_categories.xlsx', date_format='mm/dd/yyyy') as writer:
    spi_df_yearly.to_excel(writer, sheet_name='Yearly', index=False)

with pd.ExcelWriter('../data/processed/SPI_data_monthly_with_categories.xlsx', date_format='mm/dd/yyyy') as writer:
    spi_df_monthly.to_excel(writer, sheet_name='Monthly', index=False)

file_path = '../data/processed/SPI_data_monthly_with_categories.xlsx'
df = pd.read_excel(file_path, sheet_name='Monthly')
df['Mean_Rainfall_category'] = df['Mean_Rainfall'].apply(categorize_mean_rainfall)

with pd.ExcelWriter(file_path, date_format='mm/dd/yyyy') as writer:
    df.to_excel(writer, sheet_name='Monthly', index=False)


file_path = '../data/processed/SPI_data_monthly_with_categories.xlsx'
df = pd.read_excel(file_path, sheet_name='Monthly')

# Calculate transition matrix for Mean_Rainfall_category considering one day before

def calculate_transition_matrix_one_day(df, column_name):
    unique_categories = sorted(df[column_name].dropna().unique())
    num_categories = len(unique_categories)
    transition_matrix = pd.DataFrame(0, index=unique_categories, columns=unique_categories, dtype=float)

    for i in range(1, len(df)):
        prev_category = df.at[i - 1, column_name]
        current_category = df.at[i, column_name]
        if not pd.isna(prev_category) and not pd.isna(current_category):
            transition_matrix.at[prev_category, current_category] += 1

    transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0).fillna(0)
    return transition_matrix

# Calculate transition matrix for Mean_Rainfall_category considering two days before
def calculate_transition_matrix_two_days(df, column_name):
    unique_categories = sorted(df[column_name].dropna().unique())
    num_categories = len(unique_categories)
    transition_matrix = pd.DataFrame(0, index=pd.MultiIndex.from_product([unique_categories, unique_categories]), columns=unique_categories, dtype=float)

    for i in range(2, len(df)):
        prev_category_2 = df.at[i - 2, column_name]
        prev_category_1 = df.at[i - 1, column_name]
        current_category = df.at[i, column_name]
        if not pd.isna(prev_category_2) and not pd.isna(prev_category_1) and not pd.isna(current_category):
            transition_matrix.at[(prev_category_2, prev_category_1), current_category] += 1

    transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0).fillna(0)

    # Calculate AIC and BIC
    n = len(df) - 2
    k = num_categories**3 - num_categories**2
    log_likelihood = calculate_log_likelihood_transition_matrix_two_days(transition_matrix, df, column_name)
    aic = 2 * k - 2 * log_likelihood
    bic = np.log(n) * k - 2 * log_likelihood

    return transition_matrix, aic, bic

# Calculate log-likelihood for the two-day transition matrix
def calculate_log_likelihood_transition_matrix_two_days(transition_matrix, df, column_name):
    log_likelihood = 0
    for i in range(2, len(df)):
        prev_category_2 = df.at[i - 2, column_name]
        prev_category_1 = df.at[i - 1, column_name]
        current_category = df.at[i, column_name]
        if not pd.isna(prev_category_2) and not pd.isna(prev_category_1) and not pd.isna(current_category):
            prob = transition_matrix.at[(prev_category_2, prev_category_1), current_category]
            if prob > 0:
                log_likelihood += np.log(prob)
    return log_likelihood

# Calculate the transition matrices
transition_matrix_one_day = calculate_transition_matrix_one_day(df, 'Mean_Rainfall_category')
transition_matrix_two_days, aic_two_days, bic_two_days = calculate_transition_matrix_two_days(df, 'Mean_Rainfall_category')

# Print the transition matrices and AIC, BIC values
def calculate_log_likelihood_transition_matrix(transition_matrix, df, column_name):
    log_likelihood = 0
    for i in range(1, len(df)):
        prev_category = df.at[i - 1, column_name]
        current_category = df.at[i, column_name]
        if not pd.isna(prev_category) and not pd.isna(current_category):
            prob = transition_matrix.at[prev_category, current_category]
            if prob > 0:
                log_likelihood += np.log(prob)
    return log_likelihood

def calculate_transition_matrix_one_day(df, column_name):
    unique_categories = sorted(df[column_name].dropna().unique())
    num_categories = len(unique_categories)
    transition_matrix = pd.DataFrame(0, index=unique_categories, columns=unique_categories, dtype=float)

    for i in range(1, len(df)):
        prev_category = df.at[i - 1, column_name]
        current_category = df.at[i, column_name]
        if not pd.isna(prev_category) and not pd.isna(current_category):
            transition_matrix.at[prev_category, current_category] += 1

    transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0).fillna(0)

    n = len(df) - 1
    k = num_categories * (num_categories - 1)
    log_likelihood = calculate_log_likelihood_transition_matrix(transition_matrix, df, column_name)
    aic = 2 * k - 2 * log_likelihood
    bic = np.log(n) * k - 2 * log_likelihood

    return transition_matrix, aic, bic

# Calculate the transition matrix and print AIC and BIC
transition_matrix_one_day, aic_one_day, bic_one_day = calculate_transition_matrix_one_day(df, 'Mean_Rainfall_category')
print("Transition Matrix for Mean_Rainfall_category (one day):")
print(transition_matrix_one_day)
print(f"AIC: {aic_one_day:.2f}")
print(f"BIC: {bic_one_day:.2f}")

print("\nTransition Matrix for Mean_Rainfall_category (two days):")
print(transition_matrix_two_days)
print(f"AIC: {aic_two_days:.2f}")
print(f"BIC: {bic_two_days:.2f}")

# Save the transition matrices to CSV files
#transition_matrix_one_day.to_csv('transition_matrix_Mean_Rainfall_category_one_day.csv')
#transition_matrix_two_days.to_csv('transition_matrix_Mean_Rainfall_category_two_days.csv')

# Optionally, save the transition matrices to the same Excel file
with pd.ExcelWriter(file_path, date_format='mm/dd/yyyy', mode='a') as writer:
    transition_matrix_one_day.to_excel(writer, sheet_name='Transition_Matrix_One_Day')
    transition_matrix_two_days.to_excel(writer, sheet_name='Transition_Matrix_Two_Days')
