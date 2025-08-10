import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from math import pi  # For radar chart

# --- For Gemini API Integration ---
import google.generativeai as genai
import textwrap  # For formatting the output nicely

# --- Configuration ---
# IMPORTANT: Place your data files (e.g., CSV, Excel) in a subfolder named 'data'
# within your PyCharm project directory for easier management.
DATA_FOLDER = 'data'
YOUR_ACCOUNT_NAME = 'Management Science'  # Updated to match new filename: 'Management Science.xlsx'

# Configure your Gemini API key
# SECURITY WARNING: For production environments, DO NOT hardcode API keys.
# Instead, load them from environment variables for better security.
# Example: api_key = os.getenv("GEMINI_API_KEY")
# For this demonstration, we use the provided key directly.
API_KEY = (""
           "") # User provided API key, kept as is.
genai.configure(api_key=API_KEY)


# --- Helper Function for Text Formatting (Optional, but good for console output) ---
def to_markdown(text):
    """
    Formats text into a markdown block for better console readability,
    especially for Gemini API responses.
    """
    text = text.replace('•', '  *')  # Replaces bullet points with markdown list item
    return textwrap.indent(text, '> ', predicate=lambda line: True)


# --- 1. Load Individual Account Data ---
def load_account_data(file_path):
    """
    Loads data from a single account file (CSV or Excel), robustly identifying the header row.
    It searches for 'KPI' in the first few rows to determine the actual header,
    then expects numeric values in the column next to 'KPI' (e.g., 'Current').

    Args:
        file_path (str): The full path to the data file.

    Returns:
        pd.DataFrame or None: The loaded and pre-processed DataFrame, or None if an error occurs.
    """
    try:
        # Read the file initially without a header to inspect its content
        if file_path.endswith('.csv'):
            temp_df = pd.read_csv(file_path, header=None)
        elif file_path.endswith(('.xlsx', '.xls')):
            temp_df = pd.read_excel(file_path, header=None)
        else:
            print(f"Unsupported file type for {file_path}. Please use .csv or .xlsx.")
            return None

        header_row_idx = -1
        # Search for 'KPI' (case-insensitive) in the first few rows to find the actual header
        for i in range(min(temp_df.shape[0], 10)): # Check up to the first 10 rows
            row_values = temp_df.iloc[i].astype(str).str.strip().str.lower()
            if 'kpi' in row_values.values:
                header_row_idx = i
                break

        if header_row_idx == -1:
            print(f"Error: Could not find 'KPI' in the first 10 rows of {os.path.basename(file_path)}. Please ensure 'KPI' is a header.")
            return None

        # Now, load the file again, using the identified header row
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, header=header_row_idx)
        else: # .xlsx or .xls
            df = pd.read_excel(file_path, header=header_row_idx)

        # Clean column names by stripping whitespace
        df.columns = [str(col).strip() for col in df.columns]

        # Explicitly check for 'KPI' and 'Current' columns as per user's description and image
        # Rename 'KPI' column to standard 'KPI'
        found_kpi_col = None
        for col in df.columns:
            if isinstance(col, str) and 'kpi' in col.lower():
                found_kpi_col = col
                break
        if found_kpi_col:
            if found_kpi_col != 'KPI': # Only rename if necessary
                df.rename(columns={found_kpi_col: 'KPI'}, inplace=True)
        else:
            print(f"Error: 'KPI' column not found after loading with inferred header in {os.path.basename(file_path)}. Available columns: {df.columns.tolist()}")
            return None

        # Ensure 'Current' column exists and rename it to a generic 'Value' for consistent processing
        if 'Current' in df.columns:
            df.rename(columns={'Current': 'Value'}, inplace=True)
        else:
            # If 'Current' is not found, try to find the first numeric-looking column after 'KPI'
            value_col_name = None
            kpi_col_idx = df.columns.get_loc('KPI')
            for i in range(kpi_col_idx + 1, df.shape[1]):
                col_name = df.columns[i]
                # Check if the column contains mostly numeric data (more than 70% non-NaN after conversion)
                if pd.to_numeric(df[col_name], errors='coerce').notna().sum() > len(df) * 0.7:
                    value_col_name = col_name
                    break
            if value_col_name:
                df.rename(columns={value_col_name: 'Value'}, inplace=True)
            else:
                print(f"Error: Could not find a suitable 'Value' column (like 'Current') in {os.path.basename(file_path)}. Available columns: {df.columns.tolist()}")
                return None

        # Select only 'KPI' and 'Value' columns and drop rows where 'KPI' is null
        df = df[['KPI', 'Value']].dropna(subset=['KPI']).copy()

        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


# --- 2. Process and Combine Data ---
def combine_all_data(data_folder, your_account_name):
    """
    Combines data from all account files in the specified folder into a single DataFrame.
    Transposes data, infers account names, and converts numeric columns.

    Args:
        data_folder (str): The path to the folder containing data files.
        your_account_name (str): The exact name of your account to identify it in the data.

    Returns:
        pd.DataFrame: A combined DataFrame with all account data, or an empty DataFrame if no files are found.
    """
    all_accounts_data = []

    # Iterate through all files in the data folder
    for filename in os.listdir(data_folder):
        # Process only CSV or Excel files
        if filename.endswith(('.csv', '.xlsx', '.xls')):
            file_path = os.path.join(data_folder, filename)
            df_account = load_account_data(file_path)

            if df_account is not None:
                # Transpose the DataFrame to have KPIs as columns and data points as rows
                # Set 'KPI' column as index before transposing
                df_transposed = df_account.set_index('KPI').T
                df_transposed = df_transposed.reset_index(drop=True) # Reset index after transpose

                # Infer account name from filename (e.g., "Management Science.xlsx" -> "Management Science")
                account_name = os.path.splitext(filename)[0]
                df_transposed['Account'] = account_name # Add an 'Account' column
                all_accounts_data.append(df_transposed)

    if not all_accounts_data:
        print(f"No data files found in '{data_folder}'. Please ensure files are placed there.")
        return pd.DataFrame() # Return empty DataFrame if no data

    # Concatenate all individual account DataFrames into one
    combined_df = pd.concat(all_accounts_data, ignore_index=True)

    # Convert potential numeric columns to appropriate types
    potential_numeric_cols = [col for col in combined_df.columns if col != 'Account']
    for col in potential_numeric_cols:
        current_column = combined_df[col]
        # Ensure current_column is a Series, squeezing if it's a single-column DataFrame
        if isinstance(current_column, pd.DataFrame) and current_column.shape[1] == 1:
            current_column = current_column.squeeze()
        elif isinstance(current_column, pd.DataFrame) and current_column.shape[1] > 1:
            # If it's still a multi-column DataFrame after selection, log and skip
            print(f"Warning: Column '{col}' unexpectedly resulted in a multi-column DataFrame. Skipping numeric conversion for this column.")
            continue


        # Convert to string to handle commas and percentages
        temp_series = current_column.astype(str).str.replace(',', '.', regex=False)

        # Check for and handle percentage values
        if temp_series.str.contains('%').any():
            temp_series = temp_series.str.replace('%', '', regex=False)
            combined_df[col] = pd.to_numeric(temp_series, errors='coerce') / 100
        else:
            # For non-percentage columns, just convert to numeric after comma replacement
            combined_df[col] = pd.to_numeric(temp_series, errors='coerce')


    # Identify your account using the exact specified name (case-insensitive)
    combined_df['Is_Your_Account'] = (combined_df['Account'].str.lower() == your_account_name.lower())

    print("\n--- Combined Data Preview ---")
    print(combined_df.head().to_string())
    print("\n--- Data Types ---")
    print(combined_df.info())

    return combined_df


# --- Function to generate Gemini prompt summary string ---
def generate_gemini_prompt_summary_string(df_combined, your_account_name, feature_importances_df):
    """
    Generates a structured text summary of the analysis findings for Gemini.
    This summary includes underperformance, strengths, and feature importance
    to guide Gemini in generating actionable hypotheses.

    Args:
        df_combined (pd.DataFrame): The combined and processed DataFrame.
        your_account_name (str): The name of your LinkedIn account.
        feature_importances_df (pd.DataFrame): DataFrame containing feature importances from ML model.

    Returns:
        str: A formatted string containing the summary for the Gemini API.
    """
    summary = f"I am analyzing LinkedIn data for my account '{your_account_name}' against direct competitors. "
    summary += "My goal is to suggest a social media strategy to improve its impression. "
    summary += "Please generate specific, actionable hypotheses based on the following analytical findings:\n\n"

    your_account_data = df_combined[df_combined['Is_Your_Account']]
    competitors_data = df_combined[~df_combined['Is_Your_Account']]
    # Filter for numerical KPIs, excluding 'Is_Your_Account' and 'Account' columns
    numerical_kpis = [col for col in df_combined.columns if
                      pd.api.types.is_numeric_dtype(df_combined[col]) and col not in ['Is_Your_Account', 'Account']]

    if your_account_data.empty:
        return summary + "Error: Your account data not found for summary generation.\n"

    # Get your account's KPI values and competitor averages/bests
    your_account_row = your_account_data[numerical_kpis].iloc[0]
    competitors_avg = competitors_data[numerical_kpis].mean()
    competitors_best = competitors_data[numerical_kpis].max()

    # 1. Key Underperformances (from Gap Analysis - still calculated for prompt, but not explicitly printed)
    underperforming_kpis = []
    for kpi in numerical_kpis:
        your_val = your_account_row.get(kpi)
        comp_avg = competitors_avg.get(kpi)

        # Ensure values are scalar for pd.isna check if they somehow became single-element Series
        if isinstance(your_val, pd.Series) and len(your_val) == 1:
            your_val = your_val.item()
        if isinstance(comp_avg, pd.Series) and len(comp_avg) == 1:
            comp_avg = comp_avg.item()

        # Handle cases where values might be NaN (Not a Number) after conversion or missing
        if pd.isna(your_val) or pd.isna(comp_avg): # Removed comp_best from this check, as it's not used in this specific gap calculation loop.
            continue # Skip this KPI if data is missing for comparison

        if your_val != 0:
            gap_vs_avg_percent = ((comp_avg - your_val) / your_val * 100)
            if gap_vs_avg_percent > 10:  # Threshold for significant underperformance
                underperforming_kpis.append((kpi, gap_vs_avg_percent))
        elif comp_avg > 0:  # Your value is 0 but competitor has value (infinite gap)
            underperforming_kpis.append((kpi, float('inf')))

    underperforming_kpis.sort(key=lambda x: x[1], reverse=True) # Sort by largest gap
    summary += "1. **Areas of Significant Underperformance (compared to Competitor Average):**\n"
    if underperforming_kpis:
        for kpi, gap in underperforming_kpis[:5]:  # Top 5 underperforming KPIs
            if gap == float('inf'):
                summary += f"    - {kpi}: Your value is 0, while competitors have an average value.\n"
            else:
                summary += f"    - {kpi}: Needs to increase by ~{gap:.0f}% to reach competitor average.\n"
    else:
        summary += "    - No significant underperformance identified.\n"

    # 2. Key Overperformances (from Ranking)
    overperforming_kpis = []
    if not df_combined.empty:
        df_rank = df_combined[['Account'] + numerical_kpis].copy()
        for kpi in numerical_kpis:
            # Rank based on the actual values, higher is better
            temp_df = df_rank[['Account', kpi]].dropna() # Drop NaNs for accurate ranking
            if not temp_df.empty and len(temp_df) > 0:
                temp_df[f'{kpi}_Rank'] = temp_df[kpi].rank(ascending=False, method='min')  # 1 is best rank
                your_rank_row = temp_df[temp_df['Account'].str.lower() == your_account_name.lower()]
                if not your_rank_row.empty and your_rank_row[f'{kpi}_Rank'].iloc[0] == 1:
                    overperforming_kpis.append(kpi)

    summary += "\n2. **Areas of Strength (Ranking #1 among all accounts):**\n"
    if overperforming_kpis:
        for kpi in overperforming_kpis:
            summary += f"    - {kpi}\n"
    else:
        summary += "    - No clear top-ranked strengths identified.\n"

    # 3. Most Important Drivers (from Feature Importance)
    summary += "\n3. **Most Important KPIs for Predicting Engagement (based on Random Forest Feature Importance):**\n"
    if not feature_importances_df.empty:
        for index, row in feature_importances_df.head(5).iterrows():  # Top 5 important features
            summary += f"    - {row['Feature']} (Importance: {row['Importance']:.3f})\n"
    else:
        summary += "    - Feature importance data not available or not enough data for modeling.\n"

    # 4. Overall Profile (Qualitative observation prompt for the user)
    summary += "\n4. **Overall Performance Profile (from Radar Chart):**\n"
    summary += "    - (Observe your Radar Chart - 'Management Science' profile vs. Competitor Average/Best. "
    summary += "Describe if it's generally smaller, has specific 'dips' or 'peaks' compared to others.)\n"
    summary += "    - For example: 'My account's profile is generally smaller than the competitor average, especially in Avg. Engagement / Day and Total Estimated Reach, but it has a relatively strong 'Followers Growth' rate.'\n"

    return summary


# --- Main execution block ---
if __name__ == "__main__":
    # Create the 'data' folder if it doesn't exist
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(f"Created '{DATA_FOLDER}' folder. Please place your LinkedIn data files inside this folder.")
    else:
        print(f"'{DATA_FOLDER}' folder already exists. Looking for data files inside.")

    # Combine all data from the 'data' folder
    df_combined = combine_all_data(DATA_FOLDER, YOUR_ACCOUNT_NAME)

    if not df_combined.empty:
        # Separate your account's data from competitors' data
        your_account_data = df_combined[df_combined['Is_Your_Account']]
        competitors_data = df_combined[~df_combined['Is_Your_Account']]

        if your_account_data.empty:
            print(f"\nERROR: Your account '{YOUR_ACCOUNT_NAME}' not found in the combined data.")
            print(
                f"Please ensure 'YOUR_ACCOUNT_NAME' in the script EXACTLY matches an 'Account' name from your data (e.g., 'Management Science').")
            print("Detected accounts are:", df_combined['Account'].tolist())
            exit() # Exit if your account data is not found, as further analysis won't be meaningful

        elif competitors_data.empty:
            print("\nWARNING: No competitor data found. Cannot perform comparative analysis effectively.")
            # Continue, but results will be limited to your account's raw data
        else:
            print(f"Your account '{YOUR_ACCOUNT_NAME}' identified: {your_account_data.shape[0]} row(s)")
            print("\n--- Starting Competitor Analysis ---")

            # Identify numerical KPIs for analysis
            numerical_kpis = [col for col in df_combined.columns if
                              pd.api.types.is_numeric_dtype(df_combined[col]) and col not in ['Is_Your_Account',
                                                                                               'Account']]
            # Remove any non-numeric or helper columns that might have slipped through (e.g., 'index' from reset_index)
            numerical_kpis = [k for k in numerical_kpis if k not in ['index']]

            # Filter out columns that are entirely NaN after numeric conversion
            numerical_kpis = [kpi for kpi in numerical_kpis if df_combined[kpi].notna().any()]

            if not numerical_kpis:
                print("No numerical KPIs found for analysis after filtering. Check your data format.")
                exit() # Exit if no numerical KPIs are found to prevent further errors

            # --- Analysis 1: Descriptive Statistics for Comparison ---
            print("\n--- Your Account KPIs ---")
            print(your_account_data[numerical_kpis].T.to_string())

            print("\n--- Competitors' Average KPIs ---")
            competitors_avg = competitors_data[numerical_kpis].mean().to_frame(name='Average Competitor').T
            print(competitors_avg.to_string())

            print("\n--- Competitors' Best (Max) KPIs ---")
            competitors_best = competitors_data[numerical_kpis].max().to_frame(name='Best Competitor').T
            print(competitors_best.to_string())

            # --- Analysis 2: Visual Comparison (Bar Charts) ---
            print("\n--- Generating Visual Comparisons (Bar Charts) ---")
            plt.style.use('seaborn-v0_8-darkgrid') # Set a nice plot style

            # Filter for meaningful KPIs for bar charts (avoiding redundant 'Total' metrics, but including key ones)
            kpis_for_bar_charts = [k for k in numerical_kpis if
                                   'Total' not in k or 'Rate' in k or 'Growth' in k or k in ['Total Posts', 'Total Followers', 'Total Reactions', 'Total Comments', 'Total Shares', 'Total Estimated Reach', 'Total Estimated Impressions']]

            if kpis_for_bar_charts:
                n_rows = len(kpis_for_bar_charts)
                fig, axes = plt.subplots(nrows=n_rows, ncols=1, figsize=(12, 4 * n_rows))
                if n_rows == 1: # Ensure axes is iterable even for a single plot
                    axes = [axes]

                for i, kpi in enumerate(kpis_for_bar_charts):
                    # Added hue='Account' to address FutureWarning and explicitly map colors
                    sns.barplot(x='Account', y=kpi, data=df_combined, ax=axes[i], palette='viridis', hue='Account', dodge=False)
                    # Add a horizontal line for your account's value for easy comparison
                    your_account_value = your_account_data[kpi].iloc[0] if not your_account_data.empty and kpi in your_account_data.columns else None
                    if your_account_value is not None and pd.notna(your_account_value):
                        axes[i].axhline(y=your_account_value, color='red', linestyle='--',
                                         label=f'Your Account ({your_account_value:.2f})')
                    axes[i].set_title(f'Comparison of {kpi}', fontsize=14)
                    axes[i].set_ylabel(kpi, fontsize=12)
                    axes[i].set_xlabel('LinkedIn Account', fontsize=12)
                    # Removed 'ha' as it's not a valid parameter for tick_params for x-axis
                    axes[i].tick_params(axis='x', rotation=45) # Rotate x-axis labels
                    axes[i].legend(fontsize=10)
                plt.tight_layout() # Adjust plot to prevent labels overlapping
                plt.show()
            else:
                print("No suitable KPIs found for bar charts after filtering.")

            # --- Analysis 3: Gap Analysis (Removed detailed print, but summary for Gemini remains) ---
            # The logic to calculate underperforming KPIs for the Gemini prompt is still in
            # generate_gemini_prompt_summary_string, which is a good balance.

            # --- Analysis 4: Radar/Spider Chart for Profile Comparison ---
            print("\n--- Generating Radar Chart for Profile Comparison ---")
            # Define key KPIs for the radar chart, which are generally positive metrics reflecting performance
            # These KPIs were identified from the image provided.
            radar_kpis = [
                'Avg. Engagement (without clicks)',
                'Avg. Engagement / Day (without clicks)',
                'Avg. Eng. Rate by Followers (without clicks)',
                'Avg. Eng. Rate by Reach (without clicks)',
                'Avg. Eng. Rate by Impressions (without clicks)',
                'Followers Growth',
                'Total Reactions',
                'Total Comments',
                'Total Shares',
                'Total Followers',
                'Total Estimated Impressions',
                'Total Estimated Reach'
            ]
            # Filter to include only KPIs actually present in the numerical data
            radar_kpis = [kpi for kpi in numerical_kpis if kpi in radar_kpis]

            if len(radar_kpis) < 2:
                print(
                    "Not enough suitable KPIs for a meaningful radar chart. Please ensure relevant numeric KPIs are available (e.g., engagement rates, growth).")
            else:
                df_radar = df_combined[['Account'] + radar_kpis].set_index('Account')
                df_radar = df_radar.dropna()  # Drop rows with any NaN values to avoid plotting issues

                if df_radar.empty or len(df_radar.index) < 2:
                    print("Not enough complete data for radar chart after handling missing values.")
                else:
                    # Scale the data for the radar chart (Min-Max scaling to 0-1 range)
                    scaler_radar = MinMaxScaler()
                    df_radar_scaled = pd.DataFrame(scaler_radar.fit_transform(df_radar), columns=radar_kpis,
                                                 index=df_radar.index)

                    # Prepare competitor average and best for radar chart, only for the selected radar_kpis
                    comp_avg_for_radar = competitors_data[radar_kpis].mean().to_frame().T
                    comp_best_for_radar = competitors_data[radar_kpis].max().to_frame().T

                    # Add Competitor Average and Best to the scaled DataFrame if they have valid data
                    if not comp_avg_for_radar.isnull().all().all(): # Check if all values in average are NaN
                        scaled_comp_avg = scaler_radar.transform(comp_avg_for_radar)
                        df_radar_scaled.loc['Competitor Average'] = scaled_comp_avg[0] # [0] because transform returns a 2D array
                    if not comp_best_for_radar.isnull().all().all(): # Check if all values in best are NaN
                        scaled_comp_best = scaler_radar.transform(comp_best_for_radar)
                        df_radar_scaled.loc['Competitor Best'] = scaled_comp_best[0]

                    categories = radar_kpis
                    N = len(categories)
                    # Calculate angles for each category to plot on the radar chart
                    angles = [n / float(N) * 2 * pi for n in range(N)]
                    angles += angles[:1] # Complete the loop by adding the first angle again

                    # Create the radar chart
                    fig_radar = plt.figure(figsize=(12, 12))
                    ax = fig_radar.add_subplot(111, polar=True)

                    plt.xticks(angles[:-1], categories, color='grey', size=10) # Set category labels
                    ax.set_rlabel_position(0) # Set position of radial labels
                    plt.yticks([0.25, 0.5, 0.75, 1], ["0.25", "0.50", "0.75", "1.00"], color="grey", size=8) # Set radial tick labels
                    plt.ylim(0, 1) # Set y-axis limits (0 to 1 for scaled data)

                    # Plot each account's data on the radar chart
                    for i, account_name in enumerate(df_radar_scaled.index):
                        values = df_radar_scaled.loc[account_name].values.flatten().tolist()
                        values += values[:1] # Add the first value again to close the circular plot

                        if account_name.lower().strip() == YOUR_ACCOUNT_NAME.lower().strip():
                            # Highlight your account in red
                            ax.plot(angles, values, linewidth=2.5, linestyle='solid', label=account_name, color='red',
                                     marker='o', markersize=6)
                            ax.fill(angles, values, 'red', alpha=0.1)
                        elif account_name == 'Competitor Average':
                            # Plot competitor average in blue (dashed)
                            ax.plot(angles, values, linewidth=1.5, linestyle='dashed', label=account_name, color='blue',
                                     marker='x', markersize=5)
                            ax.fill(angles, values, 'blue', alpha=0.05)
                        elif account_name == 'Competitor Best':
                            # Plot competitor best in green (dashed)
                            ax.plot(angles, values, linewidth=1.5, linestyle='dashed', label=account_name,
                                     color='green', marker='^', markersize=5)
                            ax.fill(angles, values, 'green', alpha=0.05)
                        else: # Plot other competitors in grey
                            ax.plot(angles, values, linewidth=1, linestyle='solid', label=account_name, alpha=0.7,
                                     color='grey')
                            ax.fill(angles, values, 'lightgrey', alpha=0.03)

                    plt.title('LinkedIn Performance Radar Chart (Normalized KPIs)', size=16, color='black', y=1.15)
                    # Adjusted bbox_to_anchor to move the legend outside the plot area
                    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), fontsize=10)
                    plt.show()

            # --- Analysis 5: Performance Ranking ---
            print("\n--- Performance Ranking for Each KPI (1 = Best) ---")
            rank_data = {}
            for kpi in numerical_kpis:
                # Rank in descending order (higher value is better for most engagement KPIs)
                ranked_series = df_combined.dropna(subset=[kpi]).sort_values(by=kpi, ascending=False)['Account']
                rank_data[kpi] = ranked_series.tolist()

            df_rank = pd.DataFrame(rank_data)
            # Transpose to have KPIs as rows and ranks as columns for better readability
            print(df_rank.T.to_string())


            # --- Analysis 6: Feature Importance (Machine Learning Model) ---
            print("\n--- Training Machine Learning Model for Feature Importance ---")

            target_kpi = 'Avg. Eng. Rate by Reach (without clicks)'  # Choose a relevant engagement metric as target
            feature_importances_df = pd.DataFrame()  # Initialize empty DataFrame for feature importances

            if target_kpi not in numerical_kpis:
                print(f"Warning: Target KPI '{target_kpi}' not found or not numeric. Skipping Feature Importance.")
            else:
                # User-specified features for the ML model
                features = [
                    'Total Comments',
                    'Total Reactions',
                    'Total Followers',
                    'Total Shares',
                    'Total Posts'
                ]

                # Validate that all specified features exist in numerical_kpis
                available_features = [f for f in features if f in numerical_kpis]
                if len(available_features) != len(features):
                    missing_features = [f for f in features if f not in numerical_kpis]
                    print(f"Warning: The following specified features are not found in the data: {missing_features}. "
                          f"Using only available features: {available_features}")
                features = available_features # Update features to only include those found

                # Ensure there are still features left after filtering
                if not features:
                    print("No suitable features found for modeling after filtering. Check your KPI names and target selection.")
                    # Do not exit here, allow the rest of the script to run if possible, but skip ML model
                else:
                    X = df_combined[features].copy() # Features for the model
                    y = df_combined[target_kpi].copy() # Target variable

                    # Drop rows where either features or target have NaN values to ensure clean training data
                    combined_xy = pd.concat([X, y], axis=1).dropna()
                    X_clean = combined_xy[features]
                    y_clean = combined_xy[target_kpi]

                    if X_clean.empty or len(X_clean) < 2:
                        print(
                            "Not enough complete data rows after handling missing values for Feature Importance modeling.")
                    else:
                        # Scale features using StandardScaler
                        scaler_ml = StandardScaler()
                        X_scaled = scaler_ml.fit_transform(X_clean)

                        # Train a RandomForestRegressor model
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                        model.fit(X_scaled, y_clean)

                        # Extract and sort feature importances
                        feature_importances_df = pd.DataFrame({
                            'Feature': X_clean.columns,
                            'Importance': model.feature_importances_
                        }).sort_values(by='Importance', ascending=False)

                        print("\n--- Feature Importances for Predicting Engagement Rate ---")
                        print(feature_importances_df.to_string())

                        # Plot feature importances
                        plt.figure(figsize=(10, 6))
                        sns.barplot(x='Importance', y='Feature', data=feature_importances_df)
                        plt.title(f'Feature Importance for {target_kpi}', fontsize=14)
                        plt.xlabel('Importance', fontsize=12)
                        plt.ylabel('KPI Feature', fontsize=12)
                        plt.tight_layout()
                        plt.show()

            # --- Generate Hypotheses using Gemini API ---
            print("\n" + "=" * 80)
            print("--- GENERATING HYPOTHESES WITH GEMINI API ---")
            print("=" * 80)

            # Generate the prompt string based on the analysis results
            prompt_text = generate_gemini_prompt_summary_string(df_combined, YOUR_ACCOUNT_NAME, feature_importances_df)
            print("\nPrompt sent to Gemini:\n")
            # Use to_markdown for consistent formatting of the prompt output
            print(to_markdown(prompt_text))
            print("\n" + "=" * 80 + "\n")

            try:
                # Call the Gemini API to generate content
                model_genai = genai.GenerativeModel('gemini-2.0-flash')
                response = model_genai.generate_content(prompt_text)

                print("\n--- GENERATED HYPOTHESES FROM GEMINI ---")
                print("=" * 80)
                # Check if candidates and content exist before trying to access text
                if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    print(to_markdown(response.candidates[0].content.parts[0].text))
                else:
                    print("Gemini API returned an unexpected response structure.")
                    print(response) # Print raw response for debugging in case of unexpected response
                print("=" * 80)
            except Exception as e:
                print(f"Error calling Gemini API: {e}")
                print("Please check your API key, internet connection, and API usage limits.")

    print("\nAnalysis complete. Review the console output, plots, and Gemini-generated hypotheses.")
