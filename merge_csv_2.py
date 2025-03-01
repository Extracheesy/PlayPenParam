import os
import pandas as pd
import utils
from convert_to_xcel import convert_csv_for_excel

import os
import pandas as pd

def choose_row(group):
    """
    For a given group (all rows with the same ID), if all END_DATE values are NaT,
    return the first row. Otherwise, return the row with the maximum (latest) END_DATE.
    """
    if group["END_DATE"].isna().all():
        return group.iloc[0]
    else:
        return group.loc[group["END_DATE"].idxmax()]

def merge_csv_by_pattern(directory, pattern, output_filename):
    """
    Scans the specified directory (and its subdirectories) for CSV files.
    It only processes CSV files whose filenames contain the given pattern (case-insensitive).
    Each CSV is read using a fallback mechanism: first with the default settings,
    and if that fails, with a semicolon separator.
    After merging, duplicate rows are dropped, the column "Unnamed: 0" is removed,
    and for duplicate "ID" values only the row with the oldest "END_DATE" is kept.
    The final DataFrame is saved to the output CSV file in the same directory.
    """
    dataframes = []

    # Traverse the directory recursively
    for root, _, files in os.walk(directory):
        for file in files:
            # Only process CSV files that contain the pattern in the filename
            if file.endswith('.csv') and not (pattern.lower() in file.lower()):
                continue

            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                try:
                    df = pd.read_csv(file_path)
                except Exception as e:
                    print(f"Standard read failed for {file_path}: {e}. Trying with separator ';'")
                    try:
                        df = pd.read_csv(file_path, sep=';')
                    except Exception as e2:
                        print(f"Failed reading {file_path} with separator ';': {e2}")
                        continue
                dataframes.append(df)

    if dataframes:
        # Merge all DataFrames and drop duplicate rows
        merged_df = pd.concat(dataframes, ignore_index=True)
        merged_df.drop_duplicates(inplace=True)

        # Drop the column "Unnamed: 0" if it exists
        if 'Unnamed: 0' in merged_df.columns:
            merged_df.drop(columns=['Unnamed: 0'], inplace=True)

        df_unique = df.drop_duplicates(subset='ID', keep='first')
        merged_df = df_unique

        # Save the merged DataFrame in the specified directory
        output_path = os.path.join(directory, output_filename)
        merged_df.to_csv(output_path, index=False)

        # Convert CSV for Excel if needed
        new_excel_path = utils.add_exel_before_csv(output_path)
        convert_csv_for_excel(output_path, new_excel_path)

        print(f"Merged CSV saved as: {output_path}")
    else:
        print(f"No CSV files with pattern '{pattern}' were found in {directory}.")


def main():
    # Provided input values
    directory = r"C:\Users\INTRADE\PycharmProjects\Analysis\ObelixParam\test_multi_trend_selection\result_test\backup"
    pattern = "exel"
    output_filename = "full_data_merged.csv"

    merge_csv_by_pattern(directory, pattern, output_filename)


if __name__ == "__main__":
    main()


