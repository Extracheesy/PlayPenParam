import os
import sys
import pandas as pd
import utils

from convert_to_xcel import convert_csv_for_excel

def list_csv_files(input_dir: str) -> list:
    """
    Lists all CSV files in the specified directory.

    :param input_dir: Path to the directory containing CSV files.
    :return: A list of full paths to CSV files.
    """
    csv_files = []

    # List all files in the directory
    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith('.csv'):
            full_path = os.path.join(input_dir, file_name)
            csv_files.append(full_path)

    return csv_files


def merge_and_remove_duplicates(csv_files: list) -> pd.DataFrame:
    """
    Reads and merges all CSV files into a single DataFrame,
    then removes duplicate rows.

    :param csv_files: A list of full paths to CSV files.
    :return: A pandas DataFrame containing merged data without duplicates.
    """
    dataframes = []

    for f in csv_files:
        df = pd.read_csv(f, sep=';')
        dataframes.append(df)

    if not dataframes:
        print("No CSV files found. Returning an empty DataFrame.")
        return pd.DataFrame()

    # Concatenate and remove duplicates
    merged_df = pd.concat(dataframes, ignore_index=True)
    merged_df.drop_duplicates(inplace=True)

    return merged_df


def save_merged_csv(df: pd.DataFrame, output_path: str) -> None:
    """
    Saves the DataFrame to a CSV file at the specified path.

    :param df: The pandas DataFrame to be saved.
    :param output_path: The path (including filename) where the CSV will be saved.
    """
    df.to_csv(output_path, index=False)
    print(f"Data saved to: {output_path}")

    duplicated_ids_list = df["ID"][df["ID"].duplicated(keep=False)].unique().tolist()

    # Print the list
    print("Duplicated IDs list:", duplicated_ids_list)


def main():
    """
    Main function that orchestrates listing CSV files, merging them,
    removing duplicates, and saving the final CSV.

    Expected usage:
      python merge_csv.py <input_directory> <output_csv_file>
    """
    input_dir = r"C:\Users\INTRADE\PycharmProjects\Analysis\ObelixParam\test_multi_trend\result_test\run_failed\exel"
    input_dir = r"C:\Users\INTRADE\PycharmProjects\Analysis\ObelixParam\test_multi_trend_selection\result_test\backup\excel"
    output_csv = r"C:\Users\INTRADE\PycharmProjects\Analysis\ObelixParam\test_multi_trend_selection\result_test\backup\excel\output_merged.csv"

    # 1. List all CSV files
    csv_files = list_csv_files(input_dir)

    # 2. Merge them and remove duplicates
    merged_df = merge_and_remove_duplicates(csv_files)



    # 1. Convert END_DATE to datetime
    merged_df["END_DATE"] = pd.to_datetime(merged_df["END_DATE"], format="%Y-%m-%d %H:%M:%S,%f%z")

    # 2. Sort the DataFrame by END_DATE
    df_sorted = merged_df.sort_values(by="END_DATE")

    # 3. Drop duplicates based on ID, keeping the last (the one with the max END_DATE)
    df_latest = df_sorted.drop_duplicates(subset="ID", keep="last")

    merged_df = df_latest

    # 3. Save the merged DataFrame to the output path
    save_merged_csv(merged_df, output_csv)

    new_excel_path = utils.add_exel_before_csv(output_csv)
    convert_csv_for_excel(output_csv, new_excel_path)



if __name__ == "__main__":
    main()
