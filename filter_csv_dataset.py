import pandas as pd
import os
from convert_to_xcel import convert_csv_for_excel


def filter_csv(input_dir: str, filename: str, filename_excel: str, output_filename: str, output_excel_filename: str):
    input_path = os.path.join(input_dir, filename)
    input_excel_path = os.path.join(input_dir, filename_excel)
    output_path = os.path.join(input_dir, output_filename)
    output_excel_path = os.path.join(input_dir, output_excel_filename)

    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' not found.")
        return

    # Load the CSV file
    df = pd.read_csv(input_path)

    # Check if required columns exist
    required_columns = ["TYPE", "COMPARE RETURN FLAG", "TOTAL RETURN [%]", "SYMBOL", "TIMEFRAME", "MA_TYPE", "SCORE_1",
                        "SCORE_2", "SCORE_3"]
    for column in required_columns:
        if column not in df.columns:
            print(f"Error: Column '{column}' not found in the CSV file.")
            return

    # Apply filtering conditions
    df = df[~((df["TYPE"] == "kmeans") | ((df["COMPARE RETURN FLAG"] == False) & (df["TOTAL RETURN [%]"] <= 50)))]

    # Keep the row with the highest values in SCORE_1, SCORE_2, and SCORE_3 for each combination of SYMBOL, TIMEFRAME, and MA_TYPE
    df = df.sort_values(by=["SCORE_1", "SCORE_2", "SCORE_3"], ascending=False)
    df = df.groupby(["SYMBOL", "TIMEFRAME", "MA_TYPE"]).first().reset_index()

    # Drop duplicate rows
    df = df.drop_duplicates()

    # Save the filtered data
    df.to_csv(output_path, index=False)
    print(f"Filtered data saved to '{output_path}'")

    convert_csv_for_excel(output_path, output_excel_path)
    convert_csv_for_excel(input_path, input_excel_path)


def main():
    input_directory = "new_dataset"
    csv_filename = "merged_output.csv"
    csv_excel_filename = "merged_output_excel.csv"
    output_csv_filename = "filtered_output.csv"
    output_csv_excel_filename = "filtered_output_excel.csv"

    filter_csv(input_directory, csv_filename, csv_excel_filename, output_csv_filename, output_csv_excel_filename)


if __name__ == "__main__":
    main()
