import os
import pandas as pd

def merge_csv_files(input_file1, input_file2, output_dir, output_filename="merged_output.csv"):
    """
    Reads two CSV files, checks for missing columns, adds a 'TYPE' column to each,
    merges them, and saves the result in the specified output directory.

    :param input_file1: Path to the first CSV file.
    :param input_file2: Path to the second CSV file.
    :param output_dir: Directory where the merged CSV file will be saved.
    :param output_filename: Name of the output CSV file.
    """
    try:
        # Read both CSV files
        df1 = pd.read_csv(input_file1)
        df2 = pd.read_csv(input_file2)

        print(f"First file shape before adding 'TYPE': {df1.shape}")
        print(f"Second file shape before adding 'TYPE': {df2.shape}")

        # Find missing columns
        missing_in_df1 = set(df2.columns) - set(df1.columns)
        missing_in_df2 = set(df1.columns) - set(df2.columns)

        if missing_in_df1 or missing_in_df2:
            print("⚠️ Missing Columns Detected:")
            if missing_in_df1:
                print(f"Columns in {input_file2} but missing in {input_file1}: {missing_in_df1}")
            if missing_in_df2:
                print(f"Columns in {input_file1} but missing in {input_file2}: {missing_in_df2}")

            # Add missing columns with NaN values to ensure both DataFrames have the same columns
            for col in missing_in_df1:
                df1[col] = pd.NA
            for col in missing_in_df2:
                df2[col] = pd.NA

        # Add 'TYPE' column
        df1["TYPE"] = "max"
        df2["TYPE"] = "kmeans"

        print(f"First file shape after adding 'TYPE': {df1.shape}")
        print(f"Second file shape after adding 'TYPE': {df2.shape}")

        # Merge both DataFrames
        merged_df = pd.concat([df1, df2], axis=0, ignore_index=True)
        print(f"Merged data shape: {merged_df.shape}")

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save the merged DataFrame
        output_path = os.path.join(output_dir, output_filename)
        merged_df.to_csv(output_path, index=False)
        print(f"Merged CSV saved at: {output_path}")

    except FileNotFoundError as e:
        print(f"Error: One or both input files not found. {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    # Define input CSV file paths
    input_file1 = "results_vbtpro_max/scored_results.csv"
    input_file2 = "results_vbtpro_kmeans/scored_results.csv"

    # Define output directory
    output_dir = "merged_max_kmeans_output"

    # Merge and save the CSV file
    merge_csv_files(input_file1, input_file2, output_dir)

if __name__ == "__main__":
    main()

