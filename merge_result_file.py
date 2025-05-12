import os
import shutil
import re
import glob
import pandas as pd

import utils
from convert_to_xcel import convert_csv_for_excel

def main():
    result_dir = 'result'

    directory = r"C:\Users\INTRADE\PycharmProjects\Analysis\ObelixParam\test_multi_trend_selection_test_7"
    os.chdir(directory)

    base_dir = os.getcwd()

    # Step 1: Create or clean the result_merged directory.
    result_merged_dir = os.path.join(base_dir, "result_merged")
    if os.path.exists(result_merged_dir):
        for item in os.listdir(result_merged_dir):
            item_path = os.path.join(result_merged_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
    else:
        os.makedirs(result_merged_dir)

    # Regex to match files starting with one or more digits followed by an underscore.
    file_pattern = re.compile(r'^(\d+)_')

    # Get only directories in base_dir that start with "result_test_"
    result_test_dirs = [d for d in glob.glob(os.path.join(base_dir, "result_test_*")) if os.path.isdir(d)]

    # Process each result_test directory
    for test_dir in result_test_dirs:
        # Delete the CSV files named "output_data_full_test_tmp.csv" and "output_data_full_test_tmp_exel.csv"
        for pattern in ['output_data_full_test_tmp.csv', 'output_data_full_test_tmp_exel.csv']:
            matching_files = glob.glob(os.path.join(test_dir, '**', pattern), recursive=True)
            for file_path in matching_files:
                os.remove(file_path)
                print(f"Deleted: {file_path}")

        # Extract date1 from the directory name.
        # Expected format: "result_test_date1_date2"
        dirname = os.path.basename(test_dir)
        try:
            date1 = dirname.split("result_test_")[1].split("_")[0]
        except IndexError:
            print(f"Directory name {dirname} doesn't match expected format. Skipping...")
            continue

        # Recursively find and copy matching files.
        all_files = glob.glob(os.path.join(test_dir, '**', '*'), recursive=True)
        for file_path in all_files:
            if os.path.isfile(file_path):
                filename = os.path.basename(file_path)
                match = file_pattern.match(filename)
                if match:
                    numeric_part = match.group(1)
                    target_subdir = os.path.join(result_merged_dir, numeric_part)
                    os.makedirs(target_subdir, exist_ok=True)

                    # Prepend the extracted date to the filename.
                    new_filename = f"{date1}_{filename}"
                    dest_file = os.path.join(target_subdir, new_filename)

                    shutil.copy2(file_path, dest_file)
                    print(f"Copied: {file_path} -> {dest_file}")

    # Step 3: Merge all CSV files named "output_data_full_test_exel.csv" from each test_dir.
    merged_dfs = []
    for test_dir in result_test_dirs:
        # Use glob to recursively search for the target CSV file.
        exel_files = glob.glob(os.path.join(test_dir, '**', 'output_data_full_test.csv'), recursive=True)
        for file_path in exel_files:
            try:
                df = pd.read_csv(file_path)

                # Convert "Benchmark Return [%]" from string to float (remove '%' if necessary)
                df["Total Return [%]"] = df["Total Return [%]"].astype(float)
                # Filter out rows where "Benchmark Return [%]" is below 0
                df = df[df["Total Return [%]"] > 0]

                # Convert the two percentage columns to float and filter rows based on the condition:
                # Drop rows where M_CUMULATIVE_RETURN < M_BUY_HOLD_RETURN
                m_cum = df["M_CUMULATIVE_RETURN"].str.rstrip('%').astype(float)
                m_buy = df["M_BUY_HOLD_RETURN"].str.rstrip('%').astype(float)
                df = df[m_cum >= m_buy]

                df.insert(0, 'RANK BRUT', df['Total Return [%]'].rank(method='min', ascending=False).astype(int))

                diff = df["Total Return [%]"].astype(float) - df["Benchmark Return [%]"].astype(float)

                # Rank the difference in descending order (largest difference gets rank 1)
                rank_benchmark = diff.rank(method='min', ascending=False).astype(int)
                # Insert the new column as the second column (index 1)
                df.insert(1, "RANK_benchmark", rank_benchmark)

                # Convert percentage columns from strings to floats
                df["Total Return [%]"] = df["Total Return [%]"].astype(float)
                df["Benchmark Return [%]"] = df["Benchmark Return [%]"].astype(float)

                # Calculate the proportional value: Total Return [%] / Benchmark Return [%]
                proportional = df["Total Return [%]"] / df["Benchmark Return [%]"]

                # Insert the "proportional" column at position 10.
                # (Note: positions are zero-indexed, so position 10 is the 11th column.)
                df.insert(3, "proportional", proportional)

                # Create the "RANK_PROPORTIONAL" column by ranking the "proportional" values.
                # Here, higher ratios get a better rank (rank 1 is highest)
                rank_proportional = proportional.rank(method='min', ascending=False).astype(int)

                # Insert the ranking column at position 3 (fourth column in the DataFrame)
                df.insert(3, "RANK_PROPORTIONAL", rank_proportional)

                # Move "ID" as the first column
                df.insert(0, "ID", df.pop("ID"))

                merged_dfs.append(df)
                print(f"Read: {file_path}")
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")

    if merged_dfs:
        merged_df = pd.concat(merged_dfs, ignore_index=True)
        # Create output directory for the merged CSV.
        merged_output_dir = os.path.join(base_dir, "output_data_full_test")
        os.makedirs(merged_output_dir, exist_ok=True)
        merged_file_path = os.path.join(merged_output_dir, "merged_output_data_full_test.csv")

        cols_to_drop = [col for col in merged_df.columns if col.startswith('Unnamed:')]
        cols_to_drop += ["LOW_TIMEFRAME", "HIGH_TIMEFRAME", "Max Gross Exposure [%]",
                         "Max Drawdown Duration", "Avg Winning Trade Duration",
                         "Avg Losing Trade [%]", "Avg Winning Trade Duration",
                         "Avg Losing Trade Duration", ]
        merged_df = merged_df.drop(columns=cols_to_drop)
        merged_df.columns = merged_df.columns.str.upper()
        merged_df.to_csv(merged_file_path, index=False)
        print(f"Merged CSV saved to: {merged_file_path}")

        new_excel_path = utils.add_exel_before_csv(merged_file_path)
        convert_csv_for_excel(merged_file_path, new_excel_path)
    else:
        print("No CSV files named 'output_data_full_test_exel.csv' found to merge.")

    # Create 'result' directory if it doesn't exist;
    # if it does, clear out its contents.
    if os.path.exists(result_dir):
        # Remove all files and directories inside 'result'
        for item in os.listdir(result_dir):
            item_path = os.path.join(result_dir, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)  # remove file or symbolic link
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)  # remove directory and its contents
            except Exception as e:
                print(f"Error deleting {item_path}: {e}")
    else:
        os.makedirs(result_dir)

    # Now move the directories into 'result'
    shutil.move('result_merged', os.path.join(result_dir, 'result_merged'))
    shutil.move('output_data_full_test', os.path.join(result_dir, 'output_data_full_test'))

if __name__ == "__main__":
    main()
