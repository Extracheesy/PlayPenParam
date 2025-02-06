import os
import pandas as pd
import matplotlib.pyplot as plt


class CSVProcessor:
    def __init__(self, directory, file_names):
        self.directory = directory
        self.file_names = file_names
        self.dataframes = []

    def load_csv_files(self):
        """Loads CSV files, converts the first column to datetime, and sets it as the index."""
        for file_name in self.file_names:
            file_path = os.path.join(self.directory, file_name)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])  # Convert first column to datetime
                df.set_index(df.columns[0], inplace=True)  # Set as index
                self.dataframes.append(df)
            else:
                print(f"Warning: {file_name} not found in {self.directory}")

    def get_common_columns_and_length(self):
        """Finds common columns and the minimum row length across all CSVs."""
        common_columns = set(self.dataframes[0].columns)
        common_index = self.dataframes[0].index

        for df in self.dataframes[1:]:
            common_columns.intersection_update(df.columns)
            common_index = common_index.intersection(df.index)

        return list(common_columns), common_index

    def filter_common_data(self, common_columns, common_index):
        """Filters all dataframes to retain only common columns and align indices."""
        return [df[common_columns].loc[common_index] for df in self.dataframes]


class DataAnalyzer:
    def __init__(self, dataframes, common_columns, result_directory):
        self.dataframes = dataframes
        self.common_columns = common_columns
        self.result_directory = result_directory

    def compute_equal_value_percentage(self):
        """Computes the percentage of equal values across dataframes for each common column."""
        percentages = {}
        df_ref = self.dataframes[0]

        for column in self.common_columns:
            equal_counts = sum(
                (df_ref[column] == df[column].reindex(df_ref.index)).sum() for df in self.dataframes[1:]
            )
            total_comparisons = len(df_ref) * (len(self.dataframes) - 1)
            percentages[column] = (equal_counts / total_comparisons) * 100 if total_comparisons > 0 else 0

        return percentages

    def plot_common_columns(self):
        """Plots graphs for each common column across different files."""
        for column in self.common_columns:
            plt.figure(figsize=(10, 5))
            for i, df in enumerate(self.dataframes):
                plt.plot(df.index, df[column], label=f'File {i + 1}')
            plt.xlabel("Time")
            plt.ylabel(column)
            plt.title(f"Comparison of {column}")
            plt.legend()
            plt.savefig(os.path.join(self.result_directory, f"{column}_comparison.png"))
            plt.close()


def main(directory, file_names):
    result_directory = os.path.join(directory, "result")
    os.makedirs(result_directory, exist_ok=True)

    processor = CSVProcessor(directory, file_names)
    processor.load_csv_files()
    common_columns, common_index = processor.get_common_columns_and_length()
    filtered_dataframes = processor.filter_common_data(common_columns, common_index)

    analyzer = DataAnalyzer(filtered_dataframes, common_columns, result_directory)
    percentages = analyzer.compute_equal_value_percentage()
    analyzer.plot_common_columns()

    print("Percentage of equal values per column:")
    for column, percentage in percentages.items():
        print(f"{column}: {percentage:.2f}%")

if __name__ == "__main__":
    directory = "./output_data"
    file_names = ["backupt_data.csv", "data_backtest.csv"]  # Example file names
    main(directory, file_names)