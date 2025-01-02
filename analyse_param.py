import pandas as pd
import os

def convert_csv_for_excel(input_csv, output_csv):
    """
    Reads a CSV file, replaces all '.' with ',' in all columns,
    and saves the result to another CSV using ';' as the separator.

    :param input_csv: Path to the input CSV file.
    :param output_csv: Path to the output CSV file.
    """

    # Read the CSV. Use dtype=str to ensure all data is read as strings,
    # so that we can reliably replace '.' with ',' everywhere.
    df = pd.read_csv(input_csv, dtype=str)

    # Replace all '.' with ',' in every column, for every cell
    df = df.applymap(lambda x: x.replace('.', ',') if isinstance(x, str) else x)

    # Save the modified dataframe to a new CSV with ';' as the separator
    df.to_csv(output_csv, sep=';', index=False)


import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


class CSVProcessor:
    def __init__(self, input_file, chunk_size=50_000, num_threads=4):
        """
        Constructor for CSVProcessor.

        :param input_file: Path to the input CSV file.
        :param chunk_size: Number of rows to read at a time (chunk size).
        :param num_threads: Number of worker threads to use.
        """
        self.input_file = input_file
        self.chunk_size = chunk_size
        self.num_threads = num_threads

        # Keep counters for duplicates
        self.local_duplicates_dropped = 0
        self.global_duplicates_dropped = 0

    @staticmethod
    def _drop_duplicates_in_chunk(chunk_df):
        """
        A static method to drop duplicates within a single chunk DataFrame.
        Returns the deduplicated DataFrame and the number of dropped rows.
        """
        rows_before = len(chunk_df)
        dedup_df = chunk_df.drop_duplicates()
        rows_after = len(dedup_df)

        # Number of duplicates dropped in this chunk
        dup_dropped = rows_before - rows_after

        return dedup_df, dup_dropped

    def deduplicate_and_save(self, output_file):
        """
        Reads the CSV in chunks, processes them in parallel to remove duplicates within each chunk,
        then removes duplicates across the combined result, and finally saves to a new CSV.
        Also prints the number of duplicates dropped (local and global).

        :param output_file: Path to the output CSV file.
        """
        # 1. Read CSV in chunks and store them in a list (in the main thread)
        chunks = []
        for chunk_df in pd.read_csv(self.input_file, chunksize=self.chunk_size):
            chunks.append(chunk_df)

        # 2. Create a thread pool and process each chunk in parallel
        deduped_chunks = []
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit tasks for each chunk
            futures = [executor.submit(self._drop_duplicates_in_chunk, ch) for ch in chunks]

            # Gather results as they complete
            for future in as_completed(futures):
                dedup_chunk_df, chunk_dup_dropped = future.result()
                # Accumulate the number of local duplicates dropped
                self.local_duplicates_dropped += chunk_dup_dropped
                deduped_chunks.append(dedup_chunk_df)

        # 3. Combine all locally deduplicated chunks into a single DataFrame
        combined_df = pd.concat(deduped_chunks, ignore_index=True)

        # 4. Drop duplicates again globally across all chunks
        global_before = len(combined_df)
        combined_df.drop_duplicates(inplace=True)
        global_after = len(combined_df)
        self.global_duplicates_dropped = global_before - global_after

        # 5. Save the final deduplicated DataFrame to CSV
        combined_df.to_csv(output_file, index=False)

        # Print the stats
        total_dropped = self.local_duplicates_dropped + self.global_duplicates_dropped
        print(f"Local duplicates dropped (within chunks): {self.local_duplicates_dropped}")
        print(f"Global duplicates dropped (across all chunks): {self.global_duplicates_dropped}")
        print(f"Total duplicates dropped: {total_dropped}")


def main():
    path = "./batch_stats_df_merged"
    input_csv_filename = "batch_stats_df_merged.csv"  # Replace with your actual input path
    output_csv_filename = "output_batch_stats_df_merged.csv"  # Replace with your desired output path

    # Construct full paths
    input_csv = os.path.join(path, input_csv_filename)
    output_csv = os.path.join(path, output_csv_filename)

    print("Input CSV path:", input_csv)
    print("Output CSV path:", output_csv)

    # You can tune chunk_size and num_threads based on your system resources and data characteristics
    processor = CSVProcessor(input_file=input_csv, chunk_size=50_000, num_threads=4)
    processor.deduplicate_and_save(output_csv)
    print("Finished deduplicating the CSV file with multithreading.")

    input_file = output_csv  # Replace with your input file path
    output_file = "output_xls.csv"  # Replace with your desired output file path
    output_file = os.path.join(path, output_file)

    convert_csv_for_excel(input_file, output_file)
    print("CSV has been converted and saved.")

if __name__ == "__main__":
    main()


