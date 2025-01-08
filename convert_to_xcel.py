import pandas as pd


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


# Example usage:
if __name__ == "__main__":
    input_file = "input.csv"  # Replace with your input file path
    output_file = "output.csv"  # Replace with your desired output file path

    convert_csv_for_excel(input_file, output_file)
    print("CSV has been converted and saved.")
