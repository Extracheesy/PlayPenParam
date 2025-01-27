import pandas as pd

def remove_duplicates(input_file, output_file):
    """
    Reads a CSV file, drops duplicate rows, and saves the cleaned data.
    
    :param input_file: Path to the input CSV file.
    :param output_file: Path to the output CSV file.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        print(f"Original data shape: {df.shape}")

        # Drop duplicate rows
        df_no_duplicates = df.drop_duplicates()
        print(f"Data shape after removing duplicates: {df_no_duplicates.shape}")

        # Save the cleaned DataFrame to a new CSV file
        df_no_duplicates.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    input_file = "results_vbtpro/portfolio_stats_summary.csv"
    output_file = "results_vbtpro/portfolio_stats_summary_no_duplicates.csv"

    # Remove duplicates and save the cleaned file
    remove_duplicates(input_file, output_file)

if __name__ == "__main__":
    main()
