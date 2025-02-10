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
    # input_file = "./batch_stats_df_merged/batch_stats_df_merged.csv"  # Replace with your input file path
    # output_file = "./batch_stats_df_merged/batch_stats_df_merged_excel_output.csv"  # Replace with your desired output file path

    # input_file = "./best_params_per_group__2.csv"  # Replace with your input file path
    # output_file = "./best_params_per_group__2__excel_output.csv"  # Replace with your desired output file path

    # input_file = "./weighted_score/scored_results.csv"  # Replace with your input file path
    # output_file = "./weighted_score/scored_results_excel_output.csv"  # Replace with your desired output file path

    # input_file = "./weighted_score/merged_obelix_configurations.csv"  # Replace with your input file path
    # output_file = "./weighted_score/merged_obelix_configurations_output.csv"  # Replace with your desired output file path

    input_file = "./weighted_score/merged_obelix_configurations_with_scores.csv"  # Replace with your input file path
    output_file = "./weighted_score/merged_obelix_configurations_with_scores_output.csv"  # Replace with your desired output file path

    input_file = "results_vbtpro/portfolio_stats_summary.csv"  # Replace with your input file path
    output_file = "results_vbtpro/portfolio_stats_summary_exel_output.csv"  # Replace with your desired output file path

    input_file = "./results_vbtpro/scored_results.csv"
    output_file = "./results_vbtpro/scored_results_exel_output.csv"

    input_file = "./merged_max_kmeans_output/merged_output.csv"
    output_file = "./merged_max_kmeans_output/merged_output_xcel.csv"

    input_file = "./test_param/portfolio_stats_summary.csv"
    output_file = "./test_param/portfolio_stats_summary_xcel.csv"

    input_file = "./results_batches_vbtpro/batch_stats_df_merged.csv"
    output_file = "./results_batches_vbtpro/batch_stats_df_merged_xcel.csv"

    convert_csv_for_excel(input_file, output_file)
    print("CSV has been converted and saved.")
