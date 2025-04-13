import pandas as pd

csv_filename = r"C:\Users\INTRADE\PycharmProjects\Analysis\ObelixParam\test_multi_trend_selection_test_4\result_test_2024-01-01_2025-04-08\output_data_full_test.csv"

# Read the CSV file with semicolon separator
df = pd.read_csv(csv_filename, sep=';')

# Write the dataframe to a new CSV file with a comma separator
df.to_csv(csv_filename, sep=',', index=False)

print("toto")