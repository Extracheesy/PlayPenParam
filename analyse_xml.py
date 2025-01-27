import xml.etree.ElementTree as ET
import pandas as pd
import os

def parse_xml_to_dataframe(xml_file, score_type):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Initialize a list to hold configuration data
    data = []

    # Iterate over each symbol in the XML
    for symbol in root.findall('symbol'):
        symbol_name = symbol.get('name')  # Get the symbol name

        # Iterate over each config under the symbol
        for config in symbol.findall('config'):
            ma_type = config.get('ma_type')  # Get moving average type
            timeframe = config.get('timeframe')  # Get timeframe

            # Extract max and kmeans attributes if they exist
            max_attrs = config.find('max')
            kmeans_attrs = config.find('kmeans')

            # Add max attributes to the data
            if max_attrs is not None:
                row = {
                    'SYMBOL': symbol_name,
                    'MA_TYPE': ma_type,
                    'TIMEFRAME': timeframe,
                    'SCORE_TYPE': score_type,
                    'TYPE': 'max',
                    'VALUE': max_attrs.get('value', None),
                    'HIGH_OFFSET': max_attrs.get('HIGH_OFFSET', None),
                    'LOW_OFFSET': max_attrs.get('LOW_OFFSET', None),
                    'ZEMA_LEN_BUY': max_attrs.get('ZEMA_LEN_BUY', None),
                    'ZEMA_LEN_SELL': max_attrs.get('ZEMA_LEN_SELL', None),
                    'SSL_ATR_PERIOD': max_attrs.get('SSL_ATR_PERIOD', None)
                }
                data.append(row)

            # Add kmeans attributes to the data
            if kmeans_attrs is not None:
                row = {
                    'SYMBOL': symbol_name,
                    'MA_TYPE': ma_type,
                    'TIMEFRAME': timeframe,
                    'SCORE_TYPE': score_type,
                    'TYPE': 'kmeans',
                    'VALUE': None,  # No value for kmeans
                    'HIGH_OFFSET': kmeans_attrs.get('HIGH_OFFSET', None),
                    'LOW_OFFSET': kmeans_attrs.get('LOW_OFFSET', None),
                    'ZEMA_LEN_BUY': kmeans_attrs.get('ZEMA_LEN_BUY', None),
                    'ZEMA_LEN_SELL': kmeans_attrs.get('ZEMA_LEN_SELL', None),
                    'SSL_ATR_PERIOD': kmeans_attrs.get('SSL_ATR_PERIOD', None)
                }
                data.append(row)

    # Create a DataFrame from the data
    df = pd.DataFrame(data)
    df.columns = df.columns.str.upper()  # Convert column names to uppercase
    return df

def enhance_with_scored_results(df, scored_results_path):
    # Read the scored results CSV
    scored_results = pd.read_csv(scored_results_path)
    scored_results.columns = scored_results.columns.str.upper()  # Convert column names to uppercase

    # Ensure column types match between the DataFrames
    merge_columns = [
        'SYMBOL', 'MA_TYPE', 'TIMEFRAME', 'HIGH_OFFSET', 'LOW_OFFSET',
        'ZEMA_LEN_BUY', 'ZEMA_LEN_SELL', 'SSL_ATR_PERIOD'
    ]
    for col in merge_columns:
        if col in df.columns and col in scored_results.columns:
            df[col] = df[col].astype(str)
            scored_results[col] = scored_results[col].astype(str)

    # Merge scored results data for rows with Type == 'max' and 'kmeans'
    merged_max = pd.merge(
        df[df['TYPE'] == 'max'],
        scored_results,
        how='left',
        on=merge_columns
    )

    merged_kmeans = pd.merge(
        df[df['TYPE'] == 'kmeans'],
        scored_results,
        how='left',
        on=merge_columns
    )

    # Concatenate both merged DataFrames
    merged_df = pd.concat([merged_max, merged_kmeans], ignore_index=True)

    return merged_df

def main():
    # Specify the input directory
    input_directory = './weighted_score'
    scored_results_path = os.path.join(input_directory, 'scored_results.csv')

    # Initialize a list to store DataFrames
    dataframes = []

    # Iterate over files in the input directory
    for file_name in os.listdir(input_directory):
        if file_name.startswith('_obelix_synthesis_score_') and file_name.endswith('.xml'):
            # Extract the score suffix
            score_suffix = file_name.split('_')[-1].split('.')[0]
            score_type = f"score_{score_suffix}"

            # Parse the XML file
            xml_file_path = os.path.join(input_directory, file_name)
            df = parse_xml_to_dataframe(xml_file_path, score_type)
            dataframes.append(df)

    # Merge all DataFrames into one
    final_df = pd.concat(dataframes, ignore_index=True)

    # Enhance with scored results
    enhanced_df = enhance_with_scored_results(final_df, scored_results_path)

    # Save the final DataFrame to a CSV in the same directory
    output_csv_path = os.path.join(input_directory, 'merged_obelix_configurations_with_scores.csv')
    enhanced_df.to_csv(output_csv_path, index=False)

    print(f"Enhanced DataFrame saved to {output_csv_path}")

if __name__ == "__main__":
    main()
