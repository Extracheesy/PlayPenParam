import os
import re
import sys

def rename_files(directory):
    if not os.path.isdir(directory):
        print(f"The directory '{directory}' does not exist.")
        return

    # Find the maximum number prefix in filenames
    max_number_length = 0
    for filename in os.listdir(directory):
        match = re.match(r'^(\d+)_', filename)
        if match:
            number_part = match.group(1)
            max_number_length = max(max_number_length, len(number_part))

    for filename in os.listdir(directory):
        # Match filenames starting with a number followed by an underscore
        match = re.match(r'^(\d+)_', filename)
        if match:
            number_part = match.group(1)
            new_number = number_part.zfill(max_number_length)  # Adjust to max number length
            new_filename = filename.replace(number_part + "_", new_number + "_", 1)

            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)

            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")

if __name__ == "__main__":
    directory = "new_dataset\output_plot"
    rename_files(directory)
