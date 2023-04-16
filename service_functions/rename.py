import os

import csv

# Function to save a list to a one-column CSV file
def save_list_to_csv(list_data, csv_filename):
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write each list item as a separate row in the CSV
        for item in list_data:
            writer.writerow([item])
    print(f"List data saved to {csv_filename} successfully.")

# Function to rename audio files in a folder
def rename_audio_files(folder_path, prefix=""):
    filenames = []
    count = 1
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp3") or filename.endswith(".wav") or filename.endswith(".ogg"):
            # Extract the file extension
            file_extension = filename.split(".")[-1]
            # Create the new filename with prefix and suffix
            new_filename = prefix + str(count) + "." + file_extension
            # Construct the old and new file paths
            old_filepath = os.path.join(folder_path, filename)
            new_filepath = os.path.join(folder_path, new_filename)
            # Rename the file
            # os.rename(old_filepath, new_filepath)
            print(f"Renamed {filename} to {new_filename}")

            filenames.append(new_filename)
            
            count = count + 1

    save_list_to_csv(filenames, "column_names.csv")
    


# Example usage
folder_path = "./audios"  # Replace with the actual folder path
prefix = "28_128"  # Replace with the desired prefix
rename_audio_files(folder_path, prefix)
