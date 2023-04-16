from scipy.io import wavfile
import noisereduce as nr
import os
import csv
import librosa
from scipy import signal
import random


def f_high(y,sr):
    b,a = signal.butter(10, 2000/(sr/2), btype='highpass')
    yf = signal.lfilter(b,a,y)
    return yf

# # Function to save a list to a one-column CSV file
# def save_list_to_csv(list_data, csv_filename):
#     with open(csv_filename, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         # Write each list item as a separate row in the CSV
#         for item in list_data:
#             writer.writerow([item])
#     print(f"List data saved to {csv_filename} successfully.")

# def noise_reduce(folder_path, prefix=""):
#     filenames = []
#     count = 1
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".mp3") or filename.endswith(".wav") or filename.endswith(".ogg"):
#             # load data
#             file_extension = filename.split(".")[-1]
#             new_filename = prefix + str(count) + "." + file_extension
#             # Construct the old and new file paths
#             old_filepath = os.path.join(folder_path, filename)
#             new_filepath = os.path.join(folder_path, new_filename)
#             # perform noise reduction
#             audio, sr = librosa.load(old_filepath, sr=22050, mono=True)

#             reduced_noise = f_high(audio, sr)
#             wavfile.write(new_filepath, sr, reduced_noise)
#             count = count + 1
#             filenames.append(new_filename)
    
#     save_list_to_csv(filenames, "column_names.csv")


# # Example usage
# folder_path = "./audios"  # Replace with the actual folder path
# prefix = "new_"  # Replace with the desired prefix
# noise_reduce(folder_path, prefix)


audio, sr = librosa.load("./audios/audio_1.wav", sr=22050, mono=True)

reduced_noise = librosa.effects.pitch_shift(audio,sr=22050, n_steps=-2)#f_high(audio, sr)#nr.reduce_noise(y=audio, sr=sr, stationary=True, prop_decrease=0.8)

wavfile.write("reduced.wav", sr, reduced_noise)


