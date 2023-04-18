import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import io

import librosa

# Load the model.
model = hub.load('https://tfhub.dev/google/yamnet/1')

# Input: 3 seconds of silence as mono 16 kHz waveform samples.
waveform = np.zeros(3 * 16000, dtype=np.float32)

reduced_noise, sr = librosa.load("audios/audio_42.wav", sr=44100, mono=True, duration=5)

# Run the model, check the output.
scores, embeddings, log_mel_spectrogram = model(reduced_noise)

# Find the name of the class with the top score when mean-aggregated across frames.
def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    class_map_csv = io.StringIO(class_map_csv_text)
    class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
    class_names = class_names[1:]  # Skip CSV header
    return class_names


class_map_path = model.class_map_path().numpy()
class_names = class_names_from_csv(tf.io.read_file(class_map_path).numpy().decode('utf-8'))
print(class_names[scores.numpy().mean(axis=0).argmax()])  # Should print 'Silence'.