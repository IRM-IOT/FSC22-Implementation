import pyaudio

chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 1
fs = 22050  # Record at 44100 samples per second
seconds = 5

p = pyaudio.PyAudio()  # Create an interface to PortAudio

print('Recording')

stream = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunk,
                input=True)

frames = []  # Initialize array to store frames

def record_audio(sharedlst):
    i=1
    while True:
        data = stream.read(chunk)
        sharedlst.append(data)