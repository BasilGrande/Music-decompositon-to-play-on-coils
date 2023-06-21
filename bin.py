import librosa
import numpy as np
import struct

def sample_amplitude(filename, target_sampling_rate, duration):
    # Load the audio file
    audio, sr = librosa.load(filename, sr=None, duration=duration)

    # Calculate the ratio for downsampling
    downsample_ratio = int(sr / target_sampling_rate)

    # Downsample the audio array
    audio_downsampled = audio[::downsample_ratio]

    return audio_downsampled

# Set the filename, desired sampling rate, and duration in seconds
filename = "out.mp3"
target_sampling_rate = 5000  # 5kHz
duration = 10  # Number of seconds to use from the song

# Sample the amplitude
amplitude_samples = sample_amplitude(filename, target_sampling_rate, duration)

# Normalize the amplitude samples to a range of [-1, 1]
normalized_samples = amplitude_samples / np.max(np.abs(amplitude_samples))

# Scale the normalized samples to the desired range [-128, 127]
scaled_samples = normalized_samples * 127

# Round each value to the nearest digit
rounded_samples = np.round(scaled_samples)

# Convert each sample to 1-byte signed integer (int8)
rounded_samples = rounded_samples.astype(np.int8)

# Save the rounded amplitude samples to a binary file
with open("output.bin", "wb") as f:
    for sample in rounded_samples:
        f.write(struct.pack('b', sample))
