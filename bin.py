import librosa
import numpy as np
import struct
import matplotlib.pyplot as plt

def sample_amplitude(filename, target_sampling_rate, duration):
    # Load the audio file
    audio, sr = librosa.load(filename, sr=None, duration=duration)

    # Calculate the ratio for downsampling
    downsample_ratio = int(sr / target_sampling_rate)

    # Downsample the audio array
    audio_downsampled = audio[::downsample_ratio]

    return audio_downsampled

# Set the filename, desired sampling rate, and duration in seconds
filename = "7.mp3"
target_sampling_rate = 5000  # 5kHz
duration = 100  # Number of seconds to use from the song

# Sample the amplitude
amplitude_samples = sample_amplitude(filename, target_sampling_rate, duration)

# Find the index of the first non-zero value
nonzero_index = next((i for i, sample in enumerate(amplitude_samples) if sample != 0), None) + 429

# Trim the amplitude samples from the first non-zero value onwards
trimmed_samples = amplitude_samples[nonzero_index:]

# Normalize the amplitude samples to a range of [-1, 1]
normalized_samples = trimmed_samples / np.max(np.abs(trimmed_samples))

# Scale the normalized samples to the desired range [0, 255]
scaled_samples = (normalized_samples * 127) + 128

# Round each value to the nearest integer
rounded_samples = np.round(scaled_samples)

# Convert each sample to 1-byte unsigned integer (uint8)
rounded_samples = rounded_samples.astype(np.uint8)

# Save the rounded amplitude samples to a binary file
with open("output.bin", "wb") as f:
    for sample in rounded_samples:
        f.write(struct.pack('B', sample))

# Plot the rounded amplitude samples
plt.plot(rounded_samples)
plt.xlabel('Sample')
plt.ylabel('Rounded Amplitude')
plt.title('Rounded Amplitude Samples')
plt.show()
