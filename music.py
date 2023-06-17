import librosa
import numpy as np
import soundfile as sf
import csv
import matplotlib.pyplot as plt

# Load the audio file
audio_path = 'pour_elise.mp3'
audio_data, sampling_rate = librosa.load(audio_path)

# Segment length in seconds (100ms in this case)
segment_length = 0.1
segment_samples = int(segment_length * sampling_rate)

# Calculate the total number of segments
num_segments = len(audio_data) // segment_samples

# Initialize an empty array to store the reconstructed audio
reconstructed_audio = np.zeros(len(audio_data))

# Initialize a list to store the dominant frequencies and amplitudes for each segment
dominant_frequencies = []
dominant_amplitudes = []

# Iterate over each segment
for i in range(num_segments):
    # Extract the current segment
    segment = audio_data[i * segment_samples: (i + 1) * segment_samples]

    # Perform the FFT on the segment
    fft_data = np.fft.fft(segment)
    magnitude = np.abs(fft_data)

    # Find the indices of the positive frequencies (excluding DC component)
    positive_indices = np.where(np.fft.fftfreq(segment_samples, 1 / sampling_rate) > 0)[0]

    # Find the indices of the 8 most dominant positive frequencies
    num_freqs = 8
    dominant_indices = positive_indices[np.argpartition(magnitude[positive_indices], -num_freqs)[-num_freqs:]]

    # Get the dominant frequencies and amplitudes
    segment_frequencies = np.fft.fftfreq(segment_samples, 1 / sampling_rate)
    dominant_freqs = segment_frequencies[dominant_indices]
    dominant_amps = magnitude[dominant_indices]

    # Append the dominant frequencies and amplitudes to the lists
    dominant_frequencies.append(dominant_freqs)
    dominant_amplitudes.append(np.round(dominant_amps, decimals=1))

    # Create a mask to zero out the non-dominant frequencies
    mask = np.zeros_like(fft_data)
    mask[dominant_indices] = fft_data[dominant_indices]

    # Perform the IFFT on the masked data
    reconstructed_segment = np.fft.ifft(mask).real

    # Add the reconstructed segment to the output array
    start_idx = i * segment_samples
    end_idx = start_idx + segment_samples
    reconstructed_audio[start_idx:end_idx] += reconstructed_segment

# Normalize the amplitude of the reconstructed audio
reconstructed_audio = librosa.util.normalize(reconstructed_audio)

# Save the reconstructed audio as an MP3 file
output_path = 'out.mp3'
sf.write(output_path, reconstructed_audio, sampling_rate)

# Convert the lists of dominant frequencies and amplitudes to NumPy arrays
dominant_frequencies = np.array(dominant_frequencies)
dominant_amplitudes = np.array(dominant_amplitudes)

# Save the dominant frequencies and amplitudes to a CSV file
output_csv = 'dominant_frequencies.csv'
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(np.hstack((dominant_frequencies, dominant_amplitudes)))

print("Reconstructed audio saved as", output_path)
print("Dominant frequencies and amplitudes saved to", output_csv)

# Get the time axis
duration = len(audio_data) / sampling_rate
time = np.linspace(0, duration, len(audio_data))

# Plot the waveform
plt.figure(figsize=(10, 4))
plt.plot(time, audio_data, color='b')
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Waveform of the MP3 file")
plt.show()
