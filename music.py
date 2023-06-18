import librosa
import numpy as np
import soundfile as sf
import csv

# Load the original audio file
original_audio_path = 'pour_elise.mp3'
original_audio_data, sampling_rate = librosa.load(original_audio_path)

# Define frequency ranges and corresponding segment lengths
frequency_ranges = [
    (0, 250, 0.5),       # Range 1: 0-250 Hz, segment length = 0.5 seconds
    (250, 500, 0.25),    # Range 2: 250-500 Hz, segment length = 0.25 seconds
    (500, 1000, 0.1),    # Range 3: 500-1000 Hz, segment length = 0.1 seconds
    (1000, 2000, 0.05)   # Range 4: 1000-2000 Hz, segment length = 0.05 seconds
]

# Initialize an empty array to store the reconstructed audio
reconstructed_audio = np.zeros(len(original_audio_data))

# Initialize a list to store the frequencies and their amplitudes
segment_frequencies = []

# Iterate over each frequency range
for freq_range in frequency_ranges:
    # Get the frequency range and segment length
    freq_start, freq_end, segment_length = freq_range

    # Convert segment length to segment samples
    segment_samples = int(segment_length * sampling_rate)

    # Calculate the total number of segments for the current frequency range
    num_segments = len(original_audio_data) // segment_samples

    # Iterate over each segment within the current frequency range
    for i in range(num_segments):
        # Extract the current segment
        segment = original_audio_data[i * segment_samples: (i + 1) * segment_samples]

        # Perform the FFT on the segment
        fft_data = np.fft.fft(segment)
        magnitude = np.abs(fft_data)

        # Initialize an empty array to store the dominant indices
        dominant_indices = []

        # Get the frequency range indices
        freq_indices = np.where(
            np.logical_and(
                np.fft.fftfreq(segment_samples, 1 / sampling_rate) >= freq_start,
                np.fft.fftfreq(segment_samples, 1 / sampling_rate) < freq_end
            )
        )[0]

        # Find the indices of the 2 most dominant frequencies within the frequency range
        num_freqs = 2
        dominant_indices.extend(
            freq_indices[np.argpartition(magnitude[freq_indices], -num_freqs)[-num_freqs:]]
        )

        # Create a mask to zero out the non-dominant frequencies
        mask = np.zeros_like(fft_data)
        mask[dominant_indices] = fft_data[dominant_indices]

        # Perform the IFFT on the masked data
        reconstructed_segment = np.fft.ifft(mask).real

        # Add the reconstructed segment to the output array
        start_idx = i * segment_samples
        end_idx = start_idx + segment_samples
        reconstructed_audio[start_idx:end_idx] += reconstructed_segment

        # Get the frequencies and their amplitudes within the dominant indices
        segment_freq_amp = []
        for idx in dominant_indices:
            freq = np.fft.fftfreq(segment_samples, 1 / sampling_rate)[idx]
            amp = magnitude[idx]
            segment_freq_amp.extend([freq, amp])

        # Pad with zeros if there are less than 8 frequencies and amplitudes
        num_missing = 8 - len(segment_freq_amp)
        segment_freq_amp.extend([0] * num_missing)

        # Append the segment frequencies and amplitudes to the list
        segment_frequencies.append(segment_freq_amp)

# Save the reconstructed audio as an MP3 file
output_path = 'out.mp3'
sf.write(output_path, reconstructed_audio, sampling_rate)

# Create a CSV file to store the frequencies and amplitudes
csv_path = 'frequencies.csv'
with open(csv_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow([
        'Frequency 1', 'Amplitude 1', 'Frequency 2', 'Amplitude 2',
        'Frequency 3', 'Amplitude 3', 'Frequency 4', 'Amplitude 4',
        'Frequency 5', 'Amplitude 5', 'Frequency 6', 'Amplitude 6',
        'Frequency 7', 'Amplitude 7', 'Frequency 8', 'Amplitude 8'
    ])
    for freq_amp in segment_frequencies:
        writer.writerow(freq_amp)

# Print the completion message
print("Audio decomposition and reconstruction completed.")
# Calculate the quadratic error between the original audio and reconstructed audio
quadratic_error = np.mean((original_audio_data - reconstructed_audio) ** 2)

# Print the quadratic error
print("Quadratic Error:", quadratic_error)

