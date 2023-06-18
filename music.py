import librosa
import numpy as np
import soundfile as sf
import csv

# Load the original audio file
original_audio_path = 'pour_elise.mp3'
original_audio_data, sampling_rate = librosa.load(original_audio_path)

# Define frequency ranges
frequency_ranges = [
    (0, 250),    # Range 1: 0-250 Hz
    (250, 500),  # Range 2: 250-500 Hz
    (500, 1000), # Range 3: 500-1000 Hz
    (1000, 2000) # Range 4: 1000-2000 Hz
]

# Choose the segment length (e.g., 100 ms)
segment_length = 0.1

# Convert segment length to segment samples
segment_samples = int(segment_length * sampling_rate)

# Calculate the total number of segments
num_segments = len(original_audio_data) // segment_samples

# Initialize an empty array to store the reconstructed audio
reconstructed_audio = np.zeros(len(original_audio_data))

# Initialize a list to store the frequencies and their amplitudes
segment_frequencies = []

# Iterate over each segment
for i in range(num_segments):
    # Extract the current segment
    segment = original_audio_data[i * segment_samples: (i + 1) * segment_samples]

    # Perform the FFT on the segment
    fft_data = np.fft.fft(segment)
    magnitude = np.abs(fft_data)

    # Initialize an empty array to store the dominant indices
    dominant_indices = []

    # Iterate over the frequency ranges
    for freq_range in frequency_ranges:
        # Get the frequency range indices
        freq_indices = np.where(
            np.logical_and(
                np.fft.fftfreq(segment_samples, 1 / sampling_rate) >= freq_range[0],
                np.fft.fftfreq(segment_samples, 1 / sampling_rate) < freq_range[1]
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

    # Add the frequencies and amplitudes to the list
    segment_frequencies.append(segment_freq_amp)

# Normalize the reconstructed audio
reconstructed_audio = librosa.util.normalize(reconstructed_audio)

# Calculate the quadratic error between the original and reconstructed audio
error = np.mean(np.square(original_audio_data - reconstructed_audio))

# Save the frequencies and amplitudes to a CSV file
csv_filename = 'frequencies.csv'
with open(csv_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Frequency 1', 'Amplitude 1', 'Frequency 2', 'Amplitude 2',
                     'Frequency 3', 'Amplitude 3', 'Frequency 4', 'Amplitude 4',
                     'Frequency 5', 'Amplitude 5', 'Frequency 6', 'Amplitude 6',
                     'Frequency 7', 'Amplitude 7', 'Frequency 8', 'Amplitude 8'])
    for freq_amp in segment_frequencies:
        freq1, amp1, freq2, amp2, freq3, amp3, freq4, amp4, freq5, amp5, \
        freq6, amp6, freq7, amp7, freq8, amp8 = freq_amp[:16]
        writer.writerow([freq1, amp1, freq2, amp2, freq3, amp3, freq4, amp4,
                         freq5, amp5, freq6, amp6, freq7, amp7, freq8, amp8])

# Save the reconstructed audio as an MP3 file
output_path = 'out.mp3'
sf.write(output_path, reconstructed_audio, sampling_rate)

# Print the error
print(f"Quadratic Error: {error}")
