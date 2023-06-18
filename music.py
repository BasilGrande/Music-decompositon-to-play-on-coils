import librosa
import numpy as np
import soundfile as sf
import csv
from tqdm import tqdm

# Load the original audio file
original_audio_path = 'pour_elise.mp3'
original_audio_data, sampling_rate = librosa.load(original_audio_path)

# Range of segment lengths to sweep over
segment_lengths = np.linspace(0.25, 0.001, num=20)

# Initialize variables to store the best iterations
best_errors = np.inf * np.ones(3)
best_segment_lengths = np.zeros(3)
best_reconstructed_audios = [None] * 3

# Create a progress bar for the iterations
pbar_iterations = tqdm(total=len(segment_lengths), desc='Iterations')

# Iterate over each segment length
for segment_length in segment_lengths:
    # Convert segment length to segment samples
    segment_samples = int(segment_length * sampling_rate)

    # Calculate the total number of segments
    num_segments = len(original_audio_data) // segment_samples

    # Initialize an empty array to store the reconstructed audio
    reconstructed_audio = np.zeros(len(original_audio_data))

    # Iterate over each segment
    for i in range(num_segments):
        # Extract the current segment
        segment = original_audio_data[i * segment_samples: (i + 1) * segment_samples]

        # Perform the FFT on the segment
        fft_data = np.fft.fft(segment)
        magnitude = np.abs(fft_data)

        # Find the indices of the positive frequencies (excluding DC component)
        positive_indices = np.where(np.fft.fftfreq(segment_samples, 1 / sampling_rate) > 0)[0]

        # Find the indices of the 8 most dominant positive frequencies
        num_freqs = 8
        dominant_indices = positive_indices[np.argpartition(magnitude[positive_indices], -num_freqs)[-num_freqs:]]

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

    # Calculate the quadratic error between the original and reconstructed audio
    error = np.mean(np.square(original_audio_data - reconstructed_audio))

    # Find the index of the current segment length if it is one of the best iterations
    min_error_idx = np.argmin(best_errors)
    if error < best_errors[min_error_idx]:
        # Update the best errors, segment lengths, and reconstructed audios arrays
        best_errors[min_error_idx] = error
        best_segment_lengths[min_error_idx] = segment_length
        best_reconstructed_audios[min_error_idx] = reconstructed_audio

    # Update the progress bar for iterations
    pbar_iterations.update(1)

# Close the progress bar for iterations
pbar_iterations.close()

# Save the best three reconstructed audios as MP3 files
for i in range(3):
    if best_reconstructed_audios[i] is not None:
        output_path = f'out_best_segment_{i+1}.mp3'
        sf.write(output_path, best_reconstructed_audios[i], sampling_rate)

        # Print the error for each best iteration
        print(f"Error for best iteration {i+1}: {best_errors[i]}")

# Print the best segment lengths
print("Best segment lengths:")
for length in best_segment_lengths:
    print(length)
