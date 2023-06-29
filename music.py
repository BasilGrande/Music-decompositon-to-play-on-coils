import struct
import librosa
import numpy as np
import soundfile as sf

# Load the original audio file
original_audio_path = '7.mp3'
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

# Save the reconstructed audio as an MP3 file
output_path = 'out.mp3'
sf.write(output_path, reconstructed_audio, sampling_rate)

# Calculate the quadratic error between the original audio and reconstructed audio
quadratic_error = np.mean((original_audio_data - reconstructed_audio) ** 2)

# Print the quadratic error
print("Quadratic Error:", quadratic_error)


def normalize_amplitude(amplitude, max_amplitude):
    max_value = np.max(amplitude)
    scale_factor = max_amplitude / max_value
    normalized_amplitude = amplitude * scale_factor
    return normalized_amplitude

def get_normalized_amplitude_array(mp3_file, target_sr, max_amplitude):
    audio, sr = librosa.load(mp3_file, sr=None)
    audio_resampled = librosa.resample(audio, sr, target_sr)
    amplitude = librosa.amplitude_to_db(audio_resampled, ref=np.max)
    normalized_amplitude = normalize_amplitude(amplitude, max_amplitude)
    return normalized_amplitude

def write_amplitude_to_file(amplitude_array, output_file):
    with open(output_file, 'wb') as file:
        for sample in amplitude_array:
            byte = struct.pack('b', int(sample))
            file.write(byte)

# Example usage
mp3_file = 'out.mp3'
target_sr = 5000
max_amplitude = 64
output_file = 'song.bin'

normalized_amplitude_array = get_normalized_amplitude_array(mp3_file, target_sr, max_amplitude)
write_amplitude_to_file(normalized_amplitude_array, output_file)
