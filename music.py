import librosa
import numpy as np
import soundfile as sf

# Load the audio file
audio_path = 'pour_elise.mp3'
audio_data, sampling_rate = librosa.load(audio_path)

# Segment length in seconds (10ms in this case)
segment_length = 0.1
segment_samples = int(segment_length * sampling_rate)

# Calculate the total number of segments
num_segments = len(audio_data) // segment_samples

# Initialize an empty array to store the reconstructed audio
reconstructed_audio = np.zeros(len(audio_data))

# Iterate over each segment
for i in range(num_segments):
    # Extract the current segment
    segment = audio_data[i * segment_samples : (i+1) * segment_samples]

    # Perform the FFT on the segment
    fft_data = np.fft.fft(segment)
    magnitude = np.abs(fft_data)

    # Find the indices of the 8 most dominant frequencies
    num_freqs = 8
    dominant_indices = np.argpartition(magnitude, -num_freqs)[-num_freqs:]

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

print("Reconstructed audio saved as", output_path)
