import librosa
import numpy as np
import soundfile as sf

# Load the audio file
audio_path = 'pour_elise.mp3'
audio_data, sampling_rate = librosa.load(audio_path)

# Perform the FFT
fft_data = np.fft.fft(audio_data)
magnitude = np.abs(fft_data)

# Find the indices of the 8 most dominant frequencies
num_freqs = 8
dominant_indices = np.argpartition(magnitude, -num_freqs)[-num_freqs:]

# Create a mask to zero out the non-dominant frequencies
mask = np.zeros_like(fft_data)
mask[dominant_indices] = fft_data[dominant_indices]

# Perform the IFFT
reconstructed_audio = np.fft.ifft(mask)

# Convert the audio back to the time domain and rescale the amplitude
reconstructed_audio = librosa.util.normalize(np.real(reconstructed_audio))

# Save the reconstructed audio as an MP3 file
output_path = 'out.mp3'
sf.write(output_path, reconstructed_audio, sampling_rate)

print("Reconstructed audio saved as", output_path)
