import numpy as np
import audio_io
import dsp

# Load test signal (little song I made)
file_name = "weird drum groove"
signal_path_in  = f"{file_name}.wav"
signal_path_out = f"{file_name} (pitched).wav"
sample_rate, signal_in = audio_io.audio_read_mono(signal_path_in)

# Phase Vocoder parameters
stretch_amount = 0.8
fft_size = 2048
analysis_size = fft_size // 4 # Input hop length

# Compute additional parameters
synthesis_size = round(analysis_size * stretch_amount) # Output hop length
hop_ratio = synthesis_size / analysis_size
gain_compensation = 2 * synthesis_size / fft_size
new_block_length = int(np.round(fft_size / stretch_amount))
print(f"\nSetup: {analysis_size=}, {synthesis_size=}, {hop_ratio=}, {gain_compensation=}, {new_block_length=}")

# Setup
signal_out = np.zeros([round(signal_in.shape[0] / stretch_amount) + fft_size], dtype=np.float32)
unwrap_data = 2 * np.pi * analysis_size / fft_size * np.arange(fft_size, dtype=np.float32)

block_phase_prev = np.zeros_like(unwrap_data)
block_phase_new = block_phase_prev

window = np.sqrt(np.hanning(fft_size + 1)[0:fft_size])

# Perform STFT (Short-Term Fourier Transform) with overlapping windows.
# This allows us to process the signal in the frequency domain.
for n in range(0, signal_in.shape[0] - 2 * fft_size, analysis_size):
    # Load data and perform a windowed FFT
    block = signal_in[n:n + fft_size] * window
    block_mag, block_phase = dsp.fft_real_polar(block)

    # Synthesis phase calculation
    unwrapped_phase = block_phase - block_phase_prev - unwrap_data
    unwrapped_phase = dsp.wrap(unwrapped_phase, np.pi)
    unwrapped_phase = hop_ratio * (unwrapped_phase + unwrap_data)

    # Compute new phase
    block_phase_new += unwrapped_phase
    block_phase_new = dsp.wrap(block_phase_new, np.pi)
    block_phase_prev = block_phase

    # Perform a windowed IFFT
    block = dsp.ifft_real_polar(block_mag, block_phase_new)
    block *= window * gain_compensation

    # Add processed data
    block = dsp.resample(block, new_block_length)
    signal_out[n:n + new_block_length] += block


# Save stretched signal
audio_io.audio_write(signal_path_out, sample_rate, signal_out)