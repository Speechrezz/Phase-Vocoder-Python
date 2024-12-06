from scipy.io import wavfile
import numpy as np
import numpy.typing as npt
from typing import Tuple

def audio_read(filename: str) -> Tuple[int, npt.NDArray[np.float32]]:
    audio_file = wavfile.read(filename)

    signal = audio_file[1]
    signal = signal.astype(np.float32)

    # Normalise float32 array so that values are between -1.0 and +1.0
    max_int16 = 2**15
    signal = signal / max_int16

    sample_rate = audio_file[0]
    return (sample_rate, signal)

def audio_read_mono(filename: str) -> Tuple[int, npt.NDArray[np.float32]]:
    sample_rate, signal = audio_read(filename)

    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)
    
    return (sample_rate, signal)

def audio_write(filename: str, sample_rate: int, signal):
    # Ensure gain never goes above 1
    signal_max = np.max(np.abs(signal)) + 1e-5
    if signal_max > 1:
        signal = signal / signal_max

    # De-normalize float32 array
    max_int16 = 2**15
    signal = signal * max_int16
    signal = signal.astype(np.int16)

    wavfile.write(filename, sample_rate, signal)