import numpy as np
import numpy.typing as npt
import scipy.signal

def fft_real_polar(in_real: npt.NDArray):
    fft_out = np.fft.fft(in_real)
    return np.abs(fft_out), np.angle(fft_out)

def ifft_real_polar(in_mag: npt.NDArray, in_phase: npt.NDArray):
    in_complex = in_mag * np.exp(1j * in_phase)
    return np.real(np.fft.ifft(in_complex))

def wrap(data: npt.NDArray, range: float):
    data = np.fmod(data + range, range * 2)
    offset = range * ((data < 0).astype(data.dtype) * 2 - 1)
    return data + offset

def resample(data: npt.NDArray, new_length: int):
    return scipy.signal.resample(data, new_length)

if __name__ == "__main__":
    test = np.arange(8, dtype=np.float32) - 4
    print(test)
    print(wrap(test, 2))