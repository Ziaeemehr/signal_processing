# /usr/bin/env python3

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html

import numpy as np
from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt

# Generate a test signal, a 2 Vrms sine wave whose frequency
# is slowly modulated around 3kHz, corrupted by white noise
# of exponentially decreasing magnitude sampled at 10 kHz.

fs = 1e4
N = 1e5
amp = 2 * np.sqrt(2)
noise_power = 0.01 * fs / 2
time = np.arange(N) / float(fs)
mod = 500 * np.cos(2 * np.pi * 0.25 * time)
carrier = amp * np.sin(2 * np.pi * 3e3 * time + mod)
noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
noise *= np.exp(-time / 5)
x = carrier + noise

fig, ax = plt.subplots(2, figsize=(8, 7))

f, t, Sxx = signal.spectrogram(x, fs)
ax[0].pcolormesh(t, f, Sxx)
ax[0].set_ylabel('Frequency [Hz]')
ax[0].set_xlabel('Time [sec]')

# Note, if using output that is not one sided, then use the following:
f, t, Sxx = signal.spectrogram(x, fs, return_onesided=False)
ax[1].pcolormesh(t, fftshift(f), fftshift(Sxx, axes=0))
ax[1].set_ylabel('Frequency [Hz]')
ax[1].set_xlabel('Time [sec]')
plt.savefig("fig.png")
plt.show()