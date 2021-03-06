import numpy as np
from scipy.signal import chirp, spectrogram
import matplotlib.pyplot as plt

from lspopt.lsp import spectrogram_lspopt

fs = 1e4
N = 1e5
amp = 2 * np.sqrt(2)
noise_power = 0.001 * fs / 2
time = np.arange(N) / fs
freq = np.linspace(1e3, 2e3, N)
x = amp * chirp(time, 1e3, 2.0, 6e3, method='quadratic') + \
    np.random.normal(scale=np.sqrt(noise_power), size=time.shape)

f, t, Sxx = spectrogram(x, fs)

ax = plt.subplot(211)
ax.pcolormesh(t, f, Sxx)
ax.set_ylabel('Frequency [Hz]')
ax.set_xlabel('Time [sec]')

f, t, Sxx = spectrogram_lspopt(x, fs, c_parameter=20.0)

ax = plt.subplot(212)
ax.pcolormesh(t, f, Sxx)
ax.set_ylabel('Frequency [Hz]')
ax.set_xlabel('Time [sec]')
plt.savefig("f.png")
# plt.show()