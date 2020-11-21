# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html

import numpy as np
from scipy import signal
from scipy.fftpack import fftshift
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, figsize=(8, 7))

t = np.arange(0, 1, 0.002)
fs = 1.0 / (t[1] - t[0])

x = 4 * np.sin(2 * np.pi * 15 * t) + 2 * np.cos(2 * np.pi * 48 * t)
xn = x + np.random.randn(len(x))

ax[0].plot(t, xn)

f, t, Sxx = signal.spectrogram(xn, fs, noverlap=32,)
# spectrogram(xn, 64, 60, [], 500)
p = ax[1].pcolormesh(t, f, Sxx, cmap="afmhot")
cbar = fig.colorbar(p, ax=ax[1])
ax[1].set_ylabel('Frequency [Hz]')
ax[1].set_xlabel('Time [sec]')
ax[1].set_ylim(0, 60)
cbar.ax.set_ylabel("amplitude #")

plt.savefig("fig.png")
plt.show()
