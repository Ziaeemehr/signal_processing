from __future__ import division
from scipy.signal import chirp, spectrogram
import numpy as np
import pylab as pl


fig, ax = pl.subplots(1, 2, figsize=(8, 4))

t = np.linspace(0, 10, 5001)
w = chirp(t, f0=6, f1=1, t1=10, method='linear')
ax[0].plot(t, w)
ax[0].set_title("Linear Chirp, f(0)=6, f(10)=1")
ax[0].set_xlabel('t (sec)')

fs = 8000
T = 10
t = np.linspace(0, T, T * fs, endpoint=False)
# Quadratic chirp from 1500 Hz to 250 Hz over 10 seconds
# (vertex of the parabolic curve of the frequency is at t=0):

w = chirp(t, f0=1500, f1=250, t1=10, method='quadratic')

ff, tt, Sxx = spectrogram(w, fs=fs, noverlap=256, nperseg=512,
                          nfft=2048)
ax[1].pcolormesh(tt, ff[:513], Sxx[:513], cmap='gray_r')
ax[1].set_title('Quadratic Chirp, f(0)=1500, f(10)=250')
ax[1].set_xlabel('t (sec)')
ax[1].set_ylabel('Frequency (Hz)')
ax[1].grid()

pl.tight_layout()
pl.savefig("03.png")
# pl.show()
