# http://ataspinar.com/2018/04/04/machine-learning-with-signal-processing-techniques/
from __future__ import division
import numpy as np
import pylab as plt
from scipy.fftpack import fft
from sys import exit

import numpy as np
import pylab as plt
from scipy.fftpack import fft


def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values


t_n = 1
N = 100000
T = t_n / float(N)
fs = 1/T

xa = np.linspace(0, t_n, num=N)
xb = np.linspace(0, t_n/4, num=N/4)

freq = [4, 30, 60, 90]
y1a, y1b = np.sin(2*np.pi*freq[0]*xa), np.sin(2*np.pi*freq[0]*xb)
y2a, y2b = np.sin(2*np.pi*freq[1]*xa), np.sin(2*np.pi*freq[1]*xb)
y3a, y3b = np.sin(2*np.pi*freq[2]*xa), np.sin(2*np.pi*freq[2]*xb)
y4a, y4b = np.sin(2*np.pi*freq[3]*xa), np.sin(2*np.pi*freq[3]*xb)

composite_signal1 = y1a + y2a + y3a + y4a
composite_signal2 = np.concatenate([y1b, y2b, y3b, y4b])

f_values1, fft_values1 = get_fft_values(composite_signal1, T, N, fs)
f_values2, fft_values2 = get_fft_values(composite_signal2, T, N, fs)

fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
axarr[0, 0].plot(xa, composite_signal1)
axarr[1, 0].plot(xa, composite_signal2)
axarr[0, 1].plot(f_values1, fft_values1)
axarr[1, 1].plot(f_values2, fft_values2)
axarr[0, 1].set_xlim(0, 150)
axarr[1, 1].set_xlim(0, 150)
axarr[0, 1].set_xlabel("Frequency [Hz]")
axarr[1, 1].set_xlabel("Frequency [Hz]")
axarr[0, 0].set_xlabel("Time [S]")
axarr[1, 0].set_xlabel("Time [S]")
# plt.tight_layout()
plt.savefig("fig/04")
