# http://ataspinar.com/2018/04/04/machine-learning-with-signal-processing-techniques/
from __future__ import division
import numpy as np
import pylab as plt
from scipy.fftpack import fft
from sys import exit
from scipy import fftpack
from scipy import signal

import numpy as np
import pylab as plt
from scipy.fftpack import fft

import matplotlib as mpl
mpl.style.use('ggplot')

def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values


time_final = 1
numPoints = 100000
dt = time_final / float(numPoints)
fs = 1/dt

xa = np.linspace(0, time_final, num=numPoints)
xb = np.linspace(0, time_final/4, num=numPoints/4)

freq = [4, 30, 60, 90]
y1a, y1b = np.sin(2*np.pi*freq[0]*xa), np.sin(2*np.pi*freq[0]*xb)
y2a, y2b = np.sin(2*np.pi*freq[1]*xa), np.sin(2*np.pi*freq[1]*xb)
y3a, y3b = np.sin(2*np.pi*freq[2]*xa), np.sin(2*np.pi*freq[2]*xb)
y4a, y4b = np.sin(2*np.pi*freq[3]*xa), np.sin(2*np.pi*freq[3]*xb)

composite_signal1 = y1a + y2a + y3a + y4a
composite_signal2 = np.concatenate([y1b, y2b, y3b, y4b])

f_values1, fft_values1 = get_fft_values(composite_signal1, dt, numPoints, fs)
f_values2, fft_values2 = get_fft_values(composite_signal2, dt, numPoints, fs)

fig, axarr = plt.subplots(nrows=3, ncols=2, figsize=(12, 8))
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




interval = 0.05
n = int(fs * interval)
f = fftpack.fftfreq(n, 1.0 / fs)
t = np.linspace(0, interval, n)
mask = (f > 0) * (f < 100)
subdata = composite_signal1[:n]
F = fftpack.fft(subdata)


nMax = int(numPoints / n)
fvalues = np.sum(1 * mask)
spect_data = np.zeros((nMax, fvalues))
window = signal.blackman(len(subdata))


for i in range(nMax):
    subdata = composite_signal1[(n * i):(n * (i + 1))]
    F = fftpack.fft(subdata * window)
    spect_data[i, :] = np.log(abs(F[mask]))


p = axarr[2, 1].imshow(spect_data.T, origin='lower',
                  extent=(0, 100, 0, composite_signal1.shape[0] / fs),
                  aspect='auto',
                  cmap=plt.cm.RdBu_r)
cb = fig.colorbar(p, ax=axarr[2, 1])
cb.set_label("$\log|F|$", fontsize=16)
axarr[2, 1].set_ylabel("time (s)", fontsize=14)
axarr[2, 1].set_xlabel("Frequency (Hz)", fontsize=14)

plt.tight_layout()
plt.savefig("02.png")

# plt.show()
# plt.tight_layout()
