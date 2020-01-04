'''
Estimate power spectral density using a periodogram
'''

from scipy import signal
import numpy as np 
import pylab as pl 
'''
Generate a test signal, a 2 Vrms sine wave at 1234 Hz, corrupted by 0.001 V**2/Hz of white noise sampled at 10 kHz.
'''
fs = 10e3
N = 1e5
amp = 2*np.sqrt(2)
freq = 1234.0
noise_power = 0.001 * fs / 2
time = np.arange(N) / fs
print "time min %.4f max %.4f " %(min(time), max(time))
x = amp*np.sin(2*np.pi*freq*time)
x += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
f, Pxx_den = signal.periodogram(x,fs)
print type(f)
Pxx_deni = Pxx_den[Pxx_den>0.01]
fi = f[Pxx_den>0.01]
print len(Pxx_deni), len(fi)
for i in range(len(fi)):
    print fi[i], Pxx_deni[i]

pl.plot(f, Pxx_den)
pl.ylim([1e-7, 1e2])
pl.xlabel('frequency [Hz]')
pl.ylabel('PSD [V**2/Hz]')
pl.savefig('f')