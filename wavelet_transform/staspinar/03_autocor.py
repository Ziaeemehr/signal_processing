# http://ataspinar.com/2018/04/04/machine-learning-with-signal-processing-techniques/
from __future__ import division
import numpy as np
import pylab as plt
from scipy.fftpack import fft
from sys import exit


def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result)//2:]

def get_autocorr_values(y_values, T, N):
    autocorr_values = autocorr(y_values)
    x_values = np.array([T * j for j in range(0, N)])
    return x_values, autocorr_values

def get_fft_values(y_values, T, N, f_s):
    from scipy.fftpack import fft
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values

t_n = 10
N = 1000
T = t_n / N
f_s = 1/T

x_value = np.linspace(0, t_n, N)
amplitudes = [4, 6, 8, 10, 14]
frequencies = [6.5, 5, 3, 1.5, 1]

y_values = [amplitudes[ii]*np.sin(2*np.pi*frequencies[ii]*x_value)
            for ii in range(0, len(amplitudes))]

composite_y_value = np.sum(y_values, axis=0)

t_values, autocorr_values = get_autocorr_values(composite_y_value, T, N)

f_values, fft_values = get_fft_values(autocorr_values, T, N, f_s)


fig, ax = plt.subplots(2, figsize=(8,6))
ax[0].plot(t_values, autocorr_values, linestyle='-', color='blue')
ax[0].set_xlabel('time delay [s]')
ax[0].set_ylabel('Autocorrelation amplitude')

ax[1].plot(f_values, fft_values, linestyle='-', color='blue')
ax[1].set_xlabel('Frequency [Hz]', fontsize=16)
ax[1].set_ylabel('Amplitude', fontsize=16)
# ax[1].set_title("Frequency domain of the signal", fontsize=16)

plt.savefig("fig/03")



# Fun fact: the auto-correlation and the PSD are Fourier Transform pairs, 
# i.e. the PSD can be calculated by taking the FFT of the auto-correlation
# function, and the auto-correlation can be calculated by taking the 
# Inverse Fourier Transform of the PSD function.