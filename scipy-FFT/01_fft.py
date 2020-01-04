import numpy as np
import pylab as pl
from math import pi
from scipy.fftpack import fft, fftshift
from scipy import fftpack


def find_frequencies(signal, dt):
    from scipy.fftpack import fft
    N = len(signal)
    xf = np.linspace(0.0, 1.0 / (2.0 * dt), N / 2)
    yf = fft(signal)
    yfplot = 2.0 / N * np.abs(yf[0:N // 2])
    return xf, yfplot

# def find_frequencies(signal, dt):
#     N = len(signal)
#     T = N*dt
#     t = np.linspace(0.0, T, N)
#     f = np.linspace(-1. / (2.0 * dt), 1. / (2.0 * dt), N)
#     fy = fft(signal) / float(N)
#     fy = fftshift(fy)
#     yfplot = np.abs(fy)
#     return f, yfplot

def findFrequencySpectrum(signal, fs):
    
    N = len(signal)
    F = fftpack.fft(signal)
    f = fftpack.fftfreq(N, 1.0 / fs)
    mask = np.where(f >= 0)

    fig, ax = pl.subplots(1, figsize=(5, 4))
    ax.plot(f[mask], 2.0 * abs(F[mask])/N, label="real")
    ax.set_ylabel("$|F|/N$", fontsize=14)

    ax.set_xlabel("frequency (Hz)", fontsize=14)
    ax.set_ylabel("$|F|/N$", fontsize=14)
    ax.legend()

    fig.tight_layout()
    fig.savefig("fig.png")
    pl.show()
    pl.close()




def signal_samples(t, nu0=1.0, nu1=22.0, amp0=2.0, amp1=3.0, ampNoise=1.0):
    return (amp0 * np.sin(2 * pi * t * nu0) + amp1 * np.sin(2 * pi * t * nu1) +
            ampNoise * np.random.randn(*np.shape(t)))


B = 30.0  # max freqeuency to be measured.
fs = 2 * B
delta_f = 0.01
N = int(fs / delta_f)
T = N / fs
t = np.linspace(0, T, N)
ft = signal_samples(t, nu0=1.5, amp0=3, nu1=22.1, amp1=1.0, ampNoise=1.0)

findFrequencySpectrum(ft, fs)
