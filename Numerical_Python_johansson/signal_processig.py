import numpy as np
import pylab as pl
import pylab as plt
import pandas as pd
from numpy import pi
from scipy import io
from sys import exit
import scipy.io.wavfile
from scipy import signal
from scipy import fftpack


def signal_samples(t, nu0=1.0, nu1=22.0):
    return (2 * np.sin(2 * pi * t * nu0) + 3.0 * np.sin(2 * pi * t * nu1) +
            2 * np.random.randn(*np.shape(t)))


# ------------------------------------------------------------------#
B = 30.0 # max freqeuency to be measured.
f_s = 2 * B
delta_f = 0.01
N = int(f_s / delta_f)
T = N / f_s
t = np.linspace(0, T, N)
f_t = signal_samples(t)


def plot_1():
    fig, ax = pl.subplots(1, 2, figsize=(8, 3), sharey=True)
    ax[0].plot(t, f_t)
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("signal")
    ax[1].plot(t, f_t)
    ax[1].set_xlim(0, 5)
    ax[1].set_xlabel("Time (s)")
    fig.savefig("data/ch17-1-signal.png")
    fig.tight_layout()
    pl.close()

# ------------------------------------------------------------------#


F = fftpack.fft(f_t)
f = fftpack.fftfreq(N, 1.0 / f_s)
mask = np.where(f >= 0)

def plot_2():
    fig, ax = pl.subplots(3, 1, figsize=(8, 6))
    ax[0].plot(f[mask], np.log(abs(F[mask])), label="real")
    ax[0].plot(B, 0, 'r*', markersize=10)
    ax[0].set_xlim(0, 30)
    ax[0].set_ylabel("$\log(|F|)$", fontsize=14)

    ax[1].plot(f[mask], 2.0 * abs(F[mask])/N, label="real")
    ax[1].set_xlim(0, 2)
    ax[1].set_ylabel("2$|F|/N$", fontsize=14)

    ax[2].plot(f[mask], 2.0 * abs(F[mask])/N, label="real")
    ax[2].set_xlim(19, 23)
    ax[2].set_xlabel("frequency (Hz)", fontsize=14)
    ax[2].set_ylabel("2$|F|/N$", fontsize=14)
    for i in range(3):
        ax[i].legend()

    fig.tight_layout()
    fig.savefig("data/ch17-2-simulated-signal-spectrum.png")
    pl.close()

# plot_2()
# ------------------------------------------------------------------#
# Frequency-domain Filter


F_filtered = F * (abs(f) < 2)
f_t_filtered = fftpack.ifft(F_filtered)


def plot_3():
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, f_t, label='original', alpha=0.5)
    ax.plot(t, f_t_filtered.real, color="red", lw=3, label='filtered')
    ax.set_xlim(0, 10)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("signal")
    ax.legend()
    fig.tight_layout()
    fig.savefig("data/ch17-inverse-fft.png")
    pl.close()
# ------------------------------------------------------------------#

# Windowing


def plot_4():
    N = 100
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    ax.plot(signal.blackman(N), label="Blackman")
    ax.plot(signal.hann(N), label="Hann")
    ax.plot(signal.hamming(N), label="Hamming")
    ax.plot(signal.gaussian(N, N/5), label="Gaussian (std=N/5)")
    ax.plot(signal.kaiser(N, 7), label="Kaiser (beta=7)")
    ax.set_xlabel("n")
    ax.legend(loc=0)
    fig.tight_layout()
    fig.savefig("data/ch17-window-functions.png")
    pl.close()
# ------------------------------------------------------------------#


df = pd.read_csv('data/temperature_outdoor_2014.tsv',
                 delimiter="\t", names=["time", "temperature"])
df.time = pd.to_datetime(df.time.values, unit="s").tz_localize(
    'UTC').tz_convert('Europe/Stockholm')

df = df.set_index("time")
df = df.resample("1H").ffill()
df = df[(df.index >= "2014-04-01")*(df.index < "2014-06-01")].dropna()
time = df.index.astype('int') / 1e9

temperature = df.temperature.values
temperature_detrended = signal.detrend(temperature)
window = signal.blackman(len(temperature_detrended))
temperature_windowed = temperature * window
data_fft = fftpack.fft(temperature)
data_fft_detrended = fftpack.fft(temperature_detrended)
data_fft_windowed = fftpack.fft(temperature_windowed)


def plot_5():
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, temperature, label="original")
    ax.plot(df.index, temperature_detrended, label="detrended")
    ax.plot(df.index, temperature_windowed, label="windowed")
    ax.set_ylabel("temperature", fontsize=14)
    ax.legend(loc=0)
    fig.tight_layout()
    fig.savefig("data/ch17-temperature-signal.png")
    pl.close()
# ------------------------------------------------------------------#


f = fftpack.fftfreq(len(temperature_windowed), time[1]-time[0])
mask = f > 0


def plot_6():
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0.000001, 0.000025)
    # ax.set_xlim(0.000005, 0.000018)
    ax.set_xlim(0.000005, 0.00004)

    ax.axvline(1./86400, color='r', lw=0.5)
    ax.axvline(2./86400, color='r', lw=0.5)
    ax.axvline(3./86400, color='r', lw=0.5)
    ax.plot(f[mask], np.log(abs(data_fft[mask]) ** 2),
            lw=2, label="original")
    ax.plot(f[mask], np.log(abs(data_fft_detrended[mask])**2),
            lw=2, label="detrended")
    ax.plot(f[mask], np.log(abs(data_fft_windowed[mask])**2),
            lw=2, label="windowed")

    ax.set_ylabel("$\log|F|$", fontsize=14)
    ax.set_xlabel("frequency (Hz)", fontsize=14)
    ax.legend(loc=0)
    fig.tight_layout()
    fig.savefig("data/ch17-temperature-spectrum.png")
    pl.close()


# ------------------------------------------------------------------#
# Spectrogram of Guitar sound
sample_rate, data = io.wavfile.read("data/guitar.wav")

# print sample_rate
# print data.shape
data = data.mean(axis=1)
# print data.shape[0] / float(sample_rate)

N = int(sample_rate / 2.0)
# print N
f = fftpack.fftfreq(N, 1.0/sample_rate)
t = np.linspace(0, 0.5, N)
mask = (f > 0) * (f < 1000)
subdata = data[:N]
F = fftpack.fft(subdata)


def plot_7():
    fig, ax = plt.subplots(1, 2, figsize=(12, 3))
    ax[0].plot(t, subdata)
    ax[1].plot(f[mask], abs(F[mask]))

    ax[0].set_ylabel("signal", fontsize=14)
    ax[0].set_xlabel("time (s)", fontsize=14)
    ax[1].set_ylabel("$|F|$", fontsize=14)
    ax[1].set_xlabel("Frequency (Hz)", fontsize=14)
    ax[1].set_xlim(0, 1000)
    fig.tight_layout()
    fig.savefig("data/ch17-guitar-spectrum.png")
    pl.close()


# plot_7()
# ------------------------------------------------------------------#
N_max = int(data.shape[0] / N)
f_values = np.sum(1 * mask)
spect_data = np.zeros((N_max, f_values))
window = signal.blackman(len(subdata))

for n in range(N_max):
    subdata = data[(N * n):(N * (n + 1))]
    F = fftpack.fft(subdata * window)
    spect_data[n, :] = np.log(abs(F[mask]))


def plot_8():
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    p = ax.imshow(spect_data, origin='lower',
                  extent=(0, 1000, 0, data.shape[0] / sample_rate),
                  aspect='auto',
                  cmap=plt.cm.RdBu_r)
    cb = fig.colorbar(p, ax=ax)
    cb.set_label("$\log|F|$", fontsize=16)
    ax.set_ylabel("time (s)", fontsize=14)
    ax.set_xlabel("Frequency (Hz)", fontsize=14)
    fig.tight_layout()
    fig.savefig("data/ch17-spectrogram.png")
    pl.close()


# plot_8()
# ------------------------------------------------------------------#
# Signal filters
# Convolution filters

np.random.seed(0)
B = 30.0
f_s = 2 * B
delta_f = 0.01
N = int(f_s / delta_f)
T = N / f_s
t = np.linspace(0, T, N)
f_t = signal_samples(t)
f = fftpack.fftfreq(N, 1/f_s)

H = (abs(f) < 2)
h = fftpack.fftshift(fftpack.ifft(H))
f_t_filtered_conv = signal.convolve(f_t, h, mode='same')

F = fftpack.fft(f_t_filtered_conv.real)
f = fftpack.fftfreq(N, 1.0 / f_s)
mask = np.where(f >= 0)


def plot_9():

    fig = plt.figure(figsize=(8, 6))

    ax = plt.subplot2grid((2, 2), (0, 0))
    ax.plot(f, H)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("Frequency filter")
    ax.set_ylim(0, 1.5)

    ax = plt.subplot2grid((2, 2), (0, 1))
    ax.plot(t - 50, h.real)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("convolution kernel")

    ax = plt.subplot2grid((2, 2), (1, 0))  # , colspan=2)
    ax.plot(t, f_t, label='original', alpha=0.25)
    ax.plot(t, f_t_filtered.real, "r", lw=2,
            label='filtered in frequency domain')
    ax.plot(t, f_t_filtered_conv.real, 'b--', lw=2,
            label='filtered with convolution')
    ax.set_xlim(0, 10)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("signal")
    ax.legend(loc=2)

    ax = plt.subplot2grid((2, 2), (1, 1))  # , colspan=2)
    ax.plot(f[mask], np.log(abs(F[mask])), lw=2)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("log(|F|)")

    fig.tight_layout()
    fig.savefig("data/ch17-convolution-filter.png")


# plot_9()
# ------------------------------------------------------------------#
# FIR filter
n = 101
f_s = 1.0 / 3600
nyq = f_s / 2
b = signal.firwin(n, cutoff=nyq / 12.0, nyq=nyq, window="hamming")
# pl.plot(b)
# pl.show()

f, h = signal.freqz(b)


def plot_10():
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    h_ampl = 20 * np.log10(abs(h))
    h_phase = np.unwrap(np.angle(h))
    ax.plot(f/max(f), h_ampl, 'b')
    ax.set_ylim(-150, 5)
    ax.set_ylabel('frequency response (dB)', color="b")
    ax.set_xlabel(r'normalized frequency')
    ax = ax.twinx()
    ax.plot(f/max(f), h_phase, 'r')
    ax.set_ylabel('phase response', color="r")
    ax.axvline(1.0/12, color="black")
    fig.tight_layout()
    fig.savefig("data/ch17-filter-frequency-response.png")
    pl.close()


# plot_10()
# ------------------------------------------------------------------#

df = pd.read_csv('data/temperature_outdoor_2014.tsv',
                 delimiter="\t", names=["time", "temperature"])
df.time = pd.to_datetime(df.time.values, unit="s").tz_localize(
    'UTC').tz_convert('Europe/Stockholm')

df = df.set_index("time")
df = df.resample("1H").ffill()
df = df[(df.index >= "2014-01-16")*(df.index < "2014-05-22")].dropna()
temperature = df.temperature.values


temperature_filtered = signal.lfilter(b, 1, temperature)
temperature_median_filtered = signal.medfilt(temperature, 25)


def plot_11():
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df.index, temperature, label="original", alpha=0.5)
    ax.plot(df.index, temperature_filtered, color="green",
            lw=2, label="FIR")
    ax.plot(df.index, temperature_median_filtered,
            color="red", lw=2, label="median filer")
    ax.set_ylabel("temperature", fontsize=14)
    ax.legend(loc=0)
    fig.tight_layout()
    fig.savefig("data/ch17-temperature-signal-fir.png")
    pl.close()


# plot_11()


b, a = signal.butter(2, 7 / 365.0, btype="high")
# print b
# print a
temperature_iir = signal.lfilter(b, a, temperature)
temperature_filtfile = signal.filtfilt(b, a, temperature)


def plot_12():
    fig, ax = plt.subplots(figsize=(15, 3))
    ax.plot(df.index, temperature, label="original", alpha=0.5)
    ax.plot(df.index, temperature_iir, color='red', label="IIR filter")
    ax.plot(df.index, temperature_filtfile,
            color="green", label="filtfilt filtered")
    ax.set_ylabel("temperature", fontsize=14)
    ax.legend(loc=0)
    fig.savefig("data/ch17-temperature-filtfilt.pdf")
    pl.close()


plot_12()
