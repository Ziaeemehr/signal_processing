import scipy.io as sio
import pylab as pl
from sys import exit
from scipy.signal import welch, filtfilt
from scipy.signal import butter, hilbert
import numpy as np

# ----------------------------------------------#


def plot_lfp(time, lfp, ar_spikes):

    ax1 = fig1.add_subplot(211)
    ax2 = fig1.add_subplot(212)

    f, PWelch_spec = welch(lfp, fs=sf)

    ax1.plot(time, lfp)
    ax1.plot(ar_spikes, [0]*len(ar_spikes), 'go')
    ax1.set_xlabel("Time(ms)")
    ax1.set_ylabel(r"$LFP(\mu V)$")

    ax2.set_yscale('log')
    ax2.plot(f, PWelch_spec)
    pl.savefig(adrs+"fig-"+str(i)+'.png')
    pl.clf()
# ----------------------------------------------#


def average_lfp(time, lfp, ar_spikes):
    f, PWelch_spec = welch(lfp, fs=sf)
    return f, PWelch_spec
# ----------------------------------------------#


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def filtering(ar_signal, sampling_rate, lowcut, highcut, order=5):
    b, a = butter_bandpass(lowcut, highcut, sampling_rate, order)

    return filtfilt(b, a, ar_signal)
# ----------------------------------------------#


adrs = "./fig/lfp/"

spikes = sio.loadmat('spikes.mat')
spikes = spikes['spike_cell'][:, 0]

lfp = sio.loadmat('lfps.mat')  # , squeez_me=True
time = lfp['time'][0]
lfpmat = lfp['lfp_matrix']
sf = lfp['sf'][0][0]

# f , PWelch_spec = welch(lfpmat[0,:] ,fs=sf)


# fig100 = pl.figure(100, figsize=(10,5))
# fig, ax1 = pl.subplots(1, figsize=(15,5))
# ax2 = ax1.twinx()
P_ave = 0.0
num_trial = 40
list_phase_spikes = []
# for i in range(num_trial):

# plot_lfp(time, lfpmat[i,:], spikes[i][0])
# f, P = average_lfp(time, lfpmat[i,:], spikes[i][0])
# P_ave += P

filtered = filtering(lfpmat[i,:], sf, 10, 20, 5)
H_fileterd = hilbert(filtered)
# pl.plot(time, filtered,label='filter')
# pl.plot(time, lfpmat[i,:], label="Hifbert-transform")
H_norm = np.abs(H_fileterd)
H_angle = np.angle(H_fileterd)

# indx_spike = ((spikes[i][0]-time[0])/1000*sf).astype(int)

# phase_spikes = H_angle[indx_spike].tolist()

# list_phase_spikes +=  phase_spikes

# pl.plot(time, H_norm, label='norm_hilbert')
# pl.plot(time, H_angle, label='H_angle', c="k")

# pl.legend()
# pl.ylim(-15,15)

# pl.savefig("./fig/f-"+str(i)+".png")
# pl.clf()

# pl.show()
# l = np.asarray(list_phase_spikes)
# n = len(list_phase_spikes)
# plv = 1./float(n)* np.abs(np.sum(np.exp(1j*l)))
# print "PLV = %g" % plv
# pl.figure()
# hist, bins = np.histogram(list_phase_spikes, bins=10)
# width = (bins[1] - bins[0])
# center = (bins[:-1] + bins[1:]) / 2
# pl.bar(center, hist, align='center', width=width)
# pl.show()


exit(0)
P_ave /= float(num_trial)
fig2 = pl.figure()
pl.yscale('log')
pl.plot(f, P_ave)
pl.ylabel('Power A.U')
pl.xlabel("frequency(HZ)")
pl.savefig('./fig/ave.png')

# pl.show()


# print spikes[0][0]
# print spikes
# print len(spikes[0,0][0])
# print spikes[0,0][0]

# exit(0)

# print sf, len(time[0]), len(lfpmat)
# print type(lfpmat), lfpmat.shape


# pl.plot(PWelch_spec)
# pl.show()

# exit(0)
# print sf, len(time), lfpmat.shape
# print len(spikes), type(spikes), spikes.shape
