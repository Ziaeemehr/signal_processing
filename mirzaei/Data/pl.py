import scipy.io as sio
import pylab as pl
from sys import exit
from scipy.signal import welch
from scipy.signal import filtfilt, butter
import numpy as np
# , butter, butterord, lfilterm filtfilt

adrs = "./fig/"


def filter_signal(signal, sf, f1, f2, order=5):
    (b, a) = butter(order, signal, np.array([f1, f2])/(sf/2), btype='bandpass')
    filterd_signal = filtfilt(b, a, signal)

    return filterd_signal


spikes = sio.loadmat('spikes.mat')
spikes = spikes['spike_cell'][:, 0]

lfp = sio.loadmat('lfps.mat')  # , squeez_me=True
time = lfp['time'][0]
lfpmat = lfp['lfp_matrix']
sf = lfp['sf'][0][0]

num_trial = 5
f, PWelch_spec = welch(lfpmat[0, :], fs=sf)

fig1 = pl.figure()
pl.yscale('log')

for i in range(num_trial):
    filterd_signal = filter_signal(lfpmat[i, :], sf, 10, 20)
    # pl.plot(f, filterd_signal)
    # pl.savefig(adrs+"flt/f-"+str(i)+".png")
    # pl.clf()


exit(0)
P_ave = 0.0
num_trial = 5
for i in range(num_trial):

    fig, (ax1, ax2) = pl.subplots(2, figsize=(8, 8))

    spike = spikes[i][0]

    f, PWelch_spec = welch(lfpmat[i, :], fs=sf)

    P_ave += PWelch_spec

    ax1.plot(time, lfpmat[i, :])
    ax1.plot(spike, [0]*len(spike), 'go')
    ax1.set_xlabel("Time(ms)")
    ax1.set_ylabel(r"$LFP(\mu V)$")

    ax2.set_yscale('log')
    ax2.plot(f, PWelch_spec)
    pl.savefig("./fig/fig-"+str(i)+'.png')
    pl.close()


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
