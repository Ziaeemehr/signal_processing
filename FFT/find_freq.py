def find_frequencies(self, x, y, I, fs, method='welch'):
        if method == 'FFT':
            from scipy.fftpack import fft
            N = len(x)
            dt0 = x[2] - x[1]
            xf = np.linspace(0.0, 1.0 / (2.0 * dt0), N / 2)
            yf = fft(y)
            yfplot = 2.0 / N * np.abs(yf[0:N // 2])
            # pl.plot(xf, yfplot, label=str(I))
            # pl.xlim(0, 0.2)
            return xf[1:], yfplot[1:]
        elif method == 'welch':
            from scipy.signal import welch
            f, pxx = welch(y, fs=fs, nperseg=len(y))
            # pl.semilogy(f, pxx)
            return f, pxx
        elif method == 'spike_interval':
            peak = find_peaks(y)  # index of peaks
            try:
                period = x[peak[-1]] - x[peak[len(peak) - 2]]
                freq = [1. / period]
                pxx = [1]
            except:
                print 'non oscillatory waveform or contains'
                print 'less that leat 3 periods of oscillation'
                freq = [0]
                pxx = [0]
            return freq, pxx