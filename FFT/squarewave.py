import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy import signal
import numpy as np

def find_frequency(x,y):
    '''
    # N Number of samplepoints
    # dt sample spacing
    '''
    N = len(x)
    dt = x[2]-x[1]
    xf = np.linspace(0.0, 1.0/(2.0*dt), N/2)
    yf = fft(y)
    yfplot = 2.0/N * np.abs(yf[0:N//2])
    axarr[0].plot(xf, yfplot)
    ind = find_peaks(yfplot)
    # print ind 
    return xf[ind], yfplot[ind]

def find_peaks(x):
    p = []
    for i in range(1,len(x)-1):
        if ( (x[i]>x[i-1]) & (x[i]>x[i+1]) ):
            p.append(i)
    return p    
# --------------------------------------------------------#

x = np.linspace(0, 1, 500, endpoint=False)
y = signal.square(2 * np.pi * 6 * x)
f, axarr = plt.subplots(2)
freq, amp = find_frequency(x,y)
threshold = 0.05
print freq[amp>threshold]
print amp[amp>threshold]
print len(amp[amp>threshold])
axarr[1].plot(x, y)
plt.ylim(-2, 2)
plt.show()
