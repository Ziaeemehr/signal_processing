import matplotlib.pyplot as plt
from scipy.fftpack import fft
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
    plt.plot(xf, yfplot)
    # plt.xlim([0,2000])
    ind = find_peaks(yfplot)
    # print ind 
    return xf[ind], yfplot[ind]

def find_peaks(x):
    p = []
    for i in range(1,len(x)-1):
        if ( (x[i]>x[i-1]) & (x[i]>x[i+1]) ):
            p.append(i)
    return p    

T = 1./600.0
N = 6000
x = np.linspace(0.0, N*T, N)
y = np.sin(2.0 * 2.0*np.pi*x) + 0.5*np.sin(0.5 * 2.0*np.pi*x)
freq,amp = find_frequency(x,y)
for i in range(len(freq)):
    print freq[i], amp[i]

plt.show()


# for index, item in enumerate(yfplot):
#        if (item == max(yfplot)):
#           ind,it = index, item
# plt.savefig('./../data/'+str(s)+'.png')          

# select a proper interval
# yout = y2f[ (10<xf) & (xf<100) ]
# xout =  xf[ (10<xf) & (xf<100) ]
