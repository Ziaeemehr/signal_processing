
'''
Fourier Transforms 
'''

from scipy.fftpack import fft, fftfreq, fftshift, ifft
import numpy as np 
import pylab as pl
from sys import exit

N = 2**10         # Number of sample points
dt = 1.0/800.0    # sample spacing
T = N*dt
t = np.linspace(0.0,T,N)
f = np.linspace(-1./(2.0*dt),1./(2.0*dt),N)
nu1 = 50
nu2 = 57
y = np.exp(2j*np.pi*nu1*t) + np.exp(2j*np.pi*(nu2)*t)
y = y + np.random.normal(0,1,size=N)
fig,(ax1,ax2) = pl.subplots(2)
ax1.plot(t[:100],np.real(y[:100]))
fy = fft(y)/float(N)
fy = fftshift(fy)
ax2.plot(f,np.abs(fy))
tmp = np.abs(fy)
thr = 0.5
print f[tmp>thr]
print tmp[tmp>thr]
# ax2.set_xlim(0,5)
# pl.savefig('f.png')
pl.show()