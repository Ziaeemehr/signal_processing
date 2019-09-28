
'''
Fourier Transforms 
'''

from scipy.fftpack import fft, fftfreq, fftshift, ifft

import numpy as np 
import pylab as pl

# One dimensional discrete Fourier transforms
x =  np.array([1.0, 2.0, 1.0, -1.0, 1.5])
y = fft(x)
# print y
yinv = ifft(y)
# print yinv 
# print np.sum(x)


f,(ax1,ax2) = pl.subplots(2)

# The example plots the FFT of the sum of two sines.
N = 400         # Number of sample points
T = 1.0/800.0   # sample spacing
x = np.linspace(0.0,N*T,N)
nu1 = 50.0
nu2 = 63.0

y = np.sin(nu1 * 2.0*np.pi*x) + np.sin(nu2 * 2.0*np.pi*x)
#y = np.exp(50.0 * 1.j * 2.0*np.pi*x) + 0.5*np.exp(-80.0 * 1.j * 2.0*np.pi*x)
yf = fft(y)
xf = np.linspace(0.0,1./(2.0*T),N//2)
#xf = np.linspace(-1./(2.0*T),1./(2.0*T),N)
ax1.plot(xf,2./N*np.abs(yf[0:N//2]))
#ax1.plot(xf,1./N*np.abs(yf))
ax1.set_xlim(-200,200)


# using fftshift
N = 400         # Number of sample points
T = 1.0/800.0   # sample spacing
x = np.linspace(0.0,N*T,N)
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
#y = np.exp(50.0 * 1.j * 2.0*np.pi*x) + 0.5*np.exp(-80.0 * 1.j * 2.0*np.pi*x)
yf = fft(y)
xf = fftfreq(N, T)
xf = fftshift(xf)
yplot = fftshift(yf)
ax2.plot(xf,1./N * np.abs(yplot))
ax2.set_xlim(-200,200)
ax2.set_ylim(0,1)
pl.show()
