#!/usr/bin/env python

#sample point is crucial to determine the frequency
# the higher frequencies, the more sample point is needed
# min dt is 0.005 for 50 Hz

import numpy as np 
import pylab as pl 
from scipy.integrate import odeint
from scipy.fftpack import fft, fftshift
from numpy import sin

def euler(t,ic,omega):
    
    nstep = len(t)
    x = np.zeros(nstep)
    x[0] = ic
    dt = t[1]-t[0]
    
    for i in range(nstep-1):
        x[i+1]=x[i] + omega * dt
    return x

def find_frequencies(x,y):
    N = len(x)
    dt = x[2]-x[1]
    T = N*dt
    t = np.linspace(0.0, T, N)
    f = np.linspace(-1./(2.0*dt),1./(2.0*dt),N)
    fy = fft(y)/float(N)
    fy = fftshift(fy)
    yfplot = np.abs(fy)
    return f, yfplot

N = 10001 # sampling points
t = np.linspace(0, 50, N)
y0 = 0.5
nu = 52.2
omega = 2.0*np.pi*nu 
sol = euler(t,y0,omega)
print sol
a,b = find_frequencies(t,np.exp(sol*1j))
pl.plot(a,b)

pl.show()





