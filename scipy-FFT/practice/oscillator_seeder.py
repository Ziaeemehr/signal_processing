#!/usr/bin/env python

#sample point is crucial to determine the frequency
# the higher frequencies, the more sample point is needed
# min dt is 0.005 for 50 Hz

import numpy as np 
import pylab as pl 
from scipy.integrate import odeint
from scipy.fftpack import fft, fftshift
from numpy import sin

def osc(ic, t, omega):
    x = ic
    dydt = [0,0,0]
    for i in range(3):
        if i==0:
            dydt[i] = omega[i] #- sin(x[i]) 
        else:
            dydt[i] = omega[i] + np.sum(sin(x-x[i]))
    return dydt

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
y0 = [1.0,0.5,1.0]
nu =np.array([57.1,51.3,52.2])
omega = 2.0*np.pi*nu 
sol = odeint(osc, y0, t, args=(omega,))
# adding some noise
#sol = sol + np.random.normal(0,1,size=(N,2))

#for i in range(3):
    # a,b = find_frequencies(t,np.cos(sol[:,i]))
    #a,b = find_frequencies(t,np.exp(sol[:,i]*1j))
    #pl.plot(a,b)
a,b = find_frequencies(t,np.exp(sol[:,1]*1j))
pl.plot(a,b)

pl.show()





