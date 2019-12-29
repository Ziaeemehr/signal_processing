import pylab as pl 
import numpy as np 




def IN_PYTHON():
	from scipy.fftpack import fft
	# Number of sample points
	N = 2**17
	# sample spacing
	T = 1.0 / N
	x = np.linspace(0.0, N*T, N)
	y = np.sin(50.4 * 2.0*np.pi*x) + 0.5*np.sin(80.3 * 2.0*np.pi*x)
	yf = fft(y)
	xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
	import matplotlib.pyplot as plt
	plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]),label='Py')
	plt.grid()
	# plt.show()


# signal = np.loadtxt("signal.txt")
# pl.plot(signal[:,0],signal[:,1])

pl.figure()
result = np.loadtxt("RESULT.txt")
pl.plot(result[:,0],result[:,1], marker='.',label="FFTW")
# pl.xlim(80,80.01)
# IN_PYTHON()
pl.savefig('f.png')
pl.legend()
pl.show()