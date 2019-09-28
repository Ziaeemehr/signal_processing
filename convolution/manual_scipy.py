import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

np.set_printoptions(precision=4, suppress=True)
np.random.seed(5)

def sigma2fwhm(sigma):
    return sigma * np.sqrt(8 * np.log(2))

def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))


FWHM = 4
sigma = fwhm2sigma(FWHM)

n_points = 60
x_vals = np.arange(n_points)
y_vals = np.random.normal(size=n_points)

# manual method
smoothed_vals = np.zeros(y_vals.shape)
for x_position in x_vals:
    kernel = np.exp(-(x_vals - x_position) ** 2 / (2 * sigma ** 2))
    kernel = kernel / sum(kernel)
    smoothed_vals[x_position] = sum(y_vals * kernel)
plt.bar(x_vals, smoothed_vals, alpha=0.2)
plt.plot(x_vals, y_vals, lw=2, label='original')
plt.plot(x_vals, smoothed_vals, lw=3, c='r', label='manual')


# using scipy
from scipy.ndimage.filters import gaussian_filter1d
filtered = gaussian_filter1d(y_vals, sigma, mode='reflect')
pl.plot(x_vals, filtered, marker='o', c='k', label='scipy')
print len(filtered)
pl.legend(frameon=False)
pl.savefig('f.png')
pl.show()
