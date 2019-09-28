import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

np.random.seed(5)


def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))
# -------------------------------------------------------------#


def smoothed_gaussian(x_vals, y_vals, sigma):
    smoothed_vals = np.zeros(y_vals.shape)
    for x_position in x_vals:
        kernel = np.exp(-(x_vals - x_position) ** 2 / (2 * sigma ** 2))
        kernel = kernel / sum(kernel)
        smoothed_vals[x_position] = sum(y_vals * kernel)

    return smoothed_vals
# -------------------------------------------------------------#


def smoothed_morlet(x, y, sigma):
    smoothed_vals = np.zeros(y.shape)
    for x_position in x:
        kernel = np.exp(-(x - x_position) ** 2 / 
        (2 * sigma ** 2)) * np.sin(x)
        kernel = kernel / sum(kernel)
        smoothed_vals[x_position] = sum(y * kernel)

    return smoothed_vals


FWHM = 4
n_points = 60
x_vals = np.arange(n_points)
y_vals = np.random.normal(size=n_points)
sigma = fwhm2sigma(FWHM)

smoothed_g = smoothed_gaussian(x_vals, y_vals, sigma)
smoothed_m = smoothed_morlet(x_vals, y_vals, sigma)

# plt.bar(x_vals, smoothed_vals, alpha=0.2)
plt.plot(x_vals, y_vals, lw=2, label='original')
plt.plot(x_vals, smoothed_g, lw=3, c='r', label='gaussian')
# plt.plot(x_vals, smoothed_m, lw=3, c='b', label='morlet')


pl.legend(frameon=False)
pl.savefig('m.png')
pl.show()
