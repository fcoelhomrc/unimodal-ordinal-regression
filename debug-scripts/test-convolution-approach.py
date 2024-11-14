import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity


rng = np.random.default_rng(42)

size = 10000
N = 10
x1 = rng.binomial(N, p=0.3, size=size)
x2 = rng.binomial(N, p=0.35, size=size)
x3 = rng.binomial(N, p=0.4, size=size)
x4 = rng.binomial(N, p=0.42, size=size)
x5 = rng.binomial(N, p=0.32, size=size)

# Set up the plot
plt.figure(figsize=(10, 6))
x_values = np.arange(1, N+1).reshape(-1, 1)

cases = [x1, x2, x3, x4, x5]
densities = []
for data in cases:
    sample_data = data
    kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
    kde.fit(sample_data.reshape(-1, 1))
    log_density = kde.score_samples(x_values)
    density = np.exp(log_density)
    densities.append(density)
    plt.plot(x_values, density)

y = densities[0]
for density in densities[1:]:
    y = np.convolve(y, density, mode='full')
y /= y.sum()
plt.plot(np.arange(1, len(y)+1)/ len(cases), y , c="k", lw=2.)

plt.show()