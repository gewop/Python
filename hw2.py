import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

# Number of points
N = 8

# Create an NxN matrix of twiddle factors
twiddle_matrix = np.zeros((N, N), dtype=complex)

# Fill the matrix with twiddle factors W_N^(k*n)
for k in range(N):
    for n in range(N):
        twiddle_matrix[k, n] = np.exp(-2j * np.pi * k * n / N)

# Extract the second row (index 1) of the twiddle matrix
second_row = twiddle_matrix[1, :]

# Use freqz to compute the frequency response
w, h = freqz(second_row)

# Plot the magnitude response
plt.figure(figsize=(8, 5))
plt.plot(w / np.pi, np.abs(h))
plt.title('Magnitude Response of the Second Row of Twiddle Factor Matrix (N=8)')
plt.xlabel('Normalized Frequency (×π rad/sample)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()
