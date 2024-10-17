import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

#danialG 10/17/2024

# Specifications
fs = 8000  # Sampling frequency (Hz)
f0 = 2000  # Frequency to reject (Hz)
Q = 30     # Quality factor

# Design a notch filter
b, a = signal.iirnotch(f0, Q, fs)

# 1. B and A coefficients
print("B coefficients:", b)
print("A coefficients:", a)

# 2. Pole/zero plot
z, p, k = signal.tf2zpk(b, a)

plt.figure(figsize=(6,6))
plt.scatter(np.real(z), np.imag(z), label='Zeros', s=100, edgecolor='b', facecolor='none')
plt.scatter(np.real(p), np.imag(p), label='Poles', s=100, edgecolor='r', facecolor='none')
plt.title('Pole/Zero Diagram')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.axhline(0, color='black',linewidth=1)
plt.axvline(0, color='black',linewidth=1)
plt.grid()
plt.legend()
plt.show()

# 3. Frequency response
w, h = signal.freqz(b, a, fs=fs)

# Magnitude plot
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(w, 20 * np.log10(abs(h)), 'b')
plt.title('Frequency Response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
plt.grid()

# Phase plot
plt.subplot(2, 1, 2)
plt.plot(w, np.angle(h), 'g')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Phase [radians]')
plt.grid()
plt.show()
