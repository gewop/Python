import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sounddevice as sd

# Load the .mat file
data = scipy.io.loadmat('voice with whistle.mat',)
# Extract the signal and sampling frequency (adjust the variable names accordingly)
signal_data = data['data'].flatten()

Fs = 8000  # Sampling frequency (Hz)
F0 = 1540 # Frequency to reject (Hz)
Q = 2 # Quality factor

# Design a notch filter
b, a = signal.iirnotch(F0, Q, Fs)

# Filter the signal
filtered_signal = signal.filtfilt(b, a, signal_data)

# Plot the power spectral density (PSD) using pwelch
plt.figure()
plt.subplot(2, 1, 1)
f, Pxx = signal.welch(signal_data, Fs)
plt.semilogy(f, Pxx)
plt.title('Original Signal Spectrum')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power/Frequency [dB/Hz]')

plt.subplot(2, 1, 2)
f, Pxx = signal.welch(filtered_signal, Fs)
plt.semilogy(f, Pxx)
plt.title('Filtered Signal Spectrum')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power/Frequency [dB/Hz]')
plt.tight_layout()
plt.show()

print("Playing the original signal...")
sd.play(filtered_signal, Fs)
sd.wait()