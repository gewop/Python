import numpy as np
import matplotlib.pyplot as plt

# Frequency range for the plots (logarithmic scale)
frequencies = np.logspace(0, 5, 500)  # from 1 Hz to 100 kHz
omega = 2 * np.pi * frequencies

# Define cutoff frequency for low pass and high pass filters
cutoff_freq = 1000  # 1 kHz cutoff
omega_c = 2 * np.pi * cutoff_freq

# Frequency response calculations
# Ideal Differentiator: |H(jω)| = ω
ideal_diff_magnitude = omega

# Low Pass Filter: |H(jω)| = 1 / sqrt(1 + (ω/ω_c)^2)
low_pass_magnitude = 1 / np.sqrt(1 + (omega / omega_c)**2)

# Ideal Integrator: |H(jω)| = 1 / ω
ideal_integrator_magnitude = 1 / omega

# High Pass Filter: |H(jω)| = |ω / sqrt(ω_c^2 + ω^2)|
high_pass_magnitude = omega / np.sqrt(omega_c**2 + omega**2)

# Plot the results
plt.figure(figsize=(12, 8))

# Plot for Ideal Differentiator and Low Pass Filter
plt.subplot(2, 1, 1)
plt.loglog(frequencies, ideal_diff_magnitude, label="Ideal Differentiator")
plt.loglog(frequencies, low_pass_magnitude, label="Low Pass Filter")
plt.title("Frequency Response of Ideal Differentiator and Low Pass Filter")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(which="both", linestyle="--")
plt.legend()

# Plot for Ideal Integrator and High Pass Filter
plt.subplot(2, 1, 2)
plt.loglog(frequencies, ideal_integrator_magnitude, label="Ideal Integrator")
plt.loglog(frequencies, high_pass_magnitude, label="High Pass Filter")
plt.title("Frequency Response of Ideal Integrator and High Pass Filter")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(which="both", linestyle="--")
plt.legend()

plt.tight_layout()
plt.show()
