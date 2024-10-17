import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def design_equiripple_fir_filter():
    # Sampling Frequency
    Fs = 48000  # Hz
    
    # Filter Specifications (in Hz)
    Fpass = 9600  # Passband Frequency in Hz
    Fstop = 12000  # Stopband Frequency in Hz
    Dpass = 0.057501127785  # Passband Ripple (in linear scale, for 1dB)
    Dstop = 0.0001  # Stopband Attenuation (in linear scale, for 80dB)

    # Define the bands (in Hz, directly, not normalized)
    bands = [0, Fpass, Fstop, Fs / 2]  # Frequencies from 0 to Nyquist (Fs/2)

    # Define the desired magnitudes (passband = 1, stopband = 0)
    desired = [Dpass, Dpass, Dstop, Dstop]  # Magnitude in each band: passband (1), stopband (0)

    # Define the weights (passband weight, stopband weight)
    weights = [Dpass, Dpass, Dstop, Dstop]  # Weighting for each band (passband and stopband)

    # Estimate filter order (this can be adjusted for the desired performance)
    N = 50  # You may increase this for better performance

    # Design the filter using the Parks-McClellan (Remez) algorithm
    b = signal.remez(N, bands, desired, weight=weights, fs=Fs)

    # Plot Frequency Response
    w, h = signal.freqz(b, worN=2000, fs=Fs)  # Keep frequency in Hz
    plt.plot(w, 20 * np.log10(abs(h)))
    plt.title('Equiripple Lowpass Filter Frequency Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid()
    plt.show()

    # Plot Pole-Zero Plot
    z, p, k = signal.tf2zpk(b, [1])  # Since it's an FIR filter, a = [1]
    plt.scatter(np.real(z), np.imag(z), marker='o', color='blue', label='Zeros')
    plt.title('Pole-Zero Plot')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.grid()
    plt.legend()
    plt.show()

    return b, N

# Call the function to design the equiripple FIR filter
filter_coeffs, filter_order = design_equiripple_fir_filter()
print(f'Filter order: {filter_order}')
