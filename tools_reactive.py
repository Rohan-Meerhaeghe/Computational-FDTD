import numpy as np
from scipy.special import hankel1, jv
from numpy.fft import fft, ifft, fftfreq, fftshift
import matplotlib.pyplot as plt


def source_freq(omega, A, sigma, t_0, omega_0):
    result = np.zeros_like(omega).astype(complex)
    result = A * np.abs(
        np.sqrt(np.pi)
        * sigma
        / (2j)
        * np.exp(-1j * t_0 * omega)
        * (
            np.exp(-((omega - omega_0) ** 2) * sigma**2 / 4)
            - np.exp(-((omega + omega_0) ** 2) * sigma**2 / 4)
        )
    )
    return result


def post_processing(
    dt,
    timesteps,
    c,
    d,
    r,
    phi,
    r_0,
    phi_0,
    recorder,
    sigma,
    t_0,
    omega_0,
    string,
):
    n_of_samples = timesteps.size * 10
    # generate frequency axes
    fftrecorder = fft(recorder.flatten(), n_of_samples) * dt
    freqs = fftfreq(n_of_samples, dt)
    fftrecorder = fftshift(fftrecorder)
    freqs = fftshift(freqs)
    omegas = 2 * np.pi * freqs

    # Define band of interest (e.g., 40–250 Hz)
    omega_min = 0.1 * c / d
    omega_max = 4 * c / d
    mask = (omegas >= omega_min) & (omegas <= omega_max)

    # Clip out useful part
    omegas_band = omegas[mask]
    fftrecorder_band = fftrecorder[mask]
    np.savez("main_reactive_" + str(string) + ".npz", arr1=omegas_band, arr2=fftrecorder_band,arr3=timesteps,arr4=recorder)

    data = np.load("main_" + string + ".npz")
    omegas_orig = data["arr1"]
    fft_recorder_orig = data["arr2"]
    timesteps_orig = data["arr3"]
    recorder_orig = data["arr4"]

    plt.subplots(constrained_layout=True)
    plt.subplot(1, 3, 1)
    plt.plot(timesteps, recorder, label="reactive")
    plt.plot(timesteps_orig, recorder_orig, label="perfectly reflecting")
    plt.title("Time recorder")
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(omegas_band, np.abs(fftrecorder_band)/np.abs(fft_recorder_orig))
    plt.xlim(omega_min, omega_max)
    plt.xlabel(r"$\omega$ [Hz]")
    plt.ylabel("Amplitude ratio")
    plt.title("Amplitude ratio")
    plt.subplot(1, 3, 3)
    plt.plot(omegas_band, np.unwrap(np.angle(fftrecorder_band)-np.angle(fft_recorder_orig)))
    plt.xlim(omega_min, omega_max)
    plt.xlabel(r"$\omega$ [Hz]")
    plt.ylabel("Phase [radians]")
    plt.title("Phase shift")
    plt.savefig("FFT_reactive_" + string + ".png")
    # plt.show()
    plt.close()
