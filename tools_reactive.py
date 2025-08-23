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
    recorder1,
    recorder2,
    recorder3,
    comparison,
    name,
):
    n_of_samples = timesteps.size * 10
    # generate frequency axes
    fftrecorder1 = fft(recorder1.flatten(), n_of_samples) * dt
    fftrecorder2 = fft(recorder2.flatten(), n_of_samples) * dt
    fftrecorder3 = fft(recorder3.flatten(), n_of_samples) * dt
    freqs = fftfreq(n_of_samples, dt)
    fftrecorder1 = fftshift(fftrecorder1)
    fftrecorder2 = fftshift(fftrecorder2)
    fftrecorder3 = fftshift(fftrecorder3)
    freqs = fftshift(freqs)
    omegas = 2 * np.pi * freqs

    # Define band of interest (e.g., 40–250 Hz)
    omega_min = 0.1 * c / d
    omega_max = 4 * c / d
    mask = (omegas >= omega_min) & (omegas <= omega_max)

    # Clip out useful part
    omegas_band = omegas[mask]
    fftrecorder1_band = fftrecorder1[mask]
    fftrecorder2_band = fftrecorder2[mask]
    fftrecorder3_band = fftrecorder3[mask]
    np.savez(
        "main_" + str(name) + ".npz",
        arr1=omegas_band,
        arr2=fftrecorder1_band,
        arr3=fftrecorder2_band,
        arr4=fftrecorder3_band,
        arr5=timesteps,
        arr6=recorder1,
        arr7=recorder2,
        arr8=recorder3,
    )

    data = np.load(comparison)
    omegas_orig = data["arr1"]
    fft_recorder1_orig = data["arr2"]
    fft_recorder2_orig = data["arr3"]
    fft_recorder3_orig = data["arr4"]
    timesteps_orig = data["arr5"]
    recorder1_orig = data["arr6"]
    recorder2_orig = data["arr7"]
    recorder3_orig = data["arr8"]

    plt.subplots(constrained_layout=True)
    plt.subplot(1, 3, 1)
    plt.plot(timesteps, recorder1, label="Recorder 1")
    plt.plot(timesteps, recorder2, label="Recorder 2")
    plt.plot(timesteps, recorder3, label="Recorder 3")
    plt.title("Ratio time recorder")
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(omegas_band, np.abs(fftrecorder1_band) / np.abs(fft_recorder1_orig))
    plt.plot(omegas_band, np.abs(fftrecorder2_band) / np.abs(fft_recorder2_orig))
    plt.plot(omegas_band, np.abs(fftrecorder3_band) / np.abs(fft_recorder3_orig))
    plt.xlim(omega_min, omega_max)
    plt.xlabel(r"$\omega$ [Hz]")
    plt.ylabel("Amplitude ratio")
    plt.title("Amplitude ratio")
    plt.subplot(1, 3, 3)
    plt.plot(
        omegas_band, np.unwrap(np.angle(fftrecorder1_band) - np.angle(fft_recorder1_orig))
    )
    plt.plot(
        omegas_band, np.unwrap(np.angle(fftrecorder2_band) - np.angle(fft_recorder2_orig))
    )
    plt.plot(
        omegas_band, np.unwrap(np.angle(fftrecorder3_band) - np.angle(fft_recorder3_orig))
    )
    plt.xlim(omega_min, omega_max)
    plt.xlabel(r"$\omega$ [Hz]")
    plt.ylabel("Phase [radians]")
    plt.title("Phase difference")
    plt.savefig("FFT_" + name + ".png")
    plt.show()
    plt.close()
