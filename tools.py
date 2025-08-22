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


def analytical_sol(omega, r, phi, c, r_0, phi_0, sigma, t_0, omega_0, N=50):

    if r <= r_0:
        result = (
            1
            / (3j)
            * source_freq(omega=omega, A=1.0, sigma=sigma, t_0=t_0, omega_0=omega_0)
            * jv(0, omega * r / c)
            * hankel1(0, omega * r_0 / c)
        )

        for n in range(1, N):
            # part_sum = np.sum(np.abs(result))
            candidate = (
                2
                / (3j)
                * source_freq(omega=omega, A=1.0, sigma=sigma, t_0=t_0, omega_0=omega_0)
                * jv(2 * n / 3, omega * r / c)
                * hankel1(2 * n / 3, omega * r_0 / c)
                * np.cos(2 * n / 3 * phi_0)
                * np.cos(2 * n / 3 * phi)
            )
            if np.isnan(candidate).any() == False:
                result += candidate
            # print(n, "\t", np.sum(np.abs(result)) / part_sum - 1)
        return result
    else:
        result = (
            1
            / (3j)
            * source_freq(omega=omega, A=1.0, sigma=sigma, t_0=t_0, omega_0=omega_0)
            * jv(0, omega * r / c)
            * hankel1(0, omega * r_0 / c)
        )
        for n in range(1, N):
            # part_sum = np.sum(np.abs(result))
            candidate = (
                2
                / (3j)
                * source_freq(omega=omega, A=1.0, sigma=sigma, t_0=t_0, omega_0=omega_0)
                * jv(2 * n / 3, omega * r_0 / c)
                * hankel1(2 * n / 3, omega * r / c)
                * np.cos(2 * n / 3 * phi)
                * np.cos(2 * n / 3 * phi_0)
            )
            if np.isnan(candidate).any() == False:
                result += candidate
            # print(n, "\t", np.sum(np.abs(result)) / part_sum - 1)
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
    source_recorder,
):
    n_of_samples = timesteps.size
    # generate frequency axes
    fftrecorder = fft(recorder.flatten(), n_of_samples)*dt
    fftsource = fft(source_recorder.flatten(), n_of_samples)*dt
    freqs = fftfreq(n_of_samples, dt)
    fftrecorder = fftshift(fftrecorder)
    freqs = fftshift(freqs)
    fftsource = fftshift(fftsource)
    omegas = 2 * np.pi * freqs

    relevant_omegas = np.linspace(0.1 * c / d, 4 * c / d, 50)
    analytical = np.zeros_like(relevant_omegas).astype(complex)
    
    for i in range(len(analytical)):
        analytical[i] = analytical_sol(
            omega=relevant_omegas[i],
            r=r,
            phi=phi,
            c=c,
            r_0=r_0,
            phi_0=phi_0,
            sigma=sigma,
            t_0=t_0,
            omega_0=omega_0,
        )

    open_space_rec = 1j * np.pi * hankel1(0, r * omegas / c)
    open_space_theo = 1j * np.pi * hankel1(0, r * relevant_omegas / c)

    plt.plot(omegas, np.abs(fftsource), color="magenta", label="Source")
    plt.plot(relevant_omegas,source_freq(relevant_omegas,1,sigma,t_0,omega_0), color='cyan',label='Analytical')
    plt.legend()
    plt.xlabel(r"$\omega$ [Hz]")
    plt.ylabel(r"$|S(\omega)|$")
    plt.xlim(0.1 * c / d, 4 * c / d)
    plt.show()

    # analytical amplitude ratio and phase difference

    Averhouding_theorie = np.abs(analytical) / np.abs(open_space_theo)

    Pverschil_theorie = np.unwrap(np.angle(analytical)) - np.unwrap(
        np.angle(open_space_theo)
    )

    # amplitude ratio and phase difference from FDTD
    Averhouding_FDTD = np.abs(fftrecorder / open_space_rec)

    Pverschil_FDTD = np.unwrap(np.angle(fftrecorder)) - np.unwrap(
        np.angle(open_space_rec)
    )
    omega_min = 0.1 * c / d
    omega_max = 4 * c / d

    plt.subplots()
    plt.subplot(1, 3, 1)
    plt.plot(timesteps, recorder, color="blue")
    plt.title("Time recorder")
    plt.subplot(1, 3, 2)
    plt.plot(omegas, np.abs(fftrecorder), color="blue", label="FDTD")
    plt.plot(relevant_omegas, np.abs(analytical), color="orange", label="Analytical")
    plt.xlim(omega_min, omega_max)
    plt.xlabel(r"$\omega$ [Hz]")
    plt.ylabel("Amplitude FFT")
    plt.legend()
    plt.title("Amplitude")
    plt.subplot(1, 3, 3)
    plt.plot(omegas, np.unwrap(np.angle(fftrecorder)), color="blue", label="FDTD")
    plt.plot(
        relevant_omegas,
        np.unwrap(np.angle(analytical)),
        color="orange",
        label="Analytical",
    )
    plt.xlim(omega_min, omega_max)
    plt.xlabel(r"$\omega$ [Hz]")
    plt.ylabel("Phase [radians]")
    plt.legend()
    plt.title("Phase FFT")
    plt.savefig("p_field.png")
    plt.show()
    plt.close()

    # vergelijking analytisch-FDTD - comparison analytical versus FDTD

    Averhouding = Averhouding_FDTD / Averhouding_theorie

    print("Averhouding_FDTD")
    print(Averhouding_FDTD.shape)
    print("Averhouding_theorie")
    print(Averhouding_theorie.shape)
    print("omegas")
    print(omegas.shape)
    print("Averhouding")
    print(Averhouding.shape)
    plt.subplots()
    plt.subplot(2, 1, 1)
    plt.plot(omegas, Averhouding)
    plt.title("Amplitude ratio FDTD/analyt")
    plt.ylabel("ratio")
    plt.xlabel(r"$\omega$")

    # Pdifference = np.unwrap((Pverschil_FDTD+Pverschil_theorie)/aantalcellengepropageerd)
    Pdifference = Pverschil_FDTD - Pverschil_theorie

    plt.subplot(2, 1, 2)
    plt.plot(omegas, Pdifference)
    plt.title("Phase difference FDTD")
    plt.ylabel("Phase difference")
    plt.xlabel(r"$\omega$")
    # plt.show()
    plt.close()
