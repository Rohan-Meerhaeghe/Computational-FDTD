import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spec


def source(t, A, omega_0, t_0, sigma):
    return A * np.sin(omega_0 * (t - t_0)) * np.exp(-((t - t_0) ** 2 / sigma**2))


def source_freq(omega, A, sigma, t_0, omega_0):
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
            * spec.jv(0, omega * r / c)
            * spec.hankel1(0, omega * r_0 / c)
        )

        for n in range(1, N):
            # part_sum = np.sum(np.abs(result))
            candidate = (
                2
                / (3j)
                * source_freq(omega=omega, A=1.0, sigma=sigma, t_0=t_0, omega_0=omega_0)
                * spec.jv(2 * n / 3, omega * r / c)
                * spec.hankel1(2 * n / 3, omega * r_0 / c)
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
            * spec.jv(0, omega * r / c)
            * spec.hankel1(0, omega * r_0 / c)
        )
        for n in range(1, N):
            # part_sum = np.sum(np.abs(result))
            candidate = (
                2
                / (3j)
                * source_freq(omega=omega, A=1.0, sigma=sigma, t_0=t_0, omega_0=omega_0)
                * spec.jv(2 * n / 3, omega * r_0 / c)
                * spec.hankel1(2 * n / 3, omega * r / c)
                * np.cos(2 * n / 3 * phi)
                * np.cos(2 * n / 3 * phi_0)
            )
            if np.isnan(candidate).any() == False:
                result += candidate
            # print(n, "\t", np.sum(np.abs(result)) / part_sum - 1)
        return result


# Example: assume E_t is your recorded field vs time
# and dt is your time step size (same as in your FDTD grid)
c = 343
d = 5
n_d = 50
dt = 1 / (c * np.sqrt((1 / (d / n_d) ** 2) + (1 / (d / n_d) ** 2)))  # time step
n_t = int(0.2 / dt)
t = np.linspace(0, 0.2, n_t)
source_t = source(
    t, 1, c / d * 2, 4.5 * 10 ** (-2), 1 / 50
)  # just a test signal at 300 THz

# Compute FFT
fft_source = np.fft.fft(source_t,n_t*10,norm="backward")*dt
freqs = np.fft.fftfreq(n_t*10, dt)  # frequencies in Hz

# Shift so 0 frequency is centered
fft_source = np.fft.fftshift(fft_source)
freqs = np.fft.fftshift(freqs)

relevant_omegas = np.linspace(0.1 * c / d, 4 * c / d, 50)
analytical = np.zeros_like(relevant_omegas).astype(complex)
for n in range(0, 20, 3):

    for i in range(len(analytical)):
        analytical[i] = analytical_sol(
            omega=relevant_omegas[i],
            r=np.sqrt(17) * d / 2,
            phi=np.arctan(1 / 4),
            c=c,
            r_0=np.sqrt(2) * d,
            phi_0=5 * np.pi / 4,
            sigma=1 / 50,
            t_0=4.5 * 10 ** (-2),
            omega_0=c / d * 2,
            N=n,
        )
    plt.plot(relevant_omegas, np.abs(analytical), label="n = " + str(n))
plt.close()
# Convert to angular frequency if you want
omega = 2 * np.pi * freqs

# Plot spectrum magnitude
plt.plot(omega, np.abs(fft_source), color="blue", label="FFT")  # frequency in THz
plt.plot(
    relevant_omegas,
    np.abs(
        source_freq(
            omega=relevant_omegas,
            A=1.0,
            sigma=1 / 50,
            t_0=4.5 * 10 ** (-2),
            omega_0=c / d * 2,
        )
    ),
    color="orange",
    label="analytical",
)
plt.legend()
plt.xlabel(r"$\omega$ [Hz]")
plt.ylabel(r"|P($\omega$)|")
plt.xlim(0.1 * c / d, 4 * c / d)
plt.savefig("analytical_r_3")
plt.show()
