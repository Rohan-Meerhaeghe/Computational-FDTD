import numpy as np
from scipy.special import hankel1
from numpy.fft import fft
import matplotlib.pyplot as plt
from scipy.integrate import quad


def post_Afout_Pfout(
    dx,
    dy,
    c,
    dt,
    d,
    x_ref,
    x_recorder,
    y_ref,
    y_recorder,
    recorder,
    recorder_ref,
    n_of_samples,
    tijdreeks,
    fc,
):

    # generatie frequentie-as - generate frequency axes
    maxf = 1 / dt

    df = maxf / n_of_samples

    fas = np.zeros((n_of_samples))

    for loper in range(0, n_of_samples):
        fas[loper] = df * loper

    fas[0] = 0.00001  # avoiding phase problem at k=0 in analytical solution

    # amplitudeverhouding en faseverhouding analytisch - analytical amplitude
    # ratio and phase difference
    r_1 = np.sqrt(x_recorder**2 + y_recorder**2)
    phi_1 = np.arctan(y_recorder / x_recorder)
    if phi_1 < 0:
        phi_1 = phi_1 + 2 * np.pi

    r_2 = np.sqrt(x_ref**2 + y_ref**2)
    phi_2 = np.arctan(y_ref / x_ref)
    if phi_2 < 0:
        phi_2 = phi_2 + 2 * np.pi

    aantalcellengepropageerd = np.sqrt(
        (x_recorder - x_ref) ** 2 + (y_recorder - y_ref) ** 2
    )

    phi_0 = np.pi / 4
    alpha = 3 * np.pi / 4
    r_0 = np.sqrt(2) * d
    n = alpha / np.pi
    k = 2 * np.pi * fas / c

    def R(eta, rho):
        return np.sqrt(rho**2 + r_0**2 + 2 * rho * r_0 * np.cos(eta))

    def myquad(integrandum, a, b):
        return quad(integrandum, a, b)

    vector_quad = np.vectorize(myquad, excluded=["a", "b"])

    def V_d(beta, rho):
        integrandum = lambda t, beta, rho: (
            hankel1(0, k * R(1j * t, rho))
            * np.sin(beta / n)
            / (np.cosh(t / n) - np.cos(beta / n))
        )
        intfun = lambda beta, rho: quad(integrandum, 0, np.inf, args=(beta, rho))[0]
        vec_int = np.vectorize(intfun)
        return 1 / (2 * np.pi * n) * vec_int(beta, rho)

    def u_d(rho, phi):
        return (
            V_d(-np.pi - phi + phi_0, rho)
            - V_d(np.pi - phi + phi_0, rho)
            - V_d(-np.pi - phi - phi_0, rho)
            + V_d(np.pi - phi - phi_0, rho)
        )

    def distance_to_source(rho, phi):
        return np.sqrt(r_0**2 + rho**2 - 2 * r_0 * rho * np.cos(phi_0 - phi))

    def u_GO(rho, phi):
        return np.heaviside(np.pi + phi_0 - phi) * hankel1(
            k * distance_to_source(rho, phi)
        )

    def u_t(rho, phi):
        return u_d(rho, phi) + u_GO(rho, phi)

    Averhouding_theorie = np.abs(u_t(r_1, phi_1) / u_t(r_2, phi_2))

    Pverschil_theorie = np.unwrap(np.angle(1j * np.pi * u_t(r_1, phi_1))) - np.unwrap(
        np.angle(1j * np.pi * u_t(r_2, phi_2))
    )

    # amplitudeverhouding en faseverhouding FDTD - amplitude ratio and phase
    # difference from FDTD
    print(recorder.shape)
    fftrecorder = fft(recorder.flatten(), n_of_samples)
    print(fftrecorder.shape)
    fftrecorder_ref = fft(recorder_ref.flatten(), n_of_samples)

    Averhouding_FDTD = np.abs(fftrecorder_ref / fftrecorder)

    Pverschil_FDTD = np.unwrap(np.angle(fftrecorder_ref)) - np.unwrap(
        np.angle(fftrecorder)
    )

    plt.subplots()
    plt.subplot(2, 3, 1)
    plt.plot(tijdreeks, recorder)
    plt.title("t recorder")
    plt.subplot(2, 3, 2)
    plt.plot(fas, np.abs(fftrecorder))
    plt.title("fft recorder abs")
    plt.xlim([0.05 * fc, 2 * fc])

    plt.subplot(2, 3, 3)
    plt.plot(fas, np.unwrap(np.angle(fftrecorder)))
    plt.title("fft recorder phase")
    plt.xlim([0.05 * fc, 2 * fc])
    plt.ylim([-50, 0.0])

    plt.subplot(2, 3, 4)
    plt.plot(tijdreeks, recorder_ref)
    plt.title("t recorder ref")
    plt.subplot(2, 3, 5)
    plt.plot(fas, np.abs(fftrecorder_ref))
    plt.title("fft recorder ref abs")
    plt.xlim([0.05 * fc, 2 * fc])

    plt.subplot(2, 3, 6)
    plt.plot(fas, np.unwrap(np.angle(fftrecorder_ref)))
    plt.title("fft recorder phase")
    plt.xlim([0.05 * fc, 2 * fc])
    plt.ylim([-50, 0.0])

    # vergelijking analytisch-FDTD - comparison analytical versus FDTD
    lambdaoverdx = (c / fas) / dx

    Averhoudingrel = Averhouding_FDTD / Averhouding_theorie.flatten()

    Averhouding = 1 + ((Averhoudingrel - 1) / aantalcellengepropageerd)

    plt.show()
    plt.close()
    print("Averhouding_FDTD")
    print(Averhouding_FDTD.shape)
    print("Averhouding_theorie")
    print(Averhouding_theorie.shape)
    print("lambdaoverdx")
    print(lambdaoverdx.shape)
    print("Averhouding")
    print(Averhouding.shape)
    plt.subplots()
    plt.subplot(2, 1, 1)
    plt.plot(lambdaoverdx, Averhouding)

    plt.xlim([5, 20])
    plt.title("Amplitude ratio FDTD/analyt. per cel")
    plt.ylabel("ratio")
    plt.xlabel("number of cells per wavelength")
    plt.ylim([0.99, 1.01])

    # Pdifference = np.unwrap((Pverschil_FDTD+Pverschil_theorie)/aantalcellengepropageerd)
    Pdifference = (Pverschil_FDTD + Pverschil_theorie) / aantalcellengepropageerd

    plt.subplot(2, 1, 2)
    plt.plot(lambdaoverdx, Pdifference)
    plt.xlim([5, 20])
    plt.title("Phase difference FDTD - analyt. per cel")
    plt.ylabel("Difference")
    plt.xlabel("number of cells per wavelength")
    plt.ylim([-0.03, 0.03])
    plt.show()
