import numpy as np
from scipy.special import hankel1
from numpy.fft import fft
import matplotlib.pyplot as plt


def post_processing(dx, dy, c, dt, recorder, recorder_ref, n_of_samples, tijdreeks, fc):

    # generatie frequentie-as - generate frequency axes
    maxf = 1 / dt

    df = maxf / n_of_samples

    fas = np.zeros((n_of_samples))

    for loper in range(0, n_of_samples):
        fas[loper] = df * loper

    fas[0] = 0.00001  # avoiding phase problem at k=0 in analytical solution

    # amplitudeverhouding en faseverhouding FDTD - amplitude ratio and phase
    # difference from FDTD
    print(recorder.shape)
    fftrecorder = fft(recorder.flatten(), n_of_samples)
    print(fftrecorder.shape)
    fftrecorder_ref = fft(recorder_ref.flatten(), n_of_samples)

    Averhouding_FDTD = np.abs(fftrecorder_ref / fftrecorder)

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

    lambdaoverdx = (c / fas) / dx

    plt.show()
    plt.close()
    print("Averhouding_FDTD")
    print(Averhouding_FDTD.shape)
    print("lambdaoverdx")
    print(lambdaoverdx.shape)
