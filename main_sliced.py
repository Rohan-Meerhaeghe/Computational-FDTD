import numpy as np
import matplotlib.pyplot as plt
from tools import post_processing

from matplotlib.animation import ArtistAnimation


def source(t, A, fc, t0, sigma):
    return A * np.sin(2 * np.pi * fc * (t - t0)) * np.exp(-((t - t0) ** 2) / (sigma))


def kappa(d, d_PML, kappa_max, m=3.5):
    return kappa_max * (d / d_PML) ** m


def FDTD(
    c=343.0,
    d=5.0,
    CFL=1.0,
    T=0.15,
    n_d=200,
    n_PML=25,
    A=1.0,
    fc=100,
    t0=2.5e-2,
    sigma=5e-7,
    plot_kappa=False,
    show_plots=False,
    make_movie=False,
):
    # INITIALISATION 2D-GRID AND SIMULATION PARAMETERS-------------------------
    n_edge = n_PML + int(3 / 2 * n_d)
    dx = d / n_d
    dy = d / n_d
    dt = CFL / (c * np.sqrt((1 / dx**2) + (1 / dy**2)))  # time step
    n_t = int(T / dt)

    print("dx: ", dx)
    print("dt: ", dt)
    print("n_t: ", n_t)

    # location of source and receivers
    x_source = int(n_PML + n_d / 2)
    y_source = int(n_PML + n_d / 2)

    x_recorder_1 = int(n_PML + 3 / 2 * n_d)
    y_recorder_1 = int(n_PML + 2 * n_d)  # location receiver 1

    x_recorder_2 = x_recorder_1 + n_d
    y_recorder_2 = y_recorder_1  # location receiver 2

    x_recorder_3 = x_recorder_1 + 2 * n_d
    y_recorder_3 = y_recorder_1  # location receiver 3

    x_edge = np.zeros(2 * n_edge + n_d)
    y_edge = np.zeros_like(x_edge)  # coordinates of the edge, used for plotting
    for i in range(n_edge):
        x_edge[i] = n_edge
        y_edge[i] = i
    for i in range(n_edge, 2 * n_edge + n_d):
        x_edge[i] = i
        y_edge[i] = n_edge

    # initialisation of o and p fields
    length = 4 * n_d + 2 * n_PML
    height = n_edge + n_d + n_PML
    p = np.zeros((height, length))
    px = np.zeros_like(p)
    py = np.zeros_like(p)
    ox = np.zeros((height, length + 1))
    oy = np.zeros((height + 1, length))

    # initialization of the PML damping term
    kappa_left = np.transpose(np.zeros((height, n_PML)))
    kappa_right = np.transpose(np.zeros((n_PML + n_d, n_PML)))
    kappa_down = np.zeros((n_PML, n_edge))
    kappa_up = np.zeros((n_PML, length))

    kappa_max = 0.5 / dt  # choose kappa_max = 2/dt or 6/dt

    for i in range(n_PML):
        kappa_left[i] = np.full(
            height, kappa(d=(n_PML - i) * dx, kappa_max=kappa_max, d_PML=dx * n_PML)
        )
        kappa_right[i] = np.full(
            n_PML + n_d, kappa(d=(i + 1) * dx, kappa_max=kappa_max, d_PML=dx * n_PML)
        )
        kappa_down[i] = np.full(
            n_edge, kappa(d=(n_PML - i) * dy, kappa_max=kappa_max, d_PML=dy * n_PML)
        )
        kappa_up[i] = np.full(
            length, kappa(d=(i + 1) * dy, kappa_max=kappa_max, d_PML=dy * n_PML)
        )
    kappa_right = np.transpose(kappa_right)
    kappa_left = np.transpose(kappa_left)
    print("max same damping:", (1 - kappa_max * dt / 2) / (1 + kappa_max * dt / 2))
    print("max diff damping: ", 1 / (1 + kappa_max * dt / 2))

    x_PML = np.zeros(height - n_PML + length - n_PML)
    y_PML = np.zeros_like(x_PML)
    for i in range(height - n_PML):
        x_PML[i] = n_PML
        y_PML[i] = i
    for i in range(length - n_PML):
        x_PML[height - n_PML + i] = n_PML + i
        y_PML[height - n_PML + i] = height - n_PML

    if plot_kappa == True:
        kappa_vals = np.fromiter(
            (
                kappa(d=i * dx, d_PML=dx * n_PML, kappa_max=kappa_max)
                for i in range(n_PML)
            ),
            dtype=float,
        )
        kappa_same_vals = np.fromiter(
            (
                (1 - kappa(d=i * dx, d_PML=dx * n_PML, kappa_max=kappa_max) * dt / 2)
                / (1 + kappa(d=i * dx, d_PML=dx * n_PML, kappa_max=kappa_max) * dt / 2)
                for i in range(n_PML)
            ),
            dtype=float,
        )
        kappa_diff_vals = np.fromiter(
            (
                1
                / (1 + kappa(d=i * dx, d_PML=dx * n_PML, kappa_max=kappa_max) * dt / 2)
                for i in range(n_PML)
            ),
            dtype=float,
        )

        fig, ax1 = plt.subplots()

        ax1.set_xlabel("cell index")
        ax1.set_ylabel(r"$\kappa$", color="red")
        ax1.plot(kappa_vals, color="red")
        ax1.tick_params(axis="y", labelcolor="red")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Damping factor", color="blue")
        ax2.plot(
            kappa_same_vals,
            color="blue",
            label=r"$\frac{1-\kappa dt/2}{1+\kappa dt/2}$",
        )
        ax2.plot(kappa_diff_vals, color="cyan", label=r"$\frac{1}{1+\kappa dt/2}$")
        ax2.tick_params(axis="y", labelcolor="blue")
        plt.legend()
        plt.show()
        plt.close()

    # initialisation time series receivers
    recorder_1 = np.zeros((n_t, 1))
    recorder_2 = np.zeros_like(recorder_1)
    recorder_3 = np.zeros_like(recorder_1)

    timesteps = np.linspace(0, n_t * dt, n_t)
    source_vals = source(timesteps, A=A, fc=fc, t0=t0, sigma=sigma)

    # TIME ITTERATION----------------------------------------------------
    fig, ax = plt.subplots()
    plt.axis("equal")
    movie = []
    for i in range(0, n_t):
        t = (i - 1) * dt
        timesteps[i] = t
        print("%d/%d" % (i + 1, n_t), end="\r")

        # adding source term to propagation
        px[y_source, x_source] += source_vals[i]

        # propagate over one time step

        # lower side
        px[n_PML:n_edge, n_PML:n_edge] += (
            dt
            / dx
            * c**2
            * (
                ox[n_PML:n_edge, n_PML:n_edge]
                - ox[n_PML:n_edge, n_PML + 1 : n_edge + 1]
            )
        )
        py[n_PML:n_edge, n_PML:n_edge] += (
            dt
            / dy
            * c**2
            * (
                oy[n_PML:n_edge, n_PML:n_edge]
                - oy[n_PML + 1 : n_edge + 1, n_PML:n_edge]
            )
        )
        # upper side
        px[n_edge:-n_PML, n_PML:-n_PML] += (
            dt
            / dx
            * c**2
            * (
                ox[n_edge:-n_PML, n_PML : -n_PML - 1]
                - ox[n_edge:-n_PML, n_PML + 1 : -n_PML]
            )
        )
        py[n_edge:-n_PML, n_PML:-n_PML] += (
            dt
            / dy
            * c**2
            * (
                oy[n_edge : -n_PML - 1, n_PML:-n_PML]
                - oy[n_edge + 1 : -n_PML, n_PML:-n_PML]
            )
        )
        # inside PML at the left 'wall'
        px[n_PML:-n_PML, :n_PML] = (
            dt
            / (1 + kappa_left[n_PML:-n_PML, :] * dt / 2)
            / dx
            * c**2
            * (ox[n_PML:-n_PML, :n_PML] - ox[n_PML:-n_PML, 1 : n_PML + 1])
        ) + (1 - kappa_left[n_PML:-n_PML, :] * dt / 2) / (
            1 + kappa_left[n_PML:-n_PML, :] * dt / 2
        ) * px[
            n_PML:-n_PML, :n_PML
        ]
        py[n_PML:-n_PML, :n_PML] += (
            dt
            / dy
            * c**2
            * (oy[n_PML : -n_PML - 1, :n_PML] - oy[n_PML + 1 : -n_PML, :n_PML])
        )

        # inside PML at the right 'wall'
        px[n_edge:-n_PML, -n_PML:] = (
            dt
            / (1 + kappa_right[:-n_PML, :] * dt / 2)
            / dx
            * c**2
            * (ox[n_edge:-n_PML, -n_PML - 1 : -1] - ox[n_edge:-n_PML, -n_PML:])
        ) + (1 - kappa_right[:-n_PML, :] * dt / 2) / (
            1 + kappa_right[:-n_PML, :] * dt / 2
        ) * px[
            n_edge:-n_PML, -n_PML:
        ]

        py[n_edge:-n_PML, -n_PML:] += (
            dt
            / dy
            * c**2
            * (oy[n_edge : -n_PML - 1, -n_PML:] - oy[n_edge + 1 : -n_PML, -n_PML:])
        )

        # inside PML at the lower 'wall'
        px[:n_PML, n_PML:n_edge] = (
            dt
            / dx
            * c**2
            * (ox[:n_PML, n_PML:n_edge] - ox[:n_PML, n_PML + 1 : n_edge + 1])
        ) + px[:n_PML, n_PML:n_edge]
        py[:n_PML, n_PML:n_edge] = (
            dt
            / dy
            / (1 + kappa_down[:, n_PML:] * dt / 2)
            * c**2
            * (oy[:n_PML, n_PML:n_edge] - oy[1 : n_PML + 1, n_PML:n_edge])
        ) + py[:n_PML, n_PML:n_edge] * (1 - kappa_down[:, n_PML:] * dt / 2) / (
            1 + kappa_down[:, n_PML:] * dt / 2
        )
        # inside PML at the upper 'wall'
        px[-n_PML:, n_PML:-n_PML] = (
            dt
            / dx
            * c**2
            / (1 + kappa_up[:, n_PML:-n_PML] * dt / 2)
            * (ox[-n_PML:, n_PML : -n_PML - 1] - ox[-n_PML:, n_PML + 1 : -n_PML])
        ) + px[-n_PML:, n_PML:-n_PML] * (1 - kappa_up[:, n_PML:-n_PML] * dt / 2) / (
            1 + kappa_up[:, n_PML:-n_PML] * dt / 2
        )
        py[-n_PML:, n_PML:-n_PML] += (
            dt
            / dy
            * c**2
            * (oy[-n_PML - 1 : -1, n_PML:-n_PML] - oy[-n_PML:, n_PML:-n_PML])
        )

        # inside PML in the lower left corner
        px[:n_PML, :n_PML] = (
            dt
            / dx
            * c**2
            / (1 + kappa_left[:n_PML, :] * dt / 2)
            * (ox[:n_PML, :n_PML] - ox[:n_PML, 1 : n_PML + 1])
        ) + px[:n_PML, :n_PML] * (1 - kappa_left[:n_PML, :] * dt / 2) / (
            1 + kappa_left[:n_PML, :] * dt / 2
        )
        py[:n_PML, :n_PML] = (
            dt
            / dy
            * c**2
            / (1 + kappa_down[:, :n_PML] * dt / 2)
            * (oy[:n_PML, :n_PML] - oy[1 : n_PML + 1, :n_PML])
        ) + py[:n_PML, :n_PML] * (1 - kappa_down[:, :n_PML] * dt / 2) / (
            1 + kappa_down[:, :n_PML] * dt / 2
        )
        # inside PML in the upper left corner
        px[-n_PML:, :n_PML] = (
            dt
            / dx
            / (1 + kappa_left[-n_PML:, :] * dt / 2)
            * c**2
            * (ox[-n_PML:, :n_PML] - ox[-n_PML:, 1 : n_PML + 1])
        ) + px[-n_PML:, :n_PML] * (1 - kappa_left[-n_PML:, :] * dt / 2) / (
            1 + kappa_left[-n_PML:, :] * dt / 2
        )
        py[-n_PML:, :n_PML] = (
            dt
            / dy
            / (1 + kappa_up[:, -n_PML:] * dt / 2)
            * c**2
            * (oy[-n_PML - 1 : -1, :n_PML] - oy[-n_PML:, :n_PML])
        ) + py[-n_PML:, :n_PML] * (1 - kappa_up[:, -n_PML:] * dt / 2) / (
            1 + kappa_up[:, -n_PML:] * dt / 2
        )
        # inside PML in the upper right corner
        px[-n_PML:, -n_PML:] = (
            dt
            / dx
            / (1 + kappa_right[-n_PML:, :] * dt / 2)
            * c**2
            * (ox[-n_PML:, -1 - n_PML : -1] - ox[-n_PML:, -n_PML:])
        ) + px[-n_PML:, -n_PML:] * (1 - kappa_right[-n_PML:, :] * dt / 2) / (
            1 + kappa_right[-n_PML:, :] * dt / 2
        )
        py[-n_PML:, -n_PML:] = (
            dt
            / dy
            / (1 + kappa_up[:, -n_PML:] * dt / 2)
            * c**2
            * (oy[-n_PML - 1 : -1, -n_PML:] - oy[-n_PML:, -n_PML:])
        ) + py[-n_PML:, -n_PML:] * (1 - kappa_up[:, -n_PML:] * dt / 2) / (
            1 + kappa_up[:, -n_PML:] * dt / 2
        )

        # combine px and py for easy acces to update o fields
        p = px + py

        # lower side
        ox[n_PML:n_edge, n_PML + 1 : n_edge] += (
            dt
            / dx
            * (
                p[n_PML:n_edge, n_PML : n_edge - 1]
                - p[n_PML:n_edge, n_PML + 1 : n_edge]
            )
        )
        oy[n_PML + 1 : n_edge + 1, n_PML:n_edge] += (
            dt
            / dy
            * (p[n_PML:n_edge, n_PML:n_edge] - p[n_PML + 1 : n_edge + 1, n_PML:n_edge])
        )

        # upper side
        ox[n_edge:-n_PML, n_PML + 1 : -1 - n_PML] += (
            dt
            / dx
            * (
                p[n_edge:-n_PML, n_PML : -1 - n_PML]
                - p[n_edge:-n_PML, n_PML + 1 : -n_PML]
            )
        )
        oy[n_edge + 1 : -1 - n_PML, n_PML:-n_PML] += (
            dt
            / dy
            * (
                p[n_edge : -1 - n_PML, n_PML:-n_PML]
                - p[n_edge + 1 : -n_PML, n_PML:-n_PML]
            )
        )

        # inside PML at the left 'wall'
        ox[n_PML:-n_PML, 1 : n_PML + 1] = (
            dt
            / (1 + kappa_left[n_PML:-n_PML, :] * dt / 2)
            / dx
            * (p[n_PML:-n_PML, :n_PML] - p[n_PML:-n_PML, 1 : n_PML + 1])
        ) + (1 - kappa_left[n_PML:-n_PML, :] * dt / 2) / (
            1 + kappa_left[n_PML:-n_PML, :] * dt / 2
        ) * ox[
            n_PML:-n_PML, 1 : n_PML + 1
        ]
        oy[n_PML + 1 : -n_PML - 1, :n_PML] += (
            dt / dy * (p[n_PML : -n_PML - 1, :n_PML] - p[n_PML + 1 : -n_PML, :n_PML])
        )
        # inside PML at the right 'wall'
        ox[n_edge:-n_PML, -n_PML - 1 : -1] = (
            dt
            / (1 + kappa_right[:-n_PML, :] * dt / 2)
            / dx
            * (p[n_edge:-n_PML, -n_PML - 1 : -1] - p[n_edge:-n_PML, -n_PML:])
        ) + (1 - kappa_right[:-n_PML, :] * dt / 2) / (
            1 + kappa_right[:-n_PML, :] * dt / 2
        ) * ox[
            n_edge:-n_PML, -n_PML - 1 : -1
        ]
        oy[n_edge + 1 : -n_PML - 1, -n_PML:] += (
            dt
            / dy
            * (p[n_edge : -n_PML - 1, -n_PML:] - p[n_edge + 1 : -n_PML, -n_PML:])
        )

        # inside PML at the lower 'wall'
        ox[:n_PML, n_PML + 1 : n_edge] = (
            dt / dx * (p[:n_PML, n_PML : n_edge - 1] - p[:n_PML, n_PML + 1 : n_edge])
        ) + ox[:n_PML, n_PML + 1 : n_edge]

        oy[1 : n_PML + 1, n_PML:n_edge] = (
            dt
            / dy
            / (1 + kappa_down[:, n_PML:] * dt / 2)
            * (p[:n_PML, n_PML:n_edge] - p[1 : n_PML + 1, n_PML:n_edge])
        ) + oy[1 : n_PML + 1, n_PML:n_edge] * (1 - kappa_down[:, n_PML:] * dt / 2) / (
            1 + kappa_down[:, n_PML:] * dt / 2
        )

        # inside PML at the upper 'wall'
        ox[-n_PML:, n_PML + 1 : -1 - n_PML] += (
            dt / dx * (p[-n_PML:, n_PML : -1 - n_PML] - p[-n_PML:, n_PML + 1 : -n_PML])
        )
        oy[-1 - n_PML : -1, n_PML:-n_PML] = (
            dt
            / dy
            / (1 + kappa_up[:, n_PML:-n_PML] * dt / 2)
            * (p[-1 - n_PML : -1, n_PML:-n_PML] - p[-n_PML:, n_PML:-n_PML])
        ) + oy[-1 - n_PML : -1, n_PML:-n_PML] * (
            1 - kappa_up[:, n_PML:-n_PML] * dt / 2
        ) / (
            1 + kappa_up[:, n_PML:-n_PML] * dt / 2
        )

        # inside PML in the lower left corner
        ox[:n_PML, 1 : n_PML + 1] = (
            dt
            / dx
            / (1 + kappa_left[:n_PML, :] * dt / 2)
            * (p[:n_PML, :n_PML] - p[:n_PML, 1 : n_PML + 1])
        ) + ox[:n_PML, 1 : n_PML + 1] * (1 - kappa_left[:n_PML, :] * dt / 2) / (
            1 + kappa_left[:n_PML, :] * dt / 2
        )
        oy[1 : n_PML + 1, :n_PML] = (
            dt
            / dy
            / (1 + kappa_down[:, :n_PML] * dt / 2)
            * (p[:n_PML, :n_PML] - p[1 : n_PML + 1, :n_PML])
        ) + oy[1 : n_PML + 1, :n_PML] * (1 - kappa_down[:, :n_PML] * dt / 2) / (
            1 + kappa_down[:, :n_PML] * dt / 2
        )

        # inside PML in the upper left corner
        ox[-n_PML:, 1 : n_PML + 1] = (
            dt
            / dx
            / (1 + kappa_left[-n_PML:] * dt / 2)
            * (p[-n_PML:, :n_PML] - p[-n_PML:, 1 : n_PML + 1])
        ) + ox[-n_PML:, 1 : n_PML + 1] * (1 - kappa_left[-n_PML:] * dt / 2) / (
            1 + kappa_left[-n_PML:] * dt / 2
        )
        oy[-1 - n_PML : -1, :n_PML] = (
            dt
            / dy
            / (1 + kappa_up[:, :n_PML] * dt / 2)
            * (p[-1 - n_PML : -1, :n_PML] - p[-n_PML:, :n_PML])
        ) + oy[-1 - n_PML : -1, :n_PML] * (1 - kappa_up[:, :n_PML] * dt / 2) / (
            1 + kappa_up[:, :n_PML] * dt / 2
        )

        # inside PML in the upper right corner
        ox[-n_PML:, -n_PML - 1 : -1] += (
            dt / dx * (p[-n_PML:, -1 - n_PML : -1] - p[-n_PML:, -n_PML:])
        )
        oy[-1 - n_PML : -1, -n_PML:] += (
            dt / dy * (p[-1 - n_PML : -1, -n_PML:] - p[-n_PML:, -n_PML:])
        )

        recorder_1[i] = p[y_recorder_1, x_recorder_1]
        recorder_2[i] = p[y_recorder_2, x_recorder_2]
        recorder_3[i] = p[y_recorder_3, x_recorder_3]
        # store p field at receiver locations
        # presenting the p field
        plot_factor_A = 0.0001 * A
        if make_movie == True:
            artists = [
                ax.text(
                    length / 2,
                    height,
                    "Time: %f s /%f s" % (np.round(i * dt, 2), np.round(n_t * dt, 2)),
                    size=plt.rcParams["axes.titlesize"],
                    ha="center",
                ),
                ax.imshow(
                    np.clip(p, -plot_factor_A, plot_factor_A),
                    vmin=-plot_factor_A,
                    vmax=plot_factor_A,
                    origin="lower",
                ),
                ax.plot(x_source, y_source, "ks", fillstyle="none")[0],
                ax.plot(x_recorder_1, y_recorder_1, "ro", fillstyle="none")[0],
                ax.plot(x_recorder_2, y_recorder_2, "ro", fillstyle="none")[0],
                ax.plot(x_recorder_3, y_recorder_3, "ro", fillstyle="none")[0],
                ax.plot(x_edge, y_edge, "ks", fillstyle="full")[0],
                ax.plot(x_PML, y_PML, "w:")[0],
            ]
            movie.append(artists)
    if make_movie == True:
        my_anim = ArtistAnimation(
            fig, movie, interval=5 * 100 / n_t, repeat_delay=1000, blit=True
        )
        # my_anim.save("FDTD_cartesian.gif",fps=60)
        plt.show()
        # play back the stored movie
    plt.close()
    plt.plot(timesteps, recorder_1, "r-", label="Recorder 1")
    plt.plot(timesteps, recorder_2, "g-", label="Recorder 2")
    plt.plot(timesteps, recorder_3, "m-", label="Recorder 3")
    plt.legend()
    plt.savefig("recordings.png")
    if show_plots == True:
        plt.show()
    plt.close()
    plt.plot(timesteps, source(t=timesteps, A=A, fc=fc, t0=t0, sigma=sigma), "b-")
    plt.ylabel("p field source")
    plt.xlabel("Time [s]")
    plt.savefig("source.png")
    if show_plots == True:
        plt.show()
    plt.close()


FDTD(n_d=75, make_movie=True, plot_kappa=True)

"""
    # NAVERWERKING : BEREKENING FASEFOUT en AMPLITUDEFOUT---------------------------------
    # POST PROCESSING : CALCULATE PHASE and AMPLITUDE ERROR-------------------------------
    n_of_samples = 8192

    post_processing(
        dx,
        dy,
        c,
        dt,
        x_ref,
        x_recorder,
        y_ref,
        y_recorder,
        x_source,
        y_source,
        recorder,
        recorder_ref,
        n_of_samples,
        tijdreeks,
        fc,
    )
    post_processing(
        dx,
        dy,
        c,
        dt,
        x_ref2,
        x_recorder2,
        y_ref2,
        y_recorder2,
        x_source,
        y_source,
        recorder2,
        recorder2_ref,
        n_of_samples,
        tijdreeks,
        fc,
    )
"""
