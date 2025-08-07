import numpy as np
import matplotlib.pyplot as plt
from tools_old import post_processing
from tools import post_Afout_Pfout

from matplotlib.animation import ArtistAnimation


def source(t, A, fc, t0, sigma):
    return A * np.sin(2 * np.pi * fc * (t - t0)) * np.exp(-((t - t0) ** 2 / sigma))


def kappa(d, d_PML, kappa_max, m=4):
    return kappa_max * (d / d_PML) ** m


def FDTD(
    c=343.0,
    d=5.0,
    CFL=1.0,
    T=0.15,
    n_d=200,
    n_PML=30,
    A=1.0,
    t0=4.5e-2,
    sigma=1/50,
    kappa_max_factor=0.254,
    plot_kappa=False,
    show_plots=False,
    make_movie=False,
    reactive_wedge=False,
    execute_post_processing=False,
):
    # INITIALISATION 2D-GRID AND SIMULATION PARAMETERS-------------------------
    n_edge = n_PML + int(3 / 2 * n_d)
    dx = d / n_d
    dy = d / n_d
    dt = CFL / (c * np.sqrt((1 / dx**2) + (1 / dy**2)))  # time step
    n_t = int(T / dt)
    Z_r = c
    Z_0 = 0
    Z_1 = 0
    if reactive_wedge == True:
        Z_0 = 2 * c
        Z_1 = 1
    fc = c / (np.pi * d)

    # location of source and receivers
    x_source = int(n_PML + n_d / 2)
    y_source = int(n_PML + n_d / 2)

    x_recorder_1 = int(n_PML + 3 / 2 * n_d)
    y_recorder_1 = int(n_PML + 2 * n_d)  # location receiver 1

    x_recorder_2 = x_recorder_1 + n_d
    y_recorder_2 = y_recorder_1  # location receiver 2

    x_recorder_3 = x_recorder_1 + 2 * n_d
    y_recorder_3 = y_recorder_1  # location receiver 3

    # initialisation of o and p fields
    length = 4 * n_d + 2 * n_PML
    height = n_edge + n_d + n_PML
    p = np.zeros((height, length))
    px = np.zeros_like(p)
    py = np.zeros_like(p)
    ox = np.zeros((height, length + 1))
    oy = np.zeros((height + 1, length))

    kappa_max = kappa_max_factor / dt

    kappa_px = np.zeros_like(p)
    kappa_py = np.zeros_like(kappa_px)
    kappa_ox = np.zeros_like(ox)
    kappa_oy = np.zeros_like(oy)

    for i in range(n_PML):
        kappa_px[:, i] = np.full(
            height, kappa(d=(n_PML - i) * dx, kappa_max=kappa_max, d_PML=dx * n_PML)
        )
        kappa_px[:, -i] = np.full(
            height, kappa(d=(n_PML - i) * dx, kappa_max=kappa_max, d_PML=dx * n_PML)
        )
        kappa_py[i] = np.full(
            length, kappa(d=(n_PML - i) * dx, kappa_max=kappa_max, d_PML=dx * n_PML)
        )
        kappa_py[-i] = np.full(
            length, kappa(d=(n_PML - i) * dx, kappa_max=kappa_max, d_PML=dx * n_PML)
        )
    kappa_ox[:, :n_PML] = kappa_px[:, :n_PML]
    kappa_ox[:, -n_PML:] = kappa_px[:, -n_PML:]
    kappa_oy[:n_PML, :] = kappa_py[:n_PML, :]
    kappa_oy[-n_PML:, :] = kappa_py[-n_PML:, :]

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

    print("max same damping:", 1 - (1 - kappa_max * dt / 2) / (1 + kappa_max * dt / 2))
    print("max diff damping: ", 1 - 1 / (1 + kappa_max * dt / 2))

    # initialisation time series receivers
    recorder_1 = np.zeros((n_t, 1))
    recorder_1_ref = np.zeros_like(recorder_1)
    recorder_2 = np.zeros_like(recorder_1)
    recorder_2_ref = np.zeros_like(recorder_2)
    recorder_3 = np.zeros_like(recorder_1)
    recorder_3_ref = np.zeros_like(recorder_3)
    source_recorder = np.zeros_like(recorder_1)
    source_recorder_ref = np.zeros_like(source_recorder)

    timesteps = np.linspace(0, n_t * dt, n_t)
    source_vals = source(timesteps, A=A, fc=fc, t0=t0, sigma=sigma)
    update_source_vals = np.append(source_vals, 0.0)

    # TIME ITTERATION----------------------------------------------------
    fig, ax = plt.subplots()
    plt.axis("equal")
    movie = []
    for i in range(0, n_t):
        t = (i - 1) * dt
        timesteps[i] = t
        print("%d/%d" % (i + 1, n_t), end="\r")

        # propagate over one time step

        # adding source term to propagation
        px[y_source, x_source] += source_vals[i]
        # lower side
        px[:n_edge, :n_edge] = (
            dt
            / dx
            * c**2
            / (1 + kappa_px[:n_edge, :n_edge] * dt / 2)
            * (ox[:n_edge, :n_edge] - ox[:n_edge, 1 : n_edge + 1])
        ) + px[:n_edge, :n_edge] * (1 - kappa_px[:n_edge, :n_edge] * dt / 2) / (
            1 + kappa_px[:n_edge, :n_edge] * dt / 2
        )
        py[:n_edge, :n_edge] = (
            dt
            / dy
            * c**2
            / (1 + kappa_py[:n_edge, :n_edge] * dt / 2)
            * (oy[:n_edge, :n_edge] - oy[1 : n_edge + 1, :n_edge])
        ) + py[:n_edge, :n_edge] * (1 - kappa_py[:n_edge, :n_edge] * dt / 2) / (
            1 + kappa_py[:n_edge, :n_edge] * dt / 2
        )
        # upper side
        px[n_edge:, :] = (
            dt
            / dx
            / (1 + kappa_px[n_edge:, :] * dt / 2)
            * c**2
            * (ox[n_edge:, :-1] - ox[n_edge:, 1:])
        ) + px[n_edge:, :] * (1 - kappa_px[n_edge:, :] * dt / 2) / (
            1 + kappa_px[n_edge:, :] * dt / 2
        )
        py[n_edge:, :] = (
            dt
            / dy
            / (1 + kappa_py[n_edge:, :] * dt / 2)
            * c**2
            * (oy[n_edge:-1, :] - oy[n_edge + 1 :, :])
        ) + py[n_edge:, :] * (1 - kappa_py[n_edge:, :] * dt / 2) / (
            1 + kappa_py[n_edge:, :] * dt / 2
        )

        # combine px and py for easy acces to update o fields
        p = px + py

        # lower side
        ox[:n_edge, 1:n_edge] = (
            dt
            / dx
            / (1 + kappa_ox[:n_edge, 1:n_edge] * dt / 2)
            * (p[:n_edge, : n_edge - 1] - p[:n_edge, 1:n_edge])
        ) + ox[:n_edge, 1:n_edge] * (1 - kappa_ox[:n_edge, 1:n_edge] * dt / 2) / (
            1 + kappa_ox[:n_edge, 1:n_edge] * dt / 2
        )
        oy[1:n_edge, :n_edge] = (
            dt
            / dy
            / (1 + kappa_oy[1:n_edge, :n_edge] * dt / 2)
            * (p[: n_edge - 1, :n_edge] - p[1:n_edge, :n_edge])
        ) + oy[1:n_edge, :n_edge] * (1 - kappa_oy[1:n_edge, :n_edge] * dt / 2) / (
            1 + kappa_oy[1:n_edge, :n_edge] * dt / 2
        )

        # upper side
        ox[n_edge:, 1:-1] = (
            dt
            / dx
            / (1 + kappa_ox[n_edge:, 1:-1] * dt / 2)
            * (p[n_edge:, :-1] - p[n_edge:, 1:])
        ) + ox[n_edge:, 1:-1] * (1 - kappa_ox[n_edge:, 1:-1] * dt / 2) / (
            1 + kappa_ox[n_edge:, 1:-1] * dt / 2
        )
        oy[n_edge:-1, :] = (
            dt
            / dy
            / (1 + kappa_oy[n_edge:-1, :] * dt / 2)
            * (p[n_edge - 1 : -1, :] - p[n_edge:, :])
        ) + oy[n_edge:-1, :] * (1 - kappa_oy[n_edge:-1, :] * dt / 2) / (
            1 + kappa_oy[n_edge:-1, :] * dt / 2
        )

        # edges 'behind' PML
        ox[:, :1] = (
            (1 - kappa_ox[:, :1] * dt / 2 - Z_r * dt / dx) * ox[:, :1]
            - 2 * dt / dx * p[:, :1]
        ) / (1 + kappa_ox[:, :1] * dt / 2 + dt / dx * Z_r)

        ox[n_edge:, -1:] = (
            (1 - kappa_ox[n_edge:, -1:] * dt / 2 - Z_r * dt / dx) * ox[n_edge:, -1:]
            + 2 * dt / dx * p[n_edge:, -1:]
        ) / (1 + kappa_ox[n_edge:, -1:] * dt / 2 + dt / dx * Z_r)

        oy[:1, :n_edge] = (
            (1 - kappa_oy[:1, :n_edge] * dt / 2 - Z_r * dt / dy) * oy[:1, :n_edge]
            - 2 * dt / dy * p[:1, :n_edge]
        ) / (1 + kappa_oy[:1, :n_edge] * dt / 2 + dt / dy * Z_r)

        oy[-1:, :] = (
            (1 - kappa_oy[-1:, :] * dt / 2 - Z_r * dt / dy) * oy[-1:, :]
            + 2 * dt / dy * p[-1:, :]
        ) / (1 + kappa_oy[-1:, :] * dt / 2 + dt / dy * Z_r)

        # edges of wedge
        ox[:n_edge, n_edge : n_edge + 1] = (1 - Z_0 * dt / dx + 2 * Z_1 / dx) / (
            1 + Z_0 * dt / dx + 2 * Z_1 / dx
        ) * ox[:n_edge, n_edge : n_edge + 1] + 2 * dt / dx / (
            1 + Z_0 * dt / dx + 2 * Z_1 / dx
        ) * p[
            :n_edge, n_edge - 1 : n_edge
        ]

        oy[n_edge - 1 : n_edge, n_edge:] = (1 - Z_0 * dt / dy + 2 * Z_1 / dy) / (
            1 + Z_0 * dt / dy + 2 * Z_1 / dy
        ) * oy[n_edge - 1 : n_edge, n_edge:] + 2 * dt / dy / (
            1 + Z_0 * dt / dy + 2 * Z_1 / dy
        ) * p[
            n_edge : n_edge + 1, n_edge:
        ]
        # adding source term to propagation for plotting
        p[y_source, x_source] = update_source_vals[i + 1]

        # store p field at receiver locations
        recorder_1[i] = p[y_recorder_1, x_recorder_1]
        recorder_1_ref[i] = p[y_recorder_1 + 1, x_recorder_1 + 1]
        recorder_2[i] = p[y_recorder_2, x_recorder_2]
        recorder_2_ref[i] = p[y_recorder_2 + 1, x_recorder_2 + 1]
        recorder_3[i] = p[y_recorder_3, x_recorder_3]
        recorder_3_ref[i] = p[y_recorder_3 + 1, x_recorder_3 + 1]
        source_recorder[i] = p[y_source, x_source]
        source_recorder_ref[i] = p[y_source + 1, x_source + 1]
        # presenting the p field
        plot_factor_A = 0.001 * A
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
                    p,
                    vmin=-plot_factor_A,
                    vmax=plot_factor_A,
                    origin="lower",
                ),
                ax.plot(x_source, y_source, "ks", fillstyle="none", label="Source")[0],
                ax.plot(
                    x_recorder_1,
                    y_recorder_1,
                    "ro",
                    fillstyle="none",
                    label="Recorders",
                )[0],
                ax.plot(x_recorder_2, y_recorder_2, "ro", fillstyle="none")[0],
                ax.plot(x_recorder_3, y_recorder_3, "ro", fillstyle="none")[0],
                ax.plot(
                    [n_edge, n_edge, length], [0, n_edge, n_edge], "k-", label="Wedge"
                )[0],
                ax.plot(
                    [n_edge, n_PML, n_PML, length - n_PML, length - n_PML],
                    [n_PML, n_PML, height - n_PML, height - n_PML, n_edge],
                    "w:",
                    label="PML border",
                )[0],
                # ax.legend(),
                # ax.set_ylim(0,height+15)
            ]
            movie.append(artists)

    if make_movie == True:
        my_anim = ArtistAnimation(
            fig, movie, interval=5 * 50 / n_t, repeat_delay=1000, blit=True
        )
        # my_anim.save("FDTD_cartesian.gif",fps=60)
        plt.show()
        # play back the stored movie
    plt.close()
    plt.plot(timesteps, recorder_1, "r-", label="Recorder 1")
    plt.plot(timesteps, recorder_2, "g-", label="Recorder 2")
    plt.plot(timesteps, recorder_3, "m-", label="Recorder 3")
    plt.ylabel("p field")
    plt.xlabel("Time [s]")
    plt.legend()
    plt.savefig("recordings.png")
    if show_plots == True:
        plt.show()
    plt.close()
    plt.plot(
        timesteps,
        source(t=timesteps, A=A, fc=fc, t0=t0, sigma=sigma),
        "b-",
        label="Original source",
    )
    plt.plot(timesteps, source_recorder, "r-", label="Recorded source")
    plt.ylabel("p field")
    plt.xlabel("Time [s]")
    plt.legend()
    plt.savefig("source.png")
    if show_plots == True:
        plt.show()
    plt.close()
    n_of_samples = 8192

    post_Afout_Pfout(dx,dy,c,dt,d,(x_recorder_1+1)*dx,x_recorder_1*dx,(y_recorder_1+1)*dy,y_recorder_1*dy,recorder_1,recorder_1_ref,n_of_samples,timesteps,fc)
    if execute_post_processing == True:
        
        post_processing(
            dx,
            dy,
            c,
            dt,
            source_recorder,
            source_recorder_ref,
            n_of_samples,
            timesteps,
            fc,
        )
        post_processing(
            dx, dy, c, dt, recorder_1, recorder_1_ref, n_of_samples, timesteps, fc
        )
        post_processing(
            dx, dy, c, dt, recorder_2, recorder_2_ref, n_of_samples, timesteps, fc
        )
        post_processing(
            dx, dy, c, dt, recorder_3, recorder_3_ref, n_of_samples, timesteps, fc
        )

    return_array = np.zeros_like(p)
    return_array[n_PML:n_edge, n_PML:n_edge] = p[n_PML:n_edge, n_PML:n_edge]
    return_array[n_edge:-n_PML, n_PML:-n_PML] = p[n_edge:-n_PML, n_PML:-n_PML]
    return np.abs(return_array) * dx**2


"""
    # NAVERWERKING : BEREKENING FASEFOUT en AMPLITUDEFOUT---------------------------------
    # POST PROCESSING : CALCULATE PHASE and AMPLITUDE ERROR-------------------------------
    n_of_samples = 8192

    post_Afout_Pfout(
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
    post_Afout_Pfout(
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
FDTD(show_plots=True,make_movie=True)