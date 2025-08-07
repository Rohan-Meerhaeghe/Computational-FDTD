import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def distance(p0, p1):
    return np.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)


def f_source(t, f0, t0, sigma):
    return (t - t0) * np.exp(-((t - t0) ** 2) / sigma**2)


def FDTD_cartesian(a, d, c, N_x=500, N_t=200, play_animation=False):

    length = 14 * a
    height = max(3 * d, a)
    d_PML = 40
    N_y = round(
        height / length * N_x
    )  # The number of horizontal cells for a rectangular grid
    dx = length / N_x
    dy = height / N_y
    N_y += d_PML  # making sure there is space for the PML
    height = dy * N_y
    n_bottom = 3  # elevate geometry to avoid extra edge terms
    dt = 1 / (np.sqrt(1.01) * c * np.sqrt(1 / dx**2 + 1 / dy**2))  # Courant limit
    source = (
        round(23 / 8 * a / length * N_x),
        round(0.25 * a / height * N_y) + n_bottom,
    )
    probe = (
        round(87 / 8 * a / length * N_x),
        round(0.25 * a / height * N_y) + n_bottom,
    )
    second_probe = (3, round(N_y / 2))

    p_recording = np.zeros(N_t)
    p2_recording = np.zeros(N_t)
    s_recording = np.zeros(N_t)
    true_source = np.zeros_like(s_recording)
    kappa_max = 2 / dt
    print("first factor", 1 / (1 + kappa_max * dt / 2))
    print("second factor", (1 - kappa_max * dt / 2) / (1 + kappa_max * dt / 2))

    def kappa(d, m=5):
        return kappa_max * (d / d_PML) ** m

    Z = 8 * c
    Zr = c
    p = np.zeros((N_x, N_y))
    p_x = np.zeros_like(p)
    p_y = np.zeros_like(p)
    ox = np.zeros((N_x + 1, N_y))
    oy = np.zeros((N_x, N_y + 1))
    kappa_x = np.zeros((d_PML, N_y))
    kappa_y = np.zeros((N_x, d_PML))
    kappa_y = np.transpose(kappa_y)

    for i in range(d_PML):
        kappa_x[i] = np.full(N_y, kappa(d_PML - i))
        kappa_y[i] = np.full(N_x, kappa(i + 1))
    kappa_y = np.transpose(kappa_y)
    kappa_x_flipped = kappa_x[::-1, :]

    inside = np.zeros((N_x, N_y))
    inside_oy = np.zeros_like(oy)
    edge_oy = np.zeros_like(oy)
    u_edge_oy = np.zeros_like(oy)
    d_edge_oy = np.zeros_like(oy)
    inside_ox = np.zeros_like(ox)
    edge_ox = np.zeros_like(ox)
    r_edge_ox = np.zeros_like(ox)
    l_edge_ox = np.zeros_like(ox)
    inside_dots = np.zeros_like(inside)
    print(c**2 * dt**2 * (1 / dx**2 + 1 / dy**2))

    for i in range(N_x):
        edge_val = d / 2 * (1 + np.sin(4 * np.pi / a * i * dx)) + n_bottom * dy
        for j in range(N_y):
            if distance((i, j), source) < 5:
                inside_dots[i, j] = 2
            if distance((i, j), probe) < 5:
                inside_dots[i, j] = 3
            if j * dy > edge_val:
                inside[i, j] = True
            else:
                inside[i, j] = False
    inside_dots += inside
    plt.imshow(np.transpose(inside_dots), origin="lower")
    plt.show()

    for i in range(N_x - 1):
        for j in range(N_y):
            if inside[i, j] == 1 and inside[i + 1, j] == 0:
                edge_ox[i + 1, j] = 1
                r_edge_ox[i + 1, j] = 1
            if inside[i, j] == 0 and inside[i + 1, j] == 1:
                edge_ox[i + 1, j] = 1
                l_edge_ox[i + 1, j] = 1
            if inside[i, j] == 1 and inside[i + 1, j] == 1:
                inside_ox[i + 1, j] = 1
    for i in range(N_x):
        for j in range(N_y - 1):
            if inside[i, j] == 1 and inside[i, j + 1] == 0:
                edge_oy[i, j + 1] = 1
                u_edge_oy[i, j + 1] = 1
            if inside[i, j] == 0 and inside[i, j + 1] == 1:
                edge_oy[i, j + 1] = 1
                d_edge_oy[i, j + 1] = 1
            if inside[i, j] == 1 and inside[i, j + 1] == 1:
                inside_oy[i, j + 1] = 1

    def update(p_x, p_y, ox, oy):

        p_x[d_PML:-d_PML, :] += (
            -dt / dx * c**2 * (ox[1 + d_PML : -d_PML, :] - ox[d_PML : -1 - d_PML, :])
        )
        p_x[:d_PML, :] = (1 - kappa_x * dt / 2) / (1 + kappa_x * dt / 2) * p_x[
            :d_PML, :
        ] - dt / dx * c**2 * (ox[1 : d_PML + 1, :] - ox[:d_PML, :]) / (
            1 + kappa_x * dt / 2
        )
        p_x[-d_PML:, :] = (1 - kappa_x_flipped * dt / 2) / (
            1 + kappa_x_flipped * dt / 2
        ) * p_x[-d_PML:, :] - dt / dx * c**2 * (
            ox[-d_PML:, :] - ox[-1 - d_PML : -1, :]
        ) / (
            1 + kappa_x_flipped * dt / 2
        )

        p_y[:, :-d_PML] += -dt / dy * c**2 * (oy[:, 1:-d_PML] - oy[:, : -1 - d_PML])
        p_y[:, -d_PML:] = (1 - kappa_y * dt / 2) / (1 + kappa_y * dt / 2) * p_y[
            :, -d_PML:
        ] - dt / dy * c**2 * (oy[:, -d_PML:] - oy[:, -1 - d_PML : -1]) / (
            1 + kappa_y * dt / 2
        )

        ox[d_PML:-d_PML, :] += (
            -dt
            / dx
            * (
                p_x[d_PML : -(d_PML - 1), :]
                + p_y[d_PML : -(d_PML - 1), :]
                - p_x[d_PML - 1 : -d_PML, :]
                - p_y[d_PML - 1 : -d_PML, :]
            )
            * inside_ox[d_PML:-d_PML]
        )
        -2 * Z * dt / (dx + Z * dt) * ox[d_PML:-d_PML, :] * edge_ox[d_PML:-d_PML]
        +2 * dt / (dx + Z * dt) * (
            p_x[d_PML - 1 : -d_PML, :] + p_y[d_PML - 1 : -d_PML, :]
        ) * r_edge_ox[d_PML:-d_PML, :]
        -2 * dt / (dx + Z * dt) * (
            p_x[d_PML : -(d_PML - 1), :] + p_y[d_PML : -(d_PML - 1), :]
        ) * l_edge_ox[d_PML:-d_PML, :]

        ox[1:d_PML, :] = (
            (1 - kappa_x[1:, :] * dt / 2)
            / (1 + kappa_x[1:, :] * dt / 2)
            * ox[1:d_PML, :]
            - dt
            / dx
            * (
                p_x[1:d_PML, :]
                + p_y[1:d_PML, :]
                - p_x[: d_PML - 1, :]
                - p_y[: d_PML - 1, :]
            )
            / (1 + kappa_x[1:, :] * dt / 2)
        ) * inside_ox[1:d_PML, :]
        +(
            (1 - kappa_x[1:, :] * dt / 2 - Z * dt / dx)
            / (1 + kappa_x[1:, :] * dt / 2 + Z * dt / dx)
        ) * ox[1:d_PML, :] * edge_ox[1:d_PML, :]
        +2 * dt / dx / (1 + kappa_x[1:d_PML, :] * dt / 2 + Z * dt / dx) * (
            p_x[: d_PML - 1, :] + p_y[: d_PML - 1, :]
        ) * r_edge_ox[1:d_PML, :]
        -2 * dt / dx / (1 + kappa_x[1:d_PML, :] * dt / 2 + Z * dt / dx) * (
            p_x[1:d_PML, :] + p_y[1:d_PML, :]
        ) * l_edge_ox[1:d_PML, :]

        ox[0] = (dx - Zr * dt) / (dx + Zr * dt) * ox[0] - 2 * dt / (dx + Zr * dt) * (
            p_x[0] + p_y[0]
        )
        ox[-d_PML:-1, :] = (
            (1 - kappa_x_flipped[1:, :] * dt / 2)
            / (1 + kappa_x_flipped[1:, :] * dt / 2)
            * ox[-d_PML:-1, :]
            - dt
            / dx
            * (
                p_x[-(d_PML - 1) :, :]
                + p_y[-(d_PML - 1) :, :]
                - p_x[-d_PML:-1, :]
                - p_y[-d_PML:-1, :]
            )
            / (1 + kappa_x_flipped[1:, :] * dt / 2)
        ) * inside_ox[-d_PML:-1, :]
        +(
            (1 - kappa_x_flipped[1:, :] * dt / 2 - Z * dt / dx)
            / (1 + kappa_x_flipped[1:, :] * dt / 2 + Z * dt / dx)
        ) * ox[-d_PML:-1, :] * edge_ox[-d_PML:-1, :]
        +2 * dt / dx / (1 + kappa_x_flipped[1:d_PML, :] * dt / 2 + Z * dt / dx) * (
            p_x[-d_PML:-1, :] + p_y[-d_PML:-1, :]
        ) * r_edge_ox[-d_PML:-1, :]
        -2 * dt / dx / (1 + kappa_x_flipped[1:d_PML, :] * dt / 2 + Z * dt / dx) * (
            p_x[-(d_PML - 1) :, :] + p_y[-(d_PML - 1) :, :]
        ) * l_edge_ox[-d_PML:-1, :]
        ox[-1] = (dx - Zr * dt) / (dx + Zr * dt) * ox[-1] + 2 * dt / (dx + Zr * dt) * (
            p_x[-1] + p_y[-1]
        )

        oy[:, 1:-d_PML] += (
            -dt
            / dy
            * (
                p_x[:, 1 : -(d_PML - 1)]
                + p_y[:, 1 : -(d_PML - 1)]
                - p_x[:, :-d_PML]
                - p_y[:, :-d_PML]
            )
            * inside_oy[:, 1:-d_PML]
        )
        -2 * Z * dt / (dy + Z * dt) * oy[:, 1:-d_PML] * edge_oy[:, 1:-d_PML]
        -2 * dt / (dy + Z * dt) * (p_x[:, :-d_PML] + p_y[:, :-d_PML]) * u_edge_oy[
            :, 1:-d_PML
        ]
        +2 * dt / (dy + Z * dt) * (
            p_x[:, 1 : -(d_PML - 1)] + p_y[:, 1 : -(d_PML - 1)]
        ) * d_edge_oy[:, 1:-d_PML]

        oy[:, 0] = (dy - Zr * dt) / (dy + Zr * dt) * oy[:, 0] - 2 * dt / (
            dy + Zr * dt
        ) * (p_x[:, 0] + p_y[:, 0])
        oy[:, -d_PML:-1] = (
            (1 - kappa_y[:, 1:] * dt / 2)
            / (1 + kappa_y[:, 1:] * dt / 2)
            * oy[:, -d_PML:-1]
            - dt
            / dy
            * (
                p_x[:, -(d_PML - 1) :]
                + p_y[:, -(d_PML - 1) :]
                - p_x[:, -d_PML:-1]
                - p_y[:, -d_PML:-1]
            )
            / (1 + kappa_y[::-1, 1:] * dt / 2)
        ) * inside[:, -d_PML:-1]
        +(1 - kappa_y[:, 1:] * dt / 2 - Z * dt / dx) / (
            1 + kappa_y[:, 1:] * dt / 2 + Z * dt / dx
        ) * oy[:, -d_PML:-1] * edge_oy[:, -d_PML:-1]
        -2 * dt / dy / (1 + kappa_y[:, 1:] * dt / 2 + Z * dt / dx) * (
            p_x[:, -d_PML:-1] + p_y[:, -d_PML:-1]
        ) * u_edge_oy[:, -d_PML:-1]
        +2 * dt / dy / (1 + kappa_y[:, 1:] * dt / 2 + Z * dt / dx) * (
            p_x[:, -(d_PML - 1) :] + p_y[:, -(d_PML - 1) :]
        ) * d_edge_oy[:, -d_PML:-1]
        oy[:, -1] = (dy - Zr * dt) / (dy + Zr * dt) * oy[:, -1] + 2 * dt / (
            dy + Zr * dt
        ) * (p_x[:, -1] + p_y[:, -1])

        p_x[(source)] += f_source(i * dt, 5 * c / np.pi, 50 * dt, 20 * dt)

        return p_x + p_y, p_x, p_y, ox, oy

    print("Calculating...")

    if play_animation == False:
        for i in range(N_t):
            p, p_x, p_y, ox, oy = update(p_x, p_y, ox, oy)

            true_source[i] = f_source(i * dt, 5 * c / np.pi, 50 * dt, 20 * dt)
            p_recording[i] = p[(probe)]
            p2_recording[i] = p[(second_probe)]
            s_recording[i] = p[(source)]

        print("Computation done, now plotting...")
        print("shape true source: ", true_source.shape)
        print("shape fourier transform: ", np.fft.fft(true_source).shape)

        fig, ax1 = plt.subplots()

        color = "tab:red"
        ax1.set_xlabel("time")
        ax1.set_ylabel("source", color=color)
        ax1.plot(np.arange(N_t), s_recording, color=color)
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
        color = "tab:blue"
        ax2.set_ylabel("probe", color=color)  # we already handled the x-label with ax1
        ax2.plot(np.arange(N_t), p_recording, color=color)
        ax2.tick_params(axis="y", labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()
    else:
        for i in range(0, N_t):
            true_source[i] = f_source(i * dt, 5 * c / np.pi, 50 * dt, 20 * dt)
        max_val = np.max(np.abs(true_source))
        print("maxval = ", max_val)
        fig, ax = plt.subplots()
        plt.xlim([1, N_x])
        plt.ylim([1, N_y])
        movie = []
        max_frame_val = 0.01 * max_val
        for i in range(N_t):

            p, p_x, p_y, ox, oy = update(p_x, p_y, ox, oy)
            # max_frame_val = max(np.max(np.abs(p)),0.01*max_val)

            artists = [
                ax.imshow(
                    np.transpose(p),
                    vmin=-max_frame_val,
                    vmax=max_frame_val,
                    origin="lower",
                ),
            ]
            movie.append(artists)
        print("Computation done, now plotting...")
        my_anim = animation.ArtistAnimation(
            fig, movie, interval=10, repeat_delay=1000, blit=True
        )
        my_anim.save("FDTD_cartesian.gif", fps=30)
        plt.show()


FDTD_cartesian(1, 1, 53, 500, 1000)
