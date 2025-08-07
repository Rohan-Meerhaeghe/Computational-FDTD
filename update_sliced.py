for i in range(n_PML):
    kappa_left[i] = np.full(
        height, kappa(d=(n_PML - i) * dx, kappa_max=kappa_max, d_PML=dx * n_PML)
    )
    kappa_right[i] = np.full(
        n_PML + n_d,
        kappa(d=(i + 1) * dx, kappa_max=kappa_max, d_PML=dx * n_PML),
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



def update_func_sliced(px,py, ox,oy,n_PML, n_edge,dt,dx,dy,c):
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