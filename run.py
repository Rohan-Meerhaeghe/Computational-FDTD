from main import FDTD

FDTD(
    n_PML=25,
    n_d=75,
    make_movie=False,
    plot_kappa=False,
    T=0.15,
    show_plots=True,
    reactive_wedge=False,
)
