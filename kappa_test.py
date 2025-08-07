import numpy as np
import matplotlib.pyplot as plt
from main import FDTD

d_val = 5.0
i_n_d_max = 1
i_kappa_max = 50
kappa_max_vals = np.linspace(0.01, 1.0, i_kappa_max)
i_n_d_vals = np.linspace(50, (i_n_d_max) * 50, i_n_d_max)

"""sum_results = [
    [
        0.01584707,
        0.00954548,
        0.00611366,
        0.00455086,
        0.00429986,
        0.00442661,
        0.00463665,
        0.00482566,
        0.00496494,
        0.0050557,
        0.00510931,
        0.00513996,
        0.00515925,
        0.00517383,
        0.00518654,
        0.00519833,
        0.00520954,
        0.00522012,
        0.00523028,
        0.0052399,
    ],
    [
        0.00522962,
        0.00244926,
        0.00177482,
        0.00145795,
        0.0013601,
        0.00142809,
        0.00155954,
        0.00167836,
        0.00175509,
        0.00179124,
        0.00180591,
        0.0018157,
        0.00182775,
        0.00184234,
        0.00185787,
        0.00187324,
        0.00188802,
        0.00190217,
        0.00191575,
        0.00192884,
    ],
    [
        0.002957,
        0.00152958,
        0.00119964,
        0.0009494,
        0.00086504,
        0.00089938,
        0.00098559,
        0.00105892,
        0.00110494,
        0.0011191,
        0.00112199,
        0.00112673,
        0.00113611,
        0.00114799,
        0.00116044,
        0.00117261,
        0.00118426,
        0.00119541,
        0.00120613,
        0.00121649,
    ],
    [
        0.00189983,
        0.00115449,
        0.00090439,
        0.00071171,
        0.00064617,
        0.00067156,
        0.00073131,
        0.000783,
        0.00081474,
        0.00081988,
        0.00081878,
        0.00082172,
        0.00082907,
        0.00083827,
        0.00084783,
        0.0008572,
        0.00086621,
        0.00087486,
        0.0008832,
        0.00089127,
    ],
]

plt.plot(
    kappa_max_vals,
    sum_results[0] / np.max(sum_results[0]),
    "r.-",
    label="dx = %f" % (d_val / 50),
)
plt.plot(
    kappa_max_vals,
    sum_results[1] / np.max(sum_results[1]),
    "g.-",
    label="dx = %f" % (d_val / 75),
)
plt.plot(
    kappa_max_vals,
    sum_results[2] / np.max(sum_results[2]),
    "b.-",
    label="dx = %f" % (d_val / 100),
)
plt.plot(
    kappa_max_vals,
    sum_results[3] / np.max(sum_results[3]),
    "c.-",
    label="dx = %f" % (d_val / 150),
)
plt.legend()
plt.xlabel(r"$\kappa_{max} \times dt [dimensionless]$")
plt.ylabel("Normalized integration of the p-field [arbitrary units]")
plt.savefig("kappa_zoomed_out.png")
#plt.show()
plt.close()

plt.plot(
    kappa_max_vals,
    sum_results[0],
    "r.-",
    label="dx = %f" % (d_val / 50),
)
plt.plot(
    kappa_max_vals,
    sum_results[1],
    "g.-",
    label="dx = %f" % (d_val / 75),
)
plt.plot(
    kappa_max_vals,
    sum_results[2],
    "b.-",
    label="dx = %f" % (d_val / 100),
)
plt.plot(
    kappa_max_vals,
    sum_results[3],
    "c.-",
    label="dx = %f" % (d_val / 150),
)
plt.legend()
plt.xlabel(r"$\kappa_{max} \times dt [dimensionless]$")
plt.ylabel("Integration of the p-field [arbitrary units]")
plt.savefig("kappa_zoomed_out_not_normalized.png")
# plt.show()
plt.close()

sum_results = np.array(
    [
        [
            0.0111264,
            0.00958481,
            0.0086425,
            0.00798976,
            0.00742817,
            0.00689363,
            0.0063801,
            0.00590668,
            0.00549605,
            0.00516209,
            0.00490606,
            0.00471982,
            0.00459165,
            0.00451013,
            0.00446572,
            0.00445005,
            0.00445537,
            0.0044757,
            0.00450535,
            0.00454008,
        ],
        [
            0.00570591,
            0.00529321,
            0.00483453,
            0.0042704,
            0.00368067,
            0.00315597,
            0.00273295,
            0.00242779,
            0.00224541,
            0.0021487,
            0.00209873,
            0.00207342,
            0.00205887,
            0.00204764,
            0.00203835,
            0.00203054,
            0.00202542,
            0.00202551,
            0.00203284,
            0.0020454,
        ],
        [
            0.00400795,
            0.00381054,
            0.00350104,
            0.00307438,
            0.00258812,
            0.0021223,
            0.00177266,
            0.00155902,
            0.00145768,
            0.00140934,
            0.00138214,
            0.00136158,
            0.00134366,
            0.0013275,
            0.00131077,
            0.00129483,
            0.00128453,
            0.00128427,
            0.00129403,
            0.00130898,
        ],
        [
            0.00300355,
            0.00289807,
            0.00267732,
            0.0023524,
            0.00197278,
            0.00158748,
            0.00129761,
            0.00114411,
            0.00108002,
            0.00104877,
            0.00102634,
            0.00100731,
            0.00099019,
            0.00097387,
            0.00095471,
            0.00093796,
            0.0009299,
            0.00093422,
            0.00094684,
            0.00096335,
        ],
    ]
)
"""
kappa_max_vals = np.linspace(0.01, 5.0, i_kappa_max)

sum_results = np.zeros((4,50))
for i_n_d in range(i_n_d_max):
    for i_kappa in range(i_kappa_max):
        sum_results[i_n_d, i_kappa] = np.sum(
            FDTD(
                n_PML=25,
                n_d=(i_n_d + 1) * 50,
                make_movie=False,
                plot_kappa=False,
                kappa_max_factor=kappa_max_vals[i_kappa],
                T=0.14,
                d=d_val,
                reactive_wedge=False,
            )
        )
        print(
            "\n"
            + str(1 + i_n_d * i_kappa_max + i_kappa)
            + "/"
            + str(i_kappa_max * i_n_d_max)
            + "\n"
        )

print(sum_results)

kappa_max_vals = np.linspace(0.01, 0.3, i_kappa_max)

plt.plot(
    kappa_max_vals[4:],
    (sum_results[0][4:] / np.max(sum_results[0][4:])),
    "r.-",
    label="dx = %f" % (d_val / 50),
)
plt.plot(
    kappa_max_vals[4:],
    sum_results[1][4:] / np.max(sum_results[1][4:]),
    "g.-",
    label="dx = %f" % (d_val / 75),
)
plt.plot(
    kappa_max_vals[4:],
    sum_results[2][4:] / np.max(sum_results[2][4:]),
    "b.-",
    label="dx = %f" % (d_val / 100),
)
plt.plot(
    kappa_max_vals[4:],
    sum_results[3][4:] / np.max(sum_results[3][4:]),
    "c.-",
    label="dx = %f" % (d_val / 150),
)
plt.legend()
plt.xlabel(r"$\kappa_{max} \times dt [dimensionless]$")
plt.ylabel("Normalized integration of the p-field [arbitrary units]")
plt.savefig("kappa_zoomed_in.png")
# plt.show()
plt.close()


for i in range(4):
    print(
        np.min(sum_results[i]),
        np.where(sum_results[i] == np.min(sum_results[i]))[0][0],
        kappa_max_vals[np.where(sum_results[i] == np.min(sum_results[i]))[0][0]],
    )
