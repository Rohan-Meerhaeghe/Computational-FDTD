import numpy as np
import matplotlib.pyplot as plt

def source_freq(omega, sigma, t_0, omega_0):
    return np.abs(np.sqrt(np.pi)*sigma/(2j)*np.exp(-1j*t_0*omega)*(np.exp(-(omega-omega_0)**2*sigma**2/4)-np.exp(-(omega+omega_0)**2*sigma**2/4)))

c = 343.0
d = 5.0

omega_vals = np.linspace(0.4*c/d,4*c/d,100)
F_vals = source_freq(omega_vals,1/50,4.5/100,2*c/d)
Omega_vals = np.linspace(0.04*c/d,5*c/d,50)
FF_vals = source_freq(Omega_vals,1/50,4.5/100,2*c/d)

plt.plot(Omega_vals,FF_vals,'b',linestyle='dotted')
plt.plot(omega_vals,F_vals,'b')
plt.xlabel(r'$\omega$')
plt.ylabel(r'$|F(\omega)|$')
plt.savefig("source_freq.png")
plt.show()