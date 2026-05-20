#Tabella 4.2 tesi di Mattia (dati del Lambda) e ultima colonna della tabella inviata da Vittorio (Pixirad)

import numpy as np
import matplotlib.pyplot as plt

E = np.array([15.5, 20, 26, 29, 33, 37], dtype=float)
FWHM = np.array([3.5, 3.6, 4.3, 4.8, 5.4, 6.4], dtype=float)
errors = 0.2 * np.ones_like(E)  # Assuming 20% error on FWHM measurements
# fit lineare y = a*x + b
N = len(E)
sum_x = np.sum(E)
sum_y = np.sum(FWHM)
sum_x2 = np.sum(E**2)
sum_xy = np.sum(E * FWHM)

a_lin = (N * sum_xy - sum_x * sum_y) / (N * sum_x2 - sum_x**2)
b_lin = (sum_y - a_lin * sum_x) / N
print(" FIT LINEARE ")
print(f"a = {a_lin}")
print(f"b = {b_lin}")

# fit quadratico y = a*x^2 + b*x + c
sum_x3 = np.sum(E**3)
sum_x4 = np.sum(E**4)
sum_x2y = np.sum((E**2) * FWHM)

A = np.array([
    [sum_x4, sum_x3, sum_x2],
    [sum_x3, sum_x2, sum_x],
    [sum_x2, sum_x,  N]
])

B = np.array([
    sum_x2y,
    sum_xy,
    sum_y
])

# Risoluzione sistema lineare
coeff = np.linalg.solve(A, B)
a_quad = coeff[0]
b_quad = coeff[1]
c_quad = coeff[2]

print(" FIT QUADRATICO ")
print(f"a = {a_quad}")
print(f"b = {b_quad}")
print(f"c = {c_quad}")

E_fit = np.linspace(15, 55, 300)

# Retta
F_fit_lin = a_lin * E_fit + b_lin

# Parabola
F_fit_quad = (
    a_quad * E_fit**2
    + b_quad * E_fit
    + c_quad
)

plt.figure(figsize=(8,5))
plt.scatter(E, FWHM, label="Dati sperimentali")
plt.scatter(E, FWHM + errors, label="Dati sperimentali + errori", alpha=0.5)
plt.scatter(E, FWHM - errors, label="Dati sperimentali - errori", alpha=0.5)
plt.plot(E_fit, F_fit_lin, label="Fit lineare")
plt.plot(E_fit, F_fit_quad, label="Fit quadratico")
plt.xlabel("Energia (keV)")
plt.ylabel("FWHM (keV)")
plt.title("Risoluzione energetica Lambda")
plt.grid(True)
plt.legend()
plt.show()

# CHI QUADRO

sigma = 0.4
F_lin_data = a_lin * E + b_lin
F_quad_data = a_quad * E**2 + b_quad * E + c_quad
chi2_lin = np.sum(((FWHM - F_lin_data) / sigma) ** 2)
chi2_quad = np.sum(((FWHM - F_quad_data) / sigma) ** 2)

# gradi di libertà
dof_lin = N - 2       # 2 parametri: a, b
dof_quad = N - 3      # 3 parametri: a, b, c

chi2_red_lin = chi2_lin / dof_lin
chi2_red_quad = chi2_quad / dof_quad

print("\n CHI QUADRO ")
print(f"Chi^2 lineare = {chi2_lin:.4f}")
print(f"Chi^2 ridotto lineare = {chi2_red_lin:.4f}")
print(f"Chi^2 quadratico = {chi2_quad:.4f}")
print(f"Chi^2 ridotto quadratico = {chi2_red_quad:.4f}")

