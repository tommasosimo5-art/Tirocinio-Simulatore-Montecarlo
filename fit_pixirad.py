import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#??E = np.array([26, 33, 37, 50], dtype=float) #valori pixirad paper vittorio
#??FWHM = np.array([3.4, 3.6, 3.7, 4.1], dtype=float)
#E = np.array([ 13, 17, 21, 25, 29, 33, 37], dtype=float)
#FWHM = np.array([ 2.15, 2.78, 3.3, 3.4, 3.8, 4.1, 4.2], dtype=float)

# LETTURA DATI DAL CSV

data = np.genfromtxt(
    "sqrtAB_fit_residuals.csv",
    delimiter=",",
    skip_header=1
)

# Colonne CSV
E = data[:, 0]              # Energy_keV
FWHM = data[:, 1]           # FWHM_keV
FWHM_err = data[:, 2]       # FWHM_error_keV
FWHM_fit_csv = data[:, 3]   # Fit già presente nel CSV

# FIT LINEARE
# y = a*x + b
# polyfit pesato:
# w = 1/sigma
coeff_lin, cov_lin = np.polyfit(
    E,
    FWHM,
    1,
    w=1/FWHM_err,
    cov=True
)

a_lin = coeff_lin[0]
b_lin = coeff_lin[1]

err_a_lin = np.sqrt(cov_lin[0, 0])
err_b_lin = np.sqrt(cov_lin[1, 1])

print("===== FIT LINEARE PESATO =====")
print(f"a = {a_lin:.6f} ± {err_a_lin:.6f}")
print(f"b = {b_lin:.6f} ± {err_b_lin:.6f}")

# FIT QUADRATICO 
# y = a*x^2 + b*x + c

coeff_quad, cov_quad = np.polyfit(
    E,
    FWHM,
    2,
    w=1/FWHM_err,
    cov=True
)

a_quad = coeff_quad[0]
b_quad = coeff_quad[1]
c_quad = coeff_quad[2]

err_a_quad = np.sqrt(cov_quad[0, 0])
err_b_quad = np.sqrt(cov_quad[1, 1])
err_c_quad = np.sqrt(cov_quad[2, 2])

print("\n===== FIT QUADRATICO PESATO =====")
print(f"a = {a_quad:.8f} ± {err_a_quad:.8f}")
print(f"b = {b_quad:.6f} ± {err_b_quad:.6f}")
print(f"c = {c_quad:.6f} ± {err_c_quad:.6f}")


# FIT SQRT(a^2 + b*E)
def sqrt_model(E, a, b):
    return np.sqrt(a**2 + b * E)

# fit non lineare pesato
popt, pcov = curve_fit(
    sqrt_model,
    E,
    FWHM,
    sigma=FWHM_err,
    absolute_sigma=True,
    p0=[1.0, 0.1]
)

a_sqrt, b_sqrt = popt

err_a_sqrt = np.sqrt(pcov[0, 0])
err_b_sqrt = np.sqrt(pcov[1, 1])

print("\n===== FIT SQRT(a^2 + bE) =====")
print(f"a = {a_sqrt:.10f} ± {err_a_sqrt:.10f}")
print(f"b = {b_sqrt:.10f} ± {err_b_sqrt:.10f}")



# CURVE DI FIT

E_fit = np.linspace(9, 80, 500)

F_fit_lin = a_lin * E_fit + b_lin

F_fit_quad = (
    a_quad * E_fit**2
    + b_quad * E_fit
    + c_quad
)

F_fit_sqrt = sqrt_model(E_fit, a_sqrt, b_sqrt)

# CHI QUADRO

F_lin_data = a_lin * E + b_lin

F_quad_data = (
    a_quad * E**2
    + b_quad * E
    + c_quad
)

F_sqrt_data = sqrt_model(E, a_sqrt, b_sqrt)

chi2_lin = np.sum(
    ((FWHM - F_lin_data) / FWHM_err) ** 2
)

chi2_quad = np.sum(
    ((FWHM - F_quad_data) / FWHM_err) ** 2
)

chi2_sqrt = np.sum(
    ((FWHM - F_sqrt_data) / FWHM_err) ** 2
)

# GRADI DI LIBERTÀ

N = len(E)

dof_lin = N - 2
dof_quad = N - 3
dof_sqrt = N - 2

chi2_red_lin = chi2_lin / dof_lin
chi2_red_quad = chi2_quad / dof_quad
chi2_red_sqrt = chi2_sqrt / dof_sqrt

print("\n===== CHI QUADRO =====")

print("\nLineare:")
print(f"Chi^2 = {chi2_lin:.4f}")
print(f"Chi^2 ridotto = {chi2_red_lin:.4f}")

print("\nQuadratico:")
print(f"Chi^2 = {chi2_quad:.4f}")
print(f"Chi^2 ridotto = {chi2_red_quad:.4f}")

print("\nSQRT(a^2 + bE):")
print(f"Chi^2 = {chi2_sqrt:.4f}")
print(f"Chi^2 ridotto = {chi2_red_sqrt:.4f}")

# VALORI A ENERGIE SPECIFICHE

print("\n===== VALORI FIT SQRT =====")

for energy in [9, 16, 24]:
    value = sqrt_model(energy, a_sqrt, b_sqrt)
    print(f"FWHM a {energy} keV = {value:.3f} keV")

# GRAFICO

plt.figure(figsize=(9,6))

# dati sperimentali
plt.errorbar(
    E,
    FWHM,
    yerr=FWHM_err,
    fmt='o',
    capsize=4,
    label="Dati sperimentali"
)

# fit lineare
plt.plot(
    E_fit,
    F_fit_lin,
    label="Fit lineare pesato"
)

# fit quadratico
plt.plot(
    E_fit,
    F_fit_quad,
    label="Fit quadratico pesato"
)

# fit sqrt
plt.plot(
    E_fit,
    F_fit_sqrt,
    linewidth=2,
    label=r"Fit $\sqrt{a^2 + bE}$"
)

# fit presente nel CSV
plt.plot(
    E,
    FWHM_fit_csv,
    '--',
    label="Fit CSV"
)

plt.xlabel("Energia (keV)")
plt.ylabel("FWHM (keV)")

plt.title("Risoluzione energetica Pixirad")

plt.grid(True)
plt.legend()

plt.show()

all_data = np.column_stack((
    a_sqrt, b_sqrt
    ))

np.savetxt(
        "fit_results.csv",
        all_data,
        delimiter=",",
        header="a_sqrt, b_sqrt",
        comments=""
    )