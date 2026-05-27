import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.genfromtxt(
    "energy_21keV_curve.csv",
    delimiter=",",
    skip_header=1
)

# Extract columns
Energy_keV = data[:, 0]
RawThresholdEnergy_keV = data[:, 1]
Differential_counts = data[:, 2]
Differential_error = data[:, 3]
Differential_SEM = data[:, 4]
DifferentialNorm_ROI1 = data[:, 5]
DifferentialNorm_ROI2 = data[:, 6]
DifferentialNorm_ROI3 = data[:, 7]
DifferentialNorm_ROI4 = data[:, 8]
DifferentialNorm_ROI5 = data[:, 9]

# Create figure with subplots
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# Plot 1: Differential counts with error bars
ax1 = axes[0]
ax1.errorbar(
    Energy_keV,
    Differential_counts,
    yerr=Differential_error,
    fmt='o-',
    capsize=5,
    label='Differential counts',
    linewidth=1.5,
    markersize=5
)
ax1.set_xlabel('Energy (keV)')
ax1.set_ylabel('Differential counts')
ax1.set_title('Differential Counts vs Energy (21 keV)')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Normalized data from all ROIs
ax2 = axes[1]
ax2.plot(Energy_keV, DifferentialNorm_ROI1, 'o-', label='ROI 1', markersize=4)
ax2.plot(Energy_keV, DifferentialNorm_ROI2, 's-', label='ROI 2', markersize=4)
ax2.plot(Energy_keV, DifferentialNorm_ROI3, '^-', label='ROI 3', markersize=4)
ax2.plot(Energy_keV, DifferentialNorm_ROI4, 'd-', label='ROI 4', markersize=4)
ax2.plot(Energy_keV, DifferentialNorm_ROI5, 'v-', label='ROI 5', markersize=4)
ax2.set_xlabel('Energy (keV)')
ax2.set_ylabel('Normalized differential counts')
ax2.set_title('Normalized Differential Counts for All ROIs')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()