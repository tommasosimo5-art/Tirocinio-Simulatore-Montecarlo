"""Entry point for running the Python Monte Carlo simulator."""

import time

import numpy as np

from mc_sim import mc_sim
#from mc_sim_test import mc_sim

def _smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply a moving-average smoothing window.

    Parameters
    ----------
    values : numpy.ndarray
        Array of values to smooth.
    window : int, optional
        Smoothing window size.

    Returns
    -------
    numpy.ndarray
        Smoothed array with the same shape as ``values``.
    """
    if values.size < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def main() -> None:
    """Run the example simulation and plot the spectrum."""
    detector = {
        "sensor": "CdTe",
        "sensor_thickness": 1e-1, #650 * 1e-4,
        "sensor_temperature": 100, #240
        "pixel_size": 200 * 1e-4,  #62 * 1e-4, 
        "bias_V": 425, #400
        "threshold0": 8,
        "acquisition_mode": "csm",
    }

    # Energy grid and input spectrum definition.
    e_x = np.arange(5, 70.1, 0.1) # Energy grid from 5 to 70 keV with 0.1 keV steps.
    #e_x = np.arange(2, 100.1, 0.1) 
    w_in = np.zeros_like(e_x, dtype=int) # Input spectrum with 0 counts in all bins.
    w_in[e_x < 1] = 30 # Set 30 counts for energies below 1 keV.
    w_in[np.isclose(e_x, 59.5)] = int(1e4) # Set 10,000 counts at 59.5 keV to simulate a monoenergetic source.

    # Run the Monte Carlo simulation.
    start_time = time.time()
    w_det, _ = mc_sim(detector, e_x, w_in) # Run the simulation with the defined detector, energy grid, and input spectrum.
    elapsed = time.time() - start_time
    print(f"Simulation completed in {elapsed:.2f}s")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping the plot.")
        return

    # Normalize and plot the detected spectrum.
    smoothed = _smooth(w_det)
    plt.figure()
    #plt.plot(e_x, smoothed / np.max(smoothed), label="Detected spectrum")
    plt.plot(e_x, smoothed , label="Detected spectrum")
    plt.plot(e_x, w_in / np.max(w_in), label="Input spectrum", linestyle="--")
    #plt.plot(e_x, w_in, label="Input spectrum", linestyle="--")
    
    plt.xlim([3, 70])
    plt.xlabel("Energy (keV)")
    plt.ylabel("Normalized counts")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

#Usa spekpy per fare i grafici anche dell' argento, degli altri materiali e con i filtri, attento che i dati non sono in conteggi
