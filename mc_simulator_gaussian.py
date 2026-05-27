#!!!! Inserisci nel fit gaussiano i valori di sigma aspettati come parametro di partenza






#i commenti con "??" indicano valori Usati nel programma originale di Vittorio
"""Entry point for running the Python Monte Carlo simulator."""

import time

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
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

def gaussian(x, A, mu, sigma):
    return A * np.exp(
        -((x - mu)**2) / (2 * sigma**2)
    )

def triple_gaussian(
    x,
    A1, mu1, sigma1,
    A2, mu2, sigma2,
    A3, mu3, sigma3
):
    g1 = gaussian(x, A1, mu1, sigma1)
    g2 = gaussian(x, A2, mu2, sigma2)
    g3 = gaussian(x, A3, mu3, sigma3)

    return g1 + g2 + g3
def main() -> None:
    """Run the example simulation and plot the spectrum."""
    detector = {
        "sensor": "CdTe",
        "sensor_thickness":  650 * 1e-4,
        "sensor_temperature":  240,
        "pixel_size":  62 * 1e-4, 
        "bias_V":  400,
        "threshold0":  4,
        "acquisition_mode": "csm",
    }

    # Energy grid and input spectrum definition.
    e_x = np.arange(5, 70.1, 0.1) # Energy grid from 5 to 70 keV with 0.1 keV steps.
    #??e_x = np.arange(2, 100.1, 0.1)
    
    w_in = np.zeros_like(e_x, dtype=int) # Input spectrum with 0 counts in all bins.
    w_in[e_x < 1] = 30 # Set 30 counts for energies below 1 keV.
    #w_in[np.isclose(e_x, 59.5)] = int(1e4) # Set 10,000 counts at 59.5 keV to simulate a monoenergetic source.
    #??w_in[np.isclose(e_x, 37.0)] = int(1e4)
    # Sorgente con tre righe monoenergetiche
    w_in[np.isclose(e_x, 9.0)]  = int(1e2)
    w_in[np.isclose(e_x, 16.0)] = int(1e2)
    w_in[np.isclose(e_x, 24.0)] = int(1e2)

    # Run the Monte Carlo simulation.
    start_time = time.time()
    w_det, _ = mc_sim(detector, e_x, w_in) # Run the simulation with the defined detector, energy grid, and input spectrum.
    elapsed = time.time() - start_time
    print(f"Simulation completed in {elapsed:.2f}s")


    # Count total events
    total_input = np.sum(w_in)
    total_detected = np.sum(w_det)

    print(f"Total input events     : {total_input}")
    print(f"Total detected events  : {total_detected}")
    print(f"Difference             : {total_detected - total_input}")

    efficiency = 100 * total_detected / total_input
    print(f"Detection efficiency   : {efficiency:.2f}%")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping the plot.")
        return

    # Normalize and plot the detected spectrum.
    smoothed = _smooth(w_det)
    
    
    a_res = 0.08294642857142853
    b_res = 1.3163392857142868
    # Expected FWHM from calibration fit
    fwhm_9  = a_res * 9  + b_res
    fwhm_16 = a_res * 16 + b_res
    fwhm_24 = a_res * 24 + b_res

    # Convert FWHM -> sigma
    sigma_9  = fwhm_9  / 2.355
    sigma_16 = fwhm_16 / 2.355
    sigma_24 = fwhm_24 / 2.355
        
        # Initial guesses for fit parameters
    p0 = [
        np.max(smoothed), 9, sigma_9,
        np.max(smoothed), 16, sigma_16,
        np.max(smoothed), 24, sigma_24
    ]

    # Fit
    params, covariance = curve_fit(
        triple_gaussian,
        e_x,
        smoothed,
        p0=p0
    )

    # Extract fitted parameters
    A1, mu1, sigma1, \
    A2, mu2, sigma2, \
    A3, mu3, sigma3 = params

    # Convert sigma -> FWHM
    fwhm1 = 2.355 * sigma1
    fwhm2 = 2.355 * sigma2
    fwhm3 = 2.355 * sigma3

    print("\n FIT RESULTS ")

    print("Peak 9:")
    print(f"  mu     = {mu1:.3f} keV")
    print(f"  FWHM   = {fwhm1:.3f} keV")
    print("FWHM dalle tabelle a 9 keV: 2.063 keV")
    print(f"  Differenza   = {(fwhm1 - 2.063):.3f} keV")
    print("Peak 16:")
    print(f"  mu     = {mu2:.3f} keV")
    print(f"  FWHM   = {fwhm2:.3f} keV")
    print("FWHM dalle tabelle a 16 keV: 2.643 keV")
    print(f"  Differenza   = {(fwhm2 - 2.643):.3f} keV")
    print("Peak 24:")
    print(f"  mu     = {mu3:.3f} keV")
    print(f"  FWHM   = {fwhm3:.3f} keV")
    print("FWHM dalle tabelle a 24 keV: 3.307 keV")
    print(f"  Differenza   = {(fwhm3 - 3.307):.3f} keV")

    # Fitted spectrum
    fit_curve = triple_gaussian(e_x, *params)
    
    plt.figure()
    #plt.plot(e_x, smoothed / np.max(smoothed), label="Detected spectrum")
    plt.plot(e_x, smoothed , label="Detected spectrum")
    plt.plot(e_x, fit_curve, "--", label="Triple Gaussian fit")
    plt.plot(e_x, w_in / np.max(w_in), label="Input spectrum", linestyle="--")
    #plt.plot(e_x, w_in, label="Input spectrum", linestyle="--")
    np.save("detected_spectrum_smoothed.npy", smoothed)
    np.save("input_spectrum.npy", w_in)
    np.save("detected_spectrum.npy", w_det)
    np.save("FWHM_values.npy", [fwhm1, fwhm2, fwhm3])
    
    np.savetxt("dati.csv",np.column_stack((e_x, w_det)), delimiter=", ", header="Energy,w_det", comments="")
    np.savetxt("dati1.csv",np.column_stack((e_x, w_in)), delimiter=", ", header="Energy,w_in", comments="")
    np.savetxt("dati2.csv",np.column_stack((e_x, smoothed)), delimiter=", ", header="Energy,w_smoothed", comments="")
    #np.savetxt("dati3.csv",np.column_stack((e_x, fwhm1)), delimiter=", ", header="Energy,fwhm1", comments="")
    #np.savetxt("dati4.csv",np.column_stack((e_x, fwhm2)), delimiter=", ", header="Energy,fwhm2", comments="")
    #np.savetxt("dati5.csv",np.column_stack((e_x, fwhm3)), delimiter=", ", header="Energy,fwhm3", comments="")

    #np.load("detected_spectrum_smoothed.npy")
    #np.load("input_spectrum.npy")    
    #np.load("detected_spectrum.npy")    
    #np.load("FWHM_values.npy")
    
    plt.xlim([3, 40])
    plt.xlabel("Energy (keV)")
    plt.ylabel("Normalized counts")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
