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
        smoothed array with the same shape as ``values``.
    """
    if values.size < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")

def gaussian(x, A, mu, sigma):
    return A * np.exp(
        -((x - mu)**2) / (2 * sigma**2)
    )

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
    source_energy = 21.0 # keV
    # Energy grid and input spectrum definition.
    e_x = np.arange(2, 100.1, 0.1)
    
    w_in = np.zeros_like(e_x, dtype=int) # Input spectrum with 0 counts in all bins.
    w_in[e_x < 1] = 30 # Set 30 counts for energies below 1 keV.
    # Sorgente con tre righe monoenergetiche
    w_in[np.isclose(e_x, source_energy)]  = int(1e4)


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
    
    data = np.genfromtxt(
        "fit_results.csv",
        delimiter=",",
        skip_header=1
    )

    a_res = data[ 0] 
    b_res = data[ 1]

    print(f"Calibration fit parameters: a = {a_res:.10f}, b = {b_res:.10f}")
    # Expected FWHM from calibration fit
    fwhm  =np.sqrt(a_res**2  + b_res*source_energy)

    # Convert FWHM -> sigma
    sigma  = fwhm  / 2.355

        
        # Initial guesses for fit parameters
    p0 = [ np.max(w_det), source_energy, sigma]

    # Fit
    params, _ = curve_fit(
        gaussian,
        e_x,
        w_det,
        p0=p0
    )

    # Extract fitted parameters   
    _, mu1, sigma1 = params

    # Convert sigma -> FWHM
    fwhm1 = 2.355 * sigma1

    print("\n FIT RESULTS ")

    print("Peak :")
    print(f"  mu     = {mu1:.3f} keV")
    print(f"  FWHM   = {fwhm1:.3f} keV")
    print(f"FWHM dalle tabelle a {source_energy} keV: 2.063 keV")
    print(f"  Differenza   = {(fwhm1 - 2.063):.3f} keV")


    # Fitted spectrum
    fit_curve = gaussian(e_x, *params)
    
    plt.figure()
    #plt.plot(e_x, w_det / np.max(w_det), label="Detected spectrum")
    plt.plot(e_x, w_det / np.max(w_det) , label="Detected spectrum")
    #plt.plot(e_x, fit_curve / np.max(fit_curve), "--", label="Gaussian fit")
    plt.plot(e_x, w_in / np.max(w_in), label="Input spectrum", linestyle="--")
    #plt.plot(e_x, w_in, label="Input spectrum", linestyle="--")
    np.save("detected_spectrum_w_det.npy", w_det)
    np.save("input_spectrum.npy", w_in)
    np.save("detected_spectrum.npy", w_det)
    np.save("FWHM_value.npy", fwhm1)
    
    # Create one single CSV file with all spectra

    all_data = np.column_stack((
        e_x,
        w_in,
        w_det,
        fit_curve
    ))

    np.savetxt(
        "simulation_results.csv",
        all_data,
        delimiter=",",
        header="Energy_keV,InputSpectrum,DetectedSpectrum,GaussianFit",
        comments=""
    )
    #np.load("detected_spectrum_w_det.npy")
    #np.load("input_spectrum.npy")    
    #np.load("detected_spectrum.npy")    
    #np.load("FWHM_value.npy")
    
    # Try to overlay the experimental differential curve at 21 keV (if available)
    try:
        exp_data = np.genfromtxt(
            "energy_21keV_curve.csv",
            delimiter=",",
            skip_header=1
        )

        exp_E = exp_data[:, 0]
        exp_counts = exp_data[:, 2]
        exp_err = exp_data[:, 3]

        # Normalize experimental counts for plotting on the same normalized axis
        if np.max(exp_counts) > 0:
            exp_counts_norm = exp_counts / np.max(exp_counts)
            exp_err_norm = exp_err / np.max(exp_counts)
        else:
            exp_counts_norm = exp_counts
            exp_err_norm = exp_err

        plt.errorbar(
            exp_E,
            exp_counts_norm,
            yerr=exp_err_norm,
            fmt='o',
            markersize=4,
            capsize=3,
            label="Experimental 21 keV"
        )
    except Exception as exc:  # pragma: no cover - non-fatal plotting error
        print(f"Experimental data not plotted: {exc}")

    plt.xlim([3, 40])
    plt.xlabel("Energy (keV)")
    plt.ylabel("Normalized counts")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
