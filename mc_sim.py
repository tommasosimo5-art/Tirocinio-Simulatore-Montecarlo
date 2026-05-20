"""Monte Carlo simulator for direct-detection photon counting detectors."""

from __future__ import annotations

import math
import sys
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import constants
from scipy.integrate import solve_ivp

try:
    trapz = np.trapz
except AttributeError:
    trapz = np.trapezoid

try:
    import xraylib
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "xraylib Python package is required to run the simulation."
    ) from exc


def absorption_z(mu: float, rng: np.random.Generator) -> float:
    """Sample an interaction depth from exponential attenuation.

    Parameters
    ----------
    mu : float
        Linear attenuation coefficient (1/cm).
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    float
        Interaction depth in cm.
    """
    p = rng.random()
    return -math.log(1 - p) / mu


def _is_nan(value: Any) -> bool:
    """Return True when the input is a NaN float.

    Parameters
    ----------
    value : Any
        Value to inspect.

    Returns
    -------
    bool
        True if ``value`` is a NaN float.
    """
    return isinstance(value, float) and math.isnan(value)


def interacting_element(
    z_1: int,
    z_2: int,
    at_1: str,
    at_2: str,
    sensor: str,
    energy: float,
    rng: np.random.Generator,
) -> Tuple[str, bool, float]:
    """Determine the interacting element and fluorescence output.

    Parameters
    ----------
    z_1, z_2 : int
        Atomic numbers of the sensor elements.
    at_1, at_2 : str
        Element symbols for the sensor compound.
    sensor : str
        Sensor material name used for attenuation lookup.
    energy : float
        Incident photon energy in keV.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    tuple[str, bool, float]
        Element symbol that absorbed the photon, fluorescence flag, and
        fluorescence line energy (keV). The energy is NaN when fluorescence
        does not occur.
    """
    # Choose the interacting element based on relative attenuation.
    rel_w_1 = xraylib.AtomicWeight(z_1) / (
        xraylib.AtomicWeight(z_1) + xraylib.AtomicWeight(z_2)
    )
    mu_rho1 = xraylib.CS_Total_CP(at_1, energy)
    mu_rho_sensor = xraylib.CS_Total_CP(sensor, energy)
    p_1 = 100 * rel_w_1 * mu_rho1 / mu_rho_sensor

    # Decide whether fluorescence is produced and select the emitted line.
    dice_roll = 100 * rng.random()
    if dice_roll <= p_1:
        int_el = at_1
        jump = xraylib.JumpFactor(z_1, 0)
    else:
        int_el = at_2
        jump = xraylib.JumpFactor(z_2, 0)

    dice_roll = 100 * rng.random()
    if dice_roll < 100 * (jump - 1) / jump:
        fluorescence = True
        if int_el == at_1 and energy >= xraylib.EdgeEnergy(z_1, 0): # Check if the incident energy is above the K-edge of element 1 to allow for fluorescence.
            fluo_lin_prob1 = 100 * xraylib.RadRate(z_1, 0)
            dice_roll = 100 * rng.random()
            if dice_roll <= fluo_lin_prob1:
                fluo_en = xraylib.LineEnergy(z_1, 0)
            else:
                fluo_en = xraylib.LineEnergy(z_1, 1)
        elif int_el == at_2 and energy >= xraylib.EdgeEnergy(z_2, 0):
            fluo_lin_prob1 = 100 * xraylib.RadRate(z_2, 0)
            dice_roll = 100 * rng.random()
            if dice_roll <= fluo_lin_prob1:
                fluo_en = xraylib.LineEnergy(z_2, 0)
            else:
                fluo_en = xraylib.LineEnergy(z_2, 1)
        else:
            fluo_en = math.nan
            fluorescence = False
    else:
        fluorescence = False
        fluo_en = math.nan

    return int_el, fluorescence, fluo_en


def fluorescence_detection(
    sensor: str,
    thickness: float,
    fluo_en: float,
    rho: float,
    pixel_size: float,
    z_0: float,
    x_0: float,
    y_0: float,
    rng: np.random.Generator,
) -> Tuple[float, float, float, Optional[bool]]:
    """Propagate a fluorescence photon and determine if it is lost.

    Parameters
    ----------
    sensor : str
        Sensor material name used for attenuation lookup.
    thickness : float
        Sensor thickness in cm.
    fluo_en : float
        Fluorescence photon energy in keV.
    rho : float
        Sensor density in g/cm^3.
    pixel_size : float
        Pixel pitch in cm.
    z_0, x_0, y_0 : float
        Starting absorption position in cm.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    tuple[float, float, float, Optional[bool]]
        Fluorescence absorption position (x, y, z) in cm and a loss flag:
        ``False`` for absorbed in pixel, ``True`` for absorbed outside the
        central pixel region, and NaN if it exits the sensor.
    """
    mu_fluo = xraylib.CS_Total_CP(sensor, fluo_en) * rho
    d_f = absorption_z(mu_fluo, rng)
    # Isotropic emission angles.
    theta = math.pi * rng.random()
    eta = 2 * math.pi * rng.random()
    x_f = x_0 + d_f * math.sin(theta) * math.cos(eta)
    y_f = y_0 + d_f * math.sin(theta) * math.sin(eta)
    z_f = z_0 + d_f * math.cos(theta)

    # Determine if the fluorescence photon escapes the sensor or pixel.
    if z_f < 0 or z_f > thickness:
        lost = math.nan
    elif abs(x_f) > 1 * pixel_size or abs(y_f) > 1 * pixel_size:
        lost = True
    else:
        lost = False

    return x_f, y_f, z_f, lost


def electron_collection(
    energy: float,
    sensor: str,
    pixel_size: float,
    temperature: float,
    z: float,
    thickness: float,
    bias_v: float,
    xx0: float,
    yy0: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate the electron cloud distribution for a deposited energy.

    Parameters
    ----------
    energy : float
        Deposited energy in keV.
    sensor : str
        Sensor material name.
    pixel_size : float
        Pixel pitch in cm.
    temperature : float
        Sensor temperature in K.
    z : float
        Interaction depth in cm.
    thickness : float
        Sensor thickness in cm.
    bias_v : float
        Applied bias voltage in V.
    xx0, yy0 : float
        Interaction position in cm.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    numpy.ndarray
        2D electron cloud distribution over the pixel plane.
    """
    # Convert geometry to meters for diffusion calculations.
    thickness_m = thickness * 1e-2
    z_m = z * 1e-2

    if sensor == "CdTe":
        w_pc_en = 4.3e-3 # eV per electron-hole pair in CdTe.
        fano_factor = 0.1 
        mu_e = 1100e-4 # Electron mobility in m^2/V/s.
        epsilon_r = 10.31 * (1 + temperature * 2.27e-4) # Relative permittivity of CdTe.
    else:
        raise ValueError(
            "Unsupported sensor for charge collection; only CdTe is available."
        )
    if bias_v <= 0:
        raise ValueError("bias_V must be positive.")

    # Charge statistics and diffusion coefficient.
    n_c = energy / w_pc_en # Average number of charge carriers generated.
    n_c_stat = rng.normal(n_c, math.sqrt(fano_factor * n_c)) # Statistical variation in generated charge.

    k_t = constants.Boltzmann * temperature
    q = constants.elementary_charge
    d_coeff = mu_e * k_t / q # Diffusion coefficient from Einstein relation.

    def ode(_t: float, x: np.ndarray) -> np.ndarray:
        x_safe = np.maximum(x, 1e-30) # Avoid division by zero in the ODE when the cloud size is very small.
        return (
            d_coeff
            + (mu_e * n_c_stat * q)
            / (24 * math.pi ** 1.5 * epsilon_r * constants.epsilon_0 * x_safe)
        ) / x_safe

    # Drift time and initial cloud size (empirical fit from MATLAB model).
    t_end = (thickness_m - z_m) * thickness_m / (mu_e * bias_v) # Time for the charge cloud to drift from the interaction depth to the sensor surface.
    t_end = max(t_end, 1e-12) 
    s0 = 1.758e-10 * energy**2 + 2.242e-08 * energy + 2.914e-22 # Initial cloud size in meters, based on an empirical fit to a MATLAB model of charge diffusion in CdTe.
    sol = solve_ivp(ode, (1e-100, t_end), np.array([s0]), rtol=1e-6, atol=1e-9)
    sd = math.sqrt(sol.y[0, -1] ** 2 + s0**2)

    # Build the Gaussian electron distribution on the pixel grid.
    pixel_size_m = pixel_size * 1e-2
    xx0_m = xx0 * 1e-2
    yy0_m = yy0 * 1e-2
    n_grid = 301
    xx = np.linspace(-1.5 * pixel_size_m, 1.5 * pixel_size_m, n_grid)
    yy = np.linspace(-1.5 * pixel_size_m, 1.5 * pixel_size_m, n_grid)
    xx_grid, yy_grid = np.meshgrid(xx, yy)
    electron_distr = (
        n_c_stat
        * np.exp(
            -0.5
            * (
                ((xx_grid - xx0_m) / sd) ** 2
                + ((yy_grid - yy0_m) / sd) ** 2
            )
        )
        / (2 * math.pi * sd * sd)
    )

    return electron_distr


def energy_detection(
    energy: np.ndarray,
    sensor: str,
    electron_distr: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    pixel_size: float,
    csm: bool,
    thr_0: float,
    xx0: float,
    yy0: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Convert charge collection into a detected energy histogram.

    Parameters
    ----------
    energy : numpy.ndarray
        Energy bin centers in keV.
    sensor : str
        Sensor material name.
    electron_distr : numpy.ndarray
        2D electron cloud distribution.
    xx, yy : numpy.ndarray
        Coordinate grid axes in meters.
    pixel_size : float
        Pixel pitch in cm.
    csm : bool
        True to enable charge summing mode.
    thr_0 : float
        Global threshold in keV.
    xx0, yy0 : float
        Interaction position in cm.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    numpy.ndarray
        Detected counts per energy bin (same shape as ``energy``).
    """
    w_det = np.zeros_like(energy, dtype=float)

    if sensor == "CdTe":
        w_pc_en = 4.3e-3 # eV per electron-hole pair in CdTe.
        # FWHM(E) = a*E + b Obtained from fitting 
        #?? a_res = 0.02918032786852756
        #?? b_res = 2.63491803286868744
        a_res = 0.08294642857142853
        b_res = 1.3163392857142868
        def get_el_noise_sigma(energy_keV: float) -> float:
            fwhm_energy = a_res * energy_keV + b_res
            sigma_energy = fwhm_energy / 2.355
            return float(sigma_energy / w_pc_en)
    else:
        raise ValueError("Unsupported sensor for energy detection; only CdTe is available.")

    pixel_size_m = pixel_size * 1e-2
    xx0_m = xx0 * 1e-2
    yy0_m = yy0 * 1e-2
    xx_grid, yy_grid = np.meshgrid(xx, yy)

    # Single-pixel mode integrates charge only in the central pixel.
    if not csm:
        collection_area = np.zeros_like(yy_grid)
        collection_area[
            (np.abs(xx_grid) < pixel_size_m / 2)
            & (np.abs(yy_grid) < pixel_size_m / 2)
        ] = 1
        int_charge = trapz(
            trapz(electron_distr * collection_area, xx, axis=1), yy, axis=0
        )
        el_noise_sigma = get_el_noise_sigma(float(int_charge * w_pc_en))
        det_charge = rng.normal(int_charge, el_noise_sigma) # Add electronic noise to the collected charge.
        det_en = det_charge * w_pc_en # Convert collected charge to energy using the pair creation energy.
        if det_en < thr_0:
            det_en = math.nan
    else:
        # Charge summing mode evaluates the highest collected charge among 4 pixels.
        if (abs(xx0_m) <= 1.5 * pixel_size_m) and (abs(yy0_m) <= 1.5 * pixel_size_m):
            collection_area1 = np.zeros_like(yy_grid)
            collection_area1[(xx_grid < pixel_size_m / 2) & (yy_grid < pixel_size_m / 2)] = 1

            collection_area2 = np.zeros_like(yy_grid)
            collection_area2[(xx_grid > -pixel_size_m / 2) & (yy_grid < pixel_size_m / 2)] = 1

            collection_area3 = np.zeros_like(yy_grid)
            collection_area3[(xx_grid < pixel_size_m / 2) & (yy_grid > -pixel_size_m / 2)] = 1

            collection_area4 = np.zeros_like(yy_grid)
            collection_area4[(xx_grid > -pixel_size_m / 2) & (yy_grid > -pixel_size_m / 2)] = 1

            int_charge1 = trapz(
                trapz(electron_distr * collection_area1, xx, axis=1), yy, axis=0
            )
            int_charge2 = trapz(
                trapz(electron_distr * collection_area2, xx, axis=1), yy, axis=0
            )
            int_charge3 = trapz(
                trapz(electron_distr * collection_area3, xx, axis=1), yy, axis=0
            )
            int_charge4 = trapz(
                trapz(electron_distr * collection_area4, xx, axis=1), yy, axis=0
            )

            int_charge = max([int_charge1, int_charge2, int_charge3, int_charge4])
            el_noise_sigma = get_el_noise_sigma(float(int_charge * w_pc_en))
            det_charge = rng.normal(
                int_charge, math.sqrt(int_charge + el_noise_sigma**2)
            )
            det_en = det_charge * w_pc_en
        else:
            det_en = math.nan

    # Bin the detected energy into the provided histogram grid.
    bin_width = float(np.mean(np.diff(energy))) if energy.size > 1 else 0.0 
    if not math.isnan(det_en) and bin_width > 0:
        w_det[np.abs(energy - det_en) < bin_width] += 1

    return w_det


def mc_sim(
    detector: Dict[str, Any],
    e_x: np.ndarray,
    w_in: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the Monte Carlo detector simulation.

    Parameters
    ----------
    detector : dict
        Detector configuration with keys:
        ``sensor``, ``sensor_thickness``, ``sensor_temperature``,
        ``pixel_size``, ``bias_V``, ``threshold0``, ``acquisition_mode``.
    e_x : numpy.ndarray
        Energy grid in keV.
    w_in : numpy.ndarray
        Input spectrum counts per energy bin.
    rng : numpy.random.Generator, optional
        Random number generator (uses a default generator if None).
    verbose : bool, optional
        Print progress while running the simulation.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Detected spectrum and lost-event spectrum (counts per energy bin).
    """
    sensor = str(detector["sensor"])
    thickness = float(detector["sensor_thickness"])
    pixel_size = float(detector["pixel_size"])
    bias_v = float(detector["bias_V"])
    temperature = float(detector["sensor_temperature"])
    thr_0 = float(detector["threshold0"])
    csm = detector.get("acquisition_mode") == "csm"

    if rng is None:
        rng = np.random.default_rng()

    w_det = np.zeros_like(e_x, dtype=float)
    w_lost = np.zeros_like(e_x, dtype=float)

    # Material constants for the compound sensor.
    if sensor == "CdTe":
        at_1, z_1 = "Cd", 48
        at_2, z_2 = "Te", 52
        rho = 5.85
    elif sensor == "GaAs":
        at_1, z_1 = "Ga", 31
        at_2, z_2 = "As", 33
        rho = 5.32
    else:
        raise ValueError(
            "Misspelled or unsupported sensor material: use CdTe or GaAs."
        )

    # Pre-compute linear attenuation coefficients for each energy bin.
    mu = np.zeros(e_x.size, dtype=float)
    for ii, energy in enumerate(e_x):
        mu[ii] = xraylib.CS_Total_CP(sensor, float(energy)) * rho # Convert mass attenuation coefficient to linear attenuation coefficient using density.

    n_grid = 301
    xx = np.linspace(-1.5 * pixel_size, 1.5 * pixel_size, n_grid) * 1e-2
    yy = np.linspace(-1.5 * pixel_size, 1.5 * pixel_size, n_grid) * 1e-2

    start_time = time.time()
    total_events = float(np.sum(w_in))

    for jj, energy in enumerate(e_x):
        mu_j = mu[jj]
        for _ in range(int(w_in[jj])):
            # Sample a random interaction point within the pixel.
            x_0 = -0.5 * pixel_size + 0.5 * pixel_size * rng.random() # Random x position within the central pixel.
            y_0 = -0.5 * pixel_size + 0.5 * pixel_size * rng.random() # Random y position within the central pixel.
            x_f = x_0
            y_f = y_0
            electron_distr = None

            # Draw interaction depth from exponential attenuation.
            z_0 = absorption_z(mu_j, rng)
            z_f = z_0
            if z_0 > thickness:
                w_lost += 1
                continue

            k_edge = xraylib.EdgeEnergy(z_1, 0)
            if energy < k_edge:
                # Sub-K-edge: full energy is deposited locally.
                electron_distr_tmp = electron_collection(
                    energy,
                    sensor,
                    pixel_size,
                    temperature,
                    z_0,
                    thickness,
                    bias_v,
                    x_0,
                    y_0,
                    rng,
                )
                electron_distr = electron_distr_tmp
                w_det_t = energy_detection(
                    e_x,
                    sensor,
                    electron_distr,
                    xx,
                    yy,
                    pixel_size,
                    csm,
                    thr_0,
                    x_0,
                    y_0,
                    rng,
                )
                w_det += w_det_t
                continue

            # Fluorescence and secondary interactions when above K-edge.
            int_el, fluorescence, fluo_en = interacting_element(
                z_1, z_2, at_1, at_2, sensor, float(energy), rng
            )

            if not fluorescence:
                # No fluorescence: deposit full energy in the initial interaction.
                electron_distr_tmp = electron_collection(
                    energy,
                    sensor,
                    pixel_size,
                    temperature,
                    z_f,
                    thickness,
                    bias_v,
                    x_0,
                    y_0,
                    rng,
                )
                electron_distr = electron_distr_tmp
                w_det_t = energy_detection(
                    e_x,
                    sensor,
                    electron_distr_tmp,
                    xx,
                    yy,
                    pixel_size,
                    csm,
                    thr_0,
                    x_0,
                    y_0,
                    rng,
                )
                w_det += w_det_t
                continue

            if int_el == at_1:
                # Element 1 fluorescence: deposit remaining energy locally.
                dep_en = energy - fluo_en
                electron_distr_tmp = electron_collection(
                    dep_en,
                    sensor,
                    pixel_size,
                    temperature,
                    z_0,
                    thickness,
                    bias_v,
                    x_0,
                    y_0,
                    rng,
                )
                electron_distr = electron_distr_tmp
                x_f, y_f, z_f, lost = fluorescence_detection(
                    sensor,
                    thickness,
                    fluo_en,
                    rho,
                    pixel_size,
                    z_0,
                    x_0,
                    y_0,
                    rng,
                )

                if lost is True or _is_nan(lost):
                    w_det_t = energy_detection(
                        e_x,
                        sensor,
                        electron_distr,
                        xx,
                        yy,
                        pixel_size,
                        csm,
                        thr_0,
                        x_0,
                        y_0,
                        rng,
                    )
                    w_det += w_det_t

                    if csm and lost is True:
                        electron_distr = None
                        electron_distr_tmp = electron_collection(
                            fluo_en,
                            sensor,
                            pixel_size,
                            temperature,
                            z_f,
                            thickness,
                            bias_v,
                            0,
                            0,
                            rng,
                        )
                        electron_distr = electron_distr_tmp
                        w_det_t = energy_detection(
                            e_x,
                            sensor,
                            electron_distr,
                            xx,
                            yy,
                            pixel_size,
                            csm,
                            thr_0,
                            0,
                            0,
                            rng,
                        )
                        w_det += w_det_t
                        continue
                else:
                    electron_distr_tmp = electron_collection(
                        fluo_en,
                        sensor,
                        pixel_size,
                        temperature,
                        z_f,
                        thickness,
                        bias_v,
                        x_f,
                        y_f,
                        rng,
                    )
                    electron_distr = electron_distr + electron_distr_tmp
                    x_b = (fluo_en * x_f + dep_en * x_0) / (fluo_en + dep_en)
                    y_b = (fluo_en * y_f + dep_en * y_0) / (fluo_en + dep_en)
                    w_det_t = energy_detection(
                        e_x,
                        sensor,
                        electron_distr,
                        xx,
                        yy,
                        pixel_size,
                        csm,
                        thr_0,
                        x_b,
                        y_b,
                        rng,
                    )
                    w_det += w_det_t
                continue

            # Element 2 fluorescence: handle possible secondary fluorescence.
            dep_en = energy - fluo_en
            x_f, y_f, z_f, lost = fluorescence_detection(
                sensor,
                thickness,
                fluo_en,
                rho,
                pixel_size,
                z_0,
                x_0,
                y_0,
                rng,
            )

            if _is_nan(lost):
                electron_distr_tmp = electron_collection(
                    dep_en,
                    sensor,
                    pixel_size,
                    temperature,
                    z_f,
                    thickness,
                    bias_v,
                    x_0,
                    y_0,
                    rng,
                )
                electron_distr = electron_distr_tmp
                w_det_t = energy_detection(
                    e_x,
                    sensor,
                    electron_distr,
                    xx,
                    yy,
                    pixel_size,
                    csm,
                    thr_0,
                    x_0,
                    y_0,
                    rng,
                )
                w_det += w_det_t
                continue

            int_el2, fluorescence2, fluo_en2 = interacting_element(
                z_1, z_2, at_1, at_2, sensor, float(fluo_en), rng
            )
            x_f2, y_f2, z_f2, lost2 = fluorescence_detection(
                sensor,
                thickness,
                fluo_en,
                rho,
                pixel_size,
                z_f,
                x_f,
                y_f,
                rng,
            )
            dep_en1 = fluo_en - fluo_en2

            electron_distr_tmp = electron_collection(
                dep_en,
                sensor,
                pixel_size,
                temperature,
                z_0,
                thickness,
                bias_v,
                x_0,
                y_0,
                rng,
            )
            electron_distr = electron_distr_tmp

            if not fluorescence2 and lost is False:
                electron_distr_tmp = electron_collection(
                    fluo_en,
                    sensor,
                    pixel_size,
                    temperature,
                    z_f,
                    thickness,
                    bias_v,
                    x_f,
                    y_f,
                    rng,
                )
                electron_distr = electron_distr + electron_distr_tmp
                x_b = (fluo_en * x_f + dep_en * x_0) / (fluo_en + dep_en)
                y_b = (fluo_en * y_f + dep_en * y_0) / (fluo_en + dep_en)
                w_det_t = energy_detection(
                    e_x,
                    sensor,
                    electron_distr,
                    xx,
                    yy,
                    pixel_size,
                    csm,
                    thr_0,
                    x_b,
                    y_b,
                    rng,
                )
                w_det += w_det_t
                continue

            if not fluorescence2 and lost is True:
                w_det_t = energy_detection(
                    e_x,
                    sensor,
                    electron_distr,
                    xx,
                    yy,
                    pixel_size,
                    csm,
                    thr_0,
                    x_0,
                    y_0,
                    rng,
                )
                w_det += w_det_t

                if csm:
                    electron_distr = None
                    electron_distr_tmp = electron_collection(
                        fluo_en,
                        sensor,
                        pixel_size,
                        temperature,
                        z_f,
                        thickness,
                        bias_v,
                        0,
                        0,
                        rng,
                    )
                    electron_distr = electron_distr_tmp
                    w_det_t = energy_detection(
                        e_x,
                        sensor,
                        electron_distr,
                        xx,
                        yy,
                        pixel_size,
                        csm,
                        thr_0,
                        0,
                        0,
                        rng,
                    )
                    w_det += w_det_t
                continue

            if (lost is True and lost2 is True) or _is_nan(lost2):
                if _is_nan(lost2) and lost is False:
                    electron_distr_tmp = electron_collection(
                        dep_en1,
                        sensor,
                        pixel_size,
                        temperature,
                        z_f,
                        thickness,
                        bias_v,
                        x_f,
                        y_f,
                        rng,
                    )
                    electron_distr = electron_distr + electron_distr_tmp
                    x_b = (dep_en1 * x_f + dep_en * x_0) / (dep_en1 + dep_en)
                    y_b = (dep_en1 * y_f + dep_en * y_0) / (dep_en1 + dep_en)
                    w_det_t = energy_detection(
                        e_x,
                        sensor,
                        electron_distr,
                        xx,
                        yy,
                        pixel_size,
                        csm,
                        thr_0,
                        x_b,
                        y_b,
                        rng,
                    )
                    w_det += w_det_t
                    continue

            if _is_nan(lost2) and lost is True:
                w_det_t = energy_detection(
                    e_x,
                    sensor,
                    electron_distr,
                    xx,
                    yy,
                    pixel_size,
                    csm,
                    thr_0,
                    x_0,
                    y_0,
                    rng,
                )
                w_det += w_det_t

                if csm:
                    electron_distr = None
                    electron_distr_tmp = electron_collection(
                        dep_en1,
                        sensor,
                        pixel_size,
                        temperature,
                        z_f,
                        thickness,
                        bias_v,
                        0,
                        0,
                        rng,
                    )
                    electron_distr = electron_distr_tmp
                    w_det_t = energy_detection(
                        e_x,
                        sensor,
                        electron_distr,
                        xx,
                        yy,
                        pixel_size,
                        csm,
                        thr_0,
                        0,
                        0,
                        rng,
                    )
                    w_det += w_det_t
                continue

            if lost2 is False and lost is False:
                electron_distr_tmp = electron_collection(
                    dep_en1,
                    sensor,
                    pixel_size,
                    temperature,
                    z_f,
                    thickness,
                    bias_v,
                    x_f,
                    y_f,
                    rng,
                )
                electron_distr = electron_distr + electron_distr_tmp
                electron_distr_tmp = electron_collection(
                    fluo_en2,
                    sensor,
                    pixel_size,
                    temperature,
                    z_f2,
                    thickness,
                    bias_v,
                    x_f2,
                    y_f2,
                    rng,
                )
                electron_distr = electron_distr + electron_distr_tmp
                x_b = (dep_en * x_0 + dep_en1 * x_f + fluo_en2 * x_f2) / (
                    dep_en + dep_en1 + fluo_en2
                )
                y_b = (dep_en * y_0 + dep_en1 * y_f + fluo_en2 * y_f2) / (
                    dep_en + dep_en1 + fluo_en2
                )
                w_det_t = energy_detection(
                    e_x,
                    sensor,
                    electron_distr,
                    xx,
                    yy,
                    pixel_size,
                    csm,
                    thr_0,
                    x_b,
                    y_b,
                    rng,
                )
                w_det += w_det_t
                continue

            if lost2 is True and lost is False:
                electron_distr_tmp = electron_collection(
                    dep_en1,
                    sensor,
                    pixel_size,
                    temperature,
                    z_f,
                    thickness,
                    bias_v,
                    x_f,
                    y_f,
                    rng,
                )
                electron_distr = electron_distr + electron_distr_tmp
                w_det_t = energy_detection(
                    e_x,
                    sensor,
                    electron_distr,
                    xx,
                    yy,
                    pixel_size,
                    csm,
                    thr_0,
                    x_f,
                    y_f,
                    rng,
                )
                w_det += w_det_t

                if csm:
                    electron_distr = None
                    electron_distr_tmp = electron_collection(
                        fluo_en2,
                        sensor,
                        pixel_size,
                        temperature,
                        z_f2,
                        thickness,
                        bias_v,
                        0,
                        0,
                        rng,
                    )
                    electron_distr = electron_distr_tmp
                    w_det_t = energy_detection(
                        e_x,
                        sensor,
                        electron_distr,
                        xx,
                        yy,
                        pixel_size,
                        csm,
                        thr_0,
                        0,
                        0,
                        rng,
                    )
                    w_det += w_det_t
                continue

            if lost2 is False and lost is True:
                electron_distr_tmp = electron_collection(
                    fluo_en2,
                    sensor,
                    pixel_size,
                    temperature,
                    z_f2,
                    thickness,
                    bias_v,
                    x_f2,
                    y_f2,
                    rng,
                )
                electron_distr = electron_distr + electron_distr_tmp
                w_det_t = energy_detection(
                    e_x,
                    sensor,
                    electron_distr,
                    xx,
                    yy,
                    pixel_size,
                    csm,
                    thr_0,
                    x_f2,
                    y_f2,
                    rng,
                )
                w_det += w_det_t

                if csm:
                    electron_distr = None
                    electron_distr_tmp = electron_collection(
                        dep_en1,
                        sensor,
                        pixel_size,
                        temperature,
                        z_f2,
                        thickness,
                        bias_v,
                        0,
                        0,
                        rng,
                    )
                    electron_distr = electron_distr_tmp
                    w_det_t = energy_detection(
                        e_x,
                        sensor,
                        electron_distr,
                        xx,
                        yy,
                        pixel_size,
                        csm,
                        thr_0,
                        0,
                        0,
                        rng,
                    )
                    w_det += w_det_t
                continue

            if lost2 is True and lost is True:
                w_det_t = energy_detection(
                    e_x,
                    sensor,
                    electron_distr,
                    xx,
                    yy,
                    pixel_size,
                    csm,
                    thr_0,
                    x_0,
                    y_0,
                    rng,
                )
                w_det += w_det_t

                if csm:
                    electron_distr = None
                    electron_distr_tmp = electron_collection(
                        fluo_en,
                        sensor,
                        pixel_size,
                        temperature,
                        z_f2,
                        thickness,
                        bias_v,
                        0,
                        0,
                        rng,
                    )
                    electron_distr = electron_distr_tmp
                    w_det_t = energy_detection(
                        e_x,
                        sensor,
                        electron_distr,
                        xx,
                        yy,
                        pixel_size,
                        csm,
                        thr_0,
                        0,
                        0,
                        rng,
                    )
                    w_det += w_det_t
                continue

        if verbose and total_events > 0:
            progress = float(np.sum(w_in[: jj + 1])) / total_events * 100
            elapsed_time = time.time() - start_time
            remaining_time = (elapsed_time / max(np.sum(w_in[: jj + 1]), 1)) * (
                total_events - np.sum(w_in[: jj + 1])
            )
            bar = "#" * round(progress / 5)
            sys.stdout.write(
                f"\rProgress: [{bar:<20}] {progress:.1f}%"
                f" Time: Remaining={remaining_time:.2f} s | Elapsed={elapsed_time:.2f} s"
            )
            sys.stdout.flush()

    if verbose and total_events > 0:
        sys.stdout.write("\n")
        sys.stdout.flush()

    return w_det, w_lost
