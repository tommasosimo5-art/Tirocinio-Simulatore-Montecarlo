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


def _choose_absorbing_element(
    z_1: int,
    z_2: int,
    at_1: str,
    at_2: str,
    sensor: str,
    energy: float,
    rng: np.random.Generator,
) -> Tuple[str, float]:
    rel_w_1 = xraylib.AtomicWeight(z_1) / (
        xraylib.AtomicWeight(z_1) + xraylib.AtomicWeight(z_2)
    )
    mu_rho1 = xraylib.CS_Total_CP(at_1, energy)
    mu_rho_sensor = xraylib.CS_Total_CP(sensor, energy)
    p_1 = rel_w_1 * mu_rho1 / mu_rho_sensor

    if rng.random() <= p_1:
        return at_1, xraylib.JumpFactor(z_1, 0)
    return at_2, xraylib.JumpFactor(z_2, 0)


def _fluorescence_line_energy(
    int_el: str,
    z_el: int,
    energy: float,
    rng: np.random.Generator,
) -> float:
    if energy < xraylib.EdgeEnergy(z_el, 0):
        return math.nan

    if rng.random() <= xraylib.RadRate(z_el, 0):
        return xraylib.LineEnergy(z_el, 0)
    return xraylib.LineEnergy(z_el, 1)


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
    int_el, jump = _choose_absorbing_element(
        z_1, z_2, at_1, at_2, sensor, energy, rng
    )

    if rng.random() < (jump - 1) / jump:
        z_el = z_1 if int_el == at_1 else z_2
        fluo_en = _fluorescence_line_energy(int_el, z_el, energy, rng)
        fluorescence = not math.isnan(fluo_en)
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
        el_noise_sigma = 283 # Standard deviation of electronic noise in electrons, based on an empirical fit to a MATLAB model of charge collection in CdTe. This value is chosen to produce a realistic energy resolution in the simulated spectrum.
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


def get_material_properties(sensor: str) -> Tuple[str, int, str, int, float]:
    """Get material properties for the sensor."""
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
    return at_1, z_1, at_2, z_2, rho


def print_simulation_progress(
    jj: int, e_x: np.ndarray, w_in: np.ndarray, start_time: float, total_events: float
) -> None:
    """Print simulation progress."""
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


def _collect_electron_distribution(
    energy: float,
    sensor: str,
    pixel_size: float,
    temperature: float,
    z: float,
    thickness: float,
    bias_v: float,
    x: float,
    y: float,
    rng: np.random.Generator,
) -> np.ndarray:
    return electron_collection(
        energy,
        sensor,
        pixel_size,
        temperature,
        z,
        thickness,
        bias_v,
        x,
        y,
        rng,
    )


def _detect_electron_distribution(
    e_x: np.ndarray,
    sensor: str,
    electron_distr: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    pixel_size: float,
    csm: bool,
    thr_0: float,
    x: float,
    y: float,
    rng: np.random.Generator,
) -> float:
    return energy_detection(
        e_x,
        sensor,
        electron_distr,
        xx,
        yy,
        pixel_size,
        csm,
        thr_0,
        x,
        y,
        rng,
    )


def _detect_primary_and_optional_csm(
    e_x: np.ndarray,
    sensor: str,
    electron_distr: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    pixel_size: float,
    csm: bool,
    thr_0: float,
    x: float,
    y: float,
    rng: np.random.Generator,
    csm_energy: Optional[float] = None,
    csm_z: Optional[float] = None,
    thickness: Optional[float] = None,
    temperature: Optional[float] = None,
    bias_v: Optional[float] = None,
) -> float:
    detected = _detect_electron_distribution(
        e_x,
        sensor,
        electron_distr,
        xx,
        yy,
        pixel_size,
        csm,
        thr_0,
        x,
        y,
        rng,
    )
    if csm and csm_energy is not None:
        csm_distr = _collect_electron_distribution(
            csm_energy,
            sensor,
            pixel_size,
            temperature,
            csm_z,
            thickness,
            bias_v,
            0,
            0,
            rng,
        )
        detected += _detect_electron_distribution(
            e_x,
            sensor,
            csm_distr,
            xx,
            yy,
            pixel_size,
            csm,
            thr_0,
            0,
            0,
            rng,
        )
    return detected


def _deposit_and_detect(
    energy: float,
    sensor: str,
    pixel_size: float,
    temperature: float,
    z: float,
    thickness: float,
    bias_v: float,
    x: float,
    y: float,
    rng: np.random.Generator,
    e_x: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    csm: bool,
    thr_0: float,
) -> float:
    electron_distr = _collect_electron_distribution(
        energy,
        sensor,
        pixel_size,
        temperature,
        z,
        thickness,
        bias_v,
        x,
        y,
        rng,
    )
    return _detect_electron_distribution(
        e_x,
        sensor,
        electron_distr,
        xx,
        yy,
        pixel_size,
        csm,
        thr_0,
        x,
        y,
        rng,
    )


def _handle_element1_fluorescence(
    energy: float,
    fluo_en: float,
    sensor: str,
    pixel_size: float,
    temperature: float,
    thickness: float,
    bias_v: float,
    thr_0: float,
    csm: bool,
    rho: float,
    x_0: float,
    y_0: float,
    z_0: float,
    xx: np.ndarray,
    yy: np.ndarray,
    e_x: np.ndarray,
    rng: np.random.Generator,
) -> float:
    dep_en = energy - fluo_en
    electron_distr = _collect_electron_distribution(
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
        csm_energy = fluo_en if (csm and lost is True) else None
        return _detect_primary_and_optional_csm(
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
            csm_energy=csm_energy,
            csm_z=z_f,
            thickness=thickness,
            temperature=temperature,
            bias_v=bias_v,
        )

    electron_distr += _collect_electron_distribution(
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
    x_b = (fluo_en * x_f + dep_en * x_0) / energy
    y_b = (fluo_en * y_f + dep_en * y_0) / energy
    return _detect_electron_distribution(
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


def _handle_element2_fluorescence(
    energy: float,
    fluo_en: float,
    sensor: str,
    pixel_size: float,
    temperature: float,
    thickness: float,
    bias_v: float,
    thr_0: float,
    csm: bool,
    rho: float,
    at_1: str,
    z_1: int,
    at_2: str,
    z_2: int,
    x_0: float,
    y_0: float,
    z_0: float,
    xx: np.ndarray,
    yy: np.ndarray,
    e_x: np.ndarray,
    rng: np.random.Generator,
) -> float:
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
        return _deposit_and_detect(
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
            e_x,
            xx,
            yy,
            csm,
            thr_0,
        )

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
    electron_distr = _collect_electron_distribution(
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

    if not fluorescence2:
        if lost is False:
            electron_distr += _collect_electron_distribution(
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
            x_b = (fluo_en * x_f + dep_en * x_0) / energy
            y_b = (fluo_en * y_f + dep_en * y_0) / energy
            return _detect_electron_distribution(
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

        csm_energy = fluo_en if csm else None
        return _detect_primary_and_optional_csm(
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
            csm_energy=csm_energy,
            csm_z=z_f,
            thickness=thickness,
            temperature=temperature,
            bias_v=bias_v,
        )

    if (lost is True and lost2 is True) or _is_nan(lost2):
        if _is_nan(lost2) and lost is False:
            electron_distr += _collect_electron_distribution(
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
            x_b = (dep_en1 * x_f + dep_en * x_0) / (dep_en1 + dep_en)
            y_b = (dep_en1 * y_f + dep_en * y_0) / (dep_en1 + dep_en)
            return _detect_electron_distribution(
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

        if _is_nan(lost2) and lost is True:
            csm_energy = dep_en1 if csm else None
            return _detect_primary_and_optional_csm(
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
                csm_energy=csm_energy,
                csm_z=z_f,
                thickness=thickness,
                temperature=temperature,
                bias_v=bias_v,
            )

        csm_energy = fluo_en if csm else None
        return _detect_primary_and_optional_csm(
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
            csm_energy=csm_energy,
            csm_z=z_f2,
            thickness=thickness,
            temperature=temperature,
            bias_v=bias_v,
        )

    if lost2 is False and lost is False:
        electron_distr += _collect_electron_distribution(
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
        electron_distr += _collect_electron_distribution(
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
        x_b = (dep_en * x_0 + dep_en1 * x_f + fluo_en2 * x_f2) / (
            dep_en + dep_en1 + fluo_en2
        )
        y_b = (dep_en * y_0 + dep_en1 * y_f + fluo_en2 * y_f2) / (
            dep_en + dep_en1 + fluo_en2
        )
        return _detect_electron_distribution(
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

    if lost2 is True and lost is False:
        electron_distr += _collect_electron_distribution(
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
        detected = _detect_electron_distribution(
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
        if csm:
            csm_distr = _collect_electron_distribution(
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
            detected += _detect_electron_distribution(
                e_x,
                sensor,
                csm_distr,
                xx,
                yy,
                pixel_size,
                csm,
                thr_0,
                0,
                0,
                rng,
            )
        return detected

    if lost2 is False and lost is True:
        electron_distr += _collect_electron_distribution(
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
        detected = _detect_electron_distribution(
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
        if csm:
            csm_distr = _collect_electron_distribution(
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
            detected += _detect_electron_distribution(
                e_x,
                sensor,
                csm_distr,
                xx,
                yy,
                pixel_size,
                csm,
                thr_0,
                0,
                0,
                rng,
            )
        return detected

    return 0.0


def simulate_event(
    energy: float,
    mu_j: float,
    thickness: float,
    pixel_size: float,
    temperature: float,
    bias_v: float,
    thr_0: float,
    csm: bool,
    sensor: str,
    rho: float,
    at_1: str,
    z_1: int,
    at_2: str,
    z_2: int,
    xx: np.ndarray,
    yy: np.ndarray,
    e_x: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    """Simulate a single event and return detected and lost increments."""
    w_det_increment = np.zeros_like(e_x, dtype=float)
    w_lost_increment = 0.0

    x_0 = -0.5 * pixel_size + 0.5 * pixel_size * rng.random()
    y_0 = -0.5 * pixel_size + 0.5 * pixel_size * rng.random()

    z_0 = absorption_z(mu_j, rng)
    if z_0 > thickness:
        w_lost_increment += 1
        return w_det_increment, w_lost_increment

    k_edge = xraylib.EdgeEnergy(z_1, 0)
    if energy < k_edge:
        w_det_increment += _deposit_and_detect(
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
            e_x,
            xx,
            yy,
            csm,
            thr_0,
        )
        return w_det_increment, w_lost_increment

    int_el, fluorescence, fluo_en = interacting_element(
        z_1, z_2, at_1, at_2, sensor, float(energy), rng
    )
    if not fluorescence:
        w_det_increment += _deposit_and_detect(
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
            e_x,
            xx,
            yy,
            csm,
            thr_0,
        )
        return w_det_increment, w_lost_increment

    if int_el == at_1:
        w_det_increment += _handle_element1_fluorescence(
            energy,
            fluo_en,
            sensor,
            pixel_size,
            temperature,
            thickness,
            bias_v,
            thr_0,
            csm,
            rho,
            x_0,
            y_0,
            z_0,
            xx,
            yy,
            e_x,
            rng,
        )
        return w_det_increment, w_lost_increment

    w_det_increment += _handle_element2_fluorescence(
        energy,
        fluo_en,
        sensor,
        pixel_size,
        temperature,
        thickness,
        bias_v,
        thr_0,
        csm,
        rho,
        at_1,
        z_1,
        at_2,
        z_2,
        x_0,
        y_0,
        z_0,
        xx,
        yy,
        e_x,
        rng,
    )
    return w_det_increment, w_lost_increment


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

    at_1, z_1, at_2, z_2, rho = get_material_properties(sensor)

    # Pre-compute linear attenuation coefficients for each energy bin.
    mu = np.zeros(e_x.size, dtype=float)
    for ii, energy in enumerate(e_x):
        mu[ii] = xraylib.CS_Total_CP(sensor, float(energy)) * rho

    n_grid = 301
    xx = np.linspace(-1.5 * pixel_size, 1.5 * pixel_size, n_grid) * 1e-2
    yy = np.linspace(-1.5 * pixel_size, 1.5 * pixel_size, n_grid) * 1e-2

    start_time = time.time()
    total_events = float(np.sum(w_in))

    for jj, energy in enumerate(e_x):
        mu_j = mu[jj]
        for _ in range(int(w_in[jj])):
            w_det_inc, w_lost_inc = simulate_event(
                energy,
                mu_j,
                thickness,
                pixel_size,
                temperature,
                bias_v,
                thr_0,
                csm,
                sensor,
                rho,
                at_1,
                z_1,
                at_2,
                z_2,
                xx,
                yy,
                e_x,
                rng,
            )
            w_det += w_det_inc
            w_lost[jj] += w_lost_inc

        if verbose and total_events > 0:
            print_simulation_progress(jj, e_x, w_in, start_time, total_events)

    if verbose and total_events > 0:
        sys.stdout.write("\n")
        sys.stdout.flush()

    return w_det, w_lost

