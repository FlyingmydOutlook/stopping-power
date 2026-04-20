"""
elf_from_henke.py
-----------------
Convert Henke atomic scattering factors (f1, f2) into the complex dielectric
function ε(ω) and the optical Energy Loss Function (ELF) Im[-1/ε(ω)].

Physics
-------
For a single element with atomic number density n_a (atoms/cm³):

    λ = hc / E          (photon wavelength in Å, E in eV)
    δ = (n_a · r_e · λ²) / (2π) · f1
    β = (n_a · r_e · λ²) / (2π) · f2
    n_complex = (1 - δ) + i·β
    ε = n_complex²
    ε1 = (1-δ)² - β²
    ε2 = 2·(1-δ)·β
    ELF = Im[-1/ε] = ε2 / (ε1² + ε2²)

For compounds, atomic scattering factors are summed weighted by stoichiometry
and the molecular number density is used.

Physical constants
------------------
    r_e = 2.8179403e-13 cm   (classical electron radius)
    hc  = 12398.42 eV·Å
    N_A = 6.02214076e23 mol⁻¹
"""

import os
import numpy as np
from scipy.interpolate import interp1d

from henke_download import load_henke, DATA_DIR

# Physical constants
R_E = 2.8179403e-13   # classical electron radius, cm
HC = 12398.42          # hc in eV·Å
N_A = 6.02214076e23   # Avogadro's number, mol⁻¹


def _get_atomic_mass(element: str) -> float:
    """
    Return atomic mass (g/mol) for the given element symbol.

    Uses the ``periodictable`` package when available, otherwise falls back
    to the built-in table for elements used in the predefined materials.
    """
    try:
        import periodictable as _pt   # type: ignore
        return getattr(_pt, element).mass
    except Exception:
        pass
    # Minimal fallback table (used only if periodictable is not installed)
    _FALLBACK = {
        "H": 1.008, "He": 4.003, "Li": 6.941, "Be": 9.012, "B": 10.811,
        "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
        "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974,
        "S": 32.060, "Cl": 35.450, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
        "Au": 196.967,
    }
    if element not in _FALLBACK:
        raise KeyError(
            f"Atomic mass for '{element}' not found. "
            "Install the 'periodictable' package: pip install periodictable"
        )
    return _FALLBACK[element]


# ---------------------------------------------------------------------------
# Legacy dict kept for backward compatibility; new code uses _get_atomic_mass
# ---------------------------------------------------------------------------
ATOMIC_MASS = {
    "H":  1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "Si": 28.085,
    "Au": 196.967,
}

# Atomic number Z (needed only for reference; not used in Henke ELF calc)
ATOMIC_Z = {
    "H":  1,
    "C":  6,
    "N":  7,
    "O":  8,
    "Si": 14,
    "Au": 79,
}

# Material definitions: {name: (formula_dict, density_g_cm3)}
# formula_dict maps element symbol → stoichiometric count
MATERIALS = {
    "Si":   ({"Si": 1},             2.329),
    "SiO2": ({"Si": 1, "O": 2},     2.200),
    "Au":   ({"Au": 1},             19.32),
    "PMMA": ({"C": 5, "O": 2, "H": 8}, 1.190),
}

# Outer-shell (core-level) binding energies in eV for cstool .dat header
# These are the ionisation thresholds of the outermost core shells.
BINDING_ENERGIES = {
    "Si":   [99.2, 149.7, 1839.0],          # L2,3  L1  K
    "O":    [23.7, 41.6, 543.1],             # (2s, 1s rough values)
    "Au":   [83.6, 107.2, 333.9, 352.0,
             2291.0, 2584.7, 2743.0, 3148.7, 11919.0],
    "C":    [7.0, 284.2],
    "H":    [],
    "PMMA": [30.0],
    "SiO2": [99.2, 149.7, 543.1, 1839.0],
}


def atomic_number_density(formula: dict, density_g_cm3: float) -> float:
    """
    Compute molecular number density (molecules/cm³).

    Parameters
    ----------
    formula : dict
        {element_symbol: stoichiometric_count}
    density_g_cm3 : float
        Mass density in g/cm³.

    Returns
    -------
    float
        Number density of formula units (molecules/cm³).
    """
    molar_mass = sum(_get_atomic_mass(el) * n for el, n in formula.items())
    return (density_g_cm3 * N_A) / molar_mass


def compute_elf_element(element: str, density_g_cm3: float,
                        e_grid: np.ndarray = None,
                        e_min: float = 30.0) -> tuple:
    """
    Compute the optical ELF for a single-element material.

    Parameters
    ----------
    element : str
        Element symbol, e.g. 'Si'.
    density_g_cm3 : float
        Mass density in g/cm³.
    e_grid : np.ndarray or None
        Output energy grid in eV. If None, use raw Henke energies.
    e_min : float
        Minimum energy in eV (default 30 eV).

    Returns
    -------
    energy : np.ndarray
        Photon energy in eV.
    elf : np.ndarray
        Im[-1/ε(ω)] at each energy.
    eps1 : np.ndarray
        Real part of dielectric function.
    eps2 : np.ndarray
        Imaginary part of dielectric function.
    """
    formula = {element: 1}
    molar_mass = _get_atomic_mass(element)
    n_mol = (density_g_cm3 * N_A) / molar_mass   # atoms/cm³

    E_raw, f1_raw, f2_raw = load_henke(element, e_min=e_min)

    if e_grid is not None:
        # Interpolate f1 and f2 onto the requested energy grid
        mask = (e_grid >= E_raw[0]) & (e_grid <= E_raw[-1])
        E = e_grid[mask]
        f1 = interp1d(E_raw, f1_raw, kind="linear", fill_value="extrapolate")(E)
        f2 = interp1d(E_raw, f2_raw, kind="linear", fill_value="extrapolate")(E)
    else:
        E = E_raw
        f1 = f1_raw
        f2 = f2_raw

    wavelength = HC / E   # Å

    # Convert Å → cm for dimensional consistency (r_e is in cm)
    lam_cm = wavelength * 1e-8

    prefactor = (n_mol * R_E * lam_cm**2) / (2 * np.pi)

    delta = prefactor * f1
    beta  = prefactor * f2

    eps1 = (1.0 - delta)**2 - beta**2
    eps2 = 2.0 * (1.0 - delta) * beta

    denom = eps1**2 + eps2**2
    denom = np.where(denom < 1e-30, 1e-30, denom)  # avoid division by zero
    elf = eps2 / denom

    return E, elf, eps1, eps2


def compute_elf_compound(name: str, formula: dict, density_g_cm3: float,
                         e_grid: np.ndarray = None,
                         e_min: float = 30.0) -> tuple:
    """
    Compute the optical ELF for a compound material.

    For a compound A_n B_m ..., the combined atomic scattering factors are:
        f1_total = n·f1_A + m·f1_B + ...
        f2_total = n·f2_A + m·f2_B + ...
    using the molecular number density.

    Parameters
    ----------
    name : str
        Material name for display purposes.
    formula : dict
        {element_symbol: stoichiometric_count}
    density_g_cm3 : float
        Mass density in g/cm³.
    e_grid : np.ndarray or None
        Output energy grid in eV. If None, build a common grid.
    e_min : float
        Minimum energy in eV (default 30 eV).

    Returns
    -------
    energy : np.ndarray
        Photon energy in eV.
    elf : np.ndarray
        Im[-1/ε(ω)] at each energy.
    eps1 : np.ndarray
        Real part of dielectric function.
    eps2 : np.ndarray
        Imaginary part of dielectric function.
    """
    n_mol = atomic_number_density(formula, density_g_cm3)  # molecules/cm³

    # Load and interpolate all elements onto a common grid
    element_data = {}
    e_min_common = e_min
    e_max_common = 30000.0

    for el in formula:
        E_el, f1_el, f2_el = load_henke(el, e_min=e_min)
        element_data[el] = (E_el, f1_el, f2_el)
        e_min_common = max(e_min_common, E_el[0])
        e_max_common = min(e_max_common, E_el[-1])

    if e_grid is None:
        e_grid = np.geomspace(e_min_common, e_max_common, 2000)
    else:
        e_grid = e_grid[(e_grid >= e_min_common) & (e_grid <= e_max_common)]

    # Sum weighted scattering factors
    f1_total = np.zeros_like(e_grid)
    f2_total = np.zeros_like(e_grid)

    for el, count in formula.items():
        E_el, f1_el, f2_el = element_data[el]
        f1_total += count * interp1d(
            E_el, f1_el, kind="linear", fill_value="extrapolate")(e_grid)
        f2_total += count * interp1d(
            E_el, f2_el, kind="linear", fill_value="extrapolate")(e_grid)

    wavelength = HC / e_grid   # Å
    lam_cm = wavelength * 1e-8

    prefactor = (n_mol * R_E * lam_cm**2) / (2 * np.pi)

    delta = prefactor * f1_total
    beta  = prefactor * f2_total

    eps1 = (1.0 - delta)**2 - beta**2
    eps2 = 2.0 * (1.0 - delta) * beta

    denom = eps1**2 + eps2**2
    denom = np.where(denom < 1e-30, 1e-30, denom)
    elf = eps2 / denom

    return e_grid, elf, eps1, eps2


def save_elf_dat(material_name: str, energy: np.ndarray, elf: np.ndarray,
                 out_dir: str = DATA_DIR) -> str:
    """
    Save ELF data in cstool-compatible .dat format.

    Format:
        {outer_shell_binding_energies} -1
        {E1}  {ELF1}
        ...
        -1  -1

    Parameters
    ----------
    material_name : str
        Name used for the output file (e.g. 'Si', 'SiO2').
    energy : np.ndarray
        Energy grid in eV.
    elf : np.ndarray
        Im[-1/ε] values.
    out_dir : str
        Output directory.

    Returns
    -------
    str
        Path to the saved file.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{material_name}.dat")

    binding = BINDING_ENERGIES.get(material_name, [])
    header = "  ".join(f"{b:.1f}" for b in binding) + "  -1"

    with open(out_path, "w") as fh:
        fh.write(header + "\n")
        for e_val, elf_val in zip(energy, elf):
            fh.write(f"{e_val:.6e}  {elf_val:.6e}\n")
        fh.write("-1  -1\n")

    print(f"  Saved ELF data: {out_path}")
    return out_path


def compute_all_materials(out_dir: str = DATA_DIR) -> dict:
    """
    Compute ELF for all predefined materials and save .dat files.

    Returns
    -------
    dict
        {material_name: (energy, elf, eps1, eps2)}
    """
    results = {}

    for name, (formula, density) in MATERIALS.items():
        print(f"\nComputing ELF for {name} (density = {density} g/cm³) ...")

        if len(formula) == 1:
            el = list(formula.keys())[0]
            E, elf, eps1, eps2 = compute_elf_element(el, density)
        else:
            E, elf, eps1, eps2 = compute_elf_compound(name, formula, density)

        # Clip negative ELF values (unphysical artefacts from Henke data)
        elf = np.clip(elf, 0.0, None)

        save_elf_dat(name, E, elf, out_dir=out_dir)
        results[name] = (E, elf, eps1, eps2)
        print(f"  {name}: {len(E)} energy points, ELF max = {elf.max():.4f}")

    return results


if __name__ == "__main__":
    from henke_download import download_elements_for_materials
    download_elements_for_materials()
    compute_all_materials()
