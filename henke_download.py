"""
henke_download.py
-----------------
Obtain atomic scattering factors (f1, f2) from the CXRO/Henke database
for arbitrary elements.

The Henke database provides tabulated f1 and f2 values in .nff files:
  https://henke.lbl.gov/optical_constants/sf/{element_lowercase}.nff

File format:
  Line 1: element description
  Subsequent lines: E(eV)  f1  f2
  Energy range: ~10 eV to 30 keV

We only use data from 30 eV onward as specified.

Data source priority
--------------------
1. Local cache in ``data/`` (previously downloaded / bundled files).
2. The ``periodictable`` package, which ships the Henke .nff tables as
   package data — works offline and requires no extra download.
3. Direct HTTP download from https://henke.lbl.gov/ (requires internet).
"""

import os
import shutil
import numpy as np

HENKE_BASE_URL = "https://henke.lbl.gov/optical_constants/sf/{element}.nff"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _periodictable_nff_path(element: str):
    """
    Return the path to the .nff file bundled with the ``periodictable``
    package, or None if the package / file is not available.
    """
    try:
        import periodictable.xsf as xsf  # type: ignore
        pkg_dir = os.path.join(xsf.get_data_path(""), "xsf")
        candidate = os.path.join(pkg_dir, f"{element.lower()}.nff")
        if os.path.exists(candidate):
            return candidate
    except Exception:
        pass
    return None


def download_henke(element: str, cache_dir: str = DATA_DIR) -> str:
    """
    Obtain the Henke .nff file for a given element symbol.

    The file is resolved in this order:
      1. Local cache in ``cache_dir`` (from a previous run).
      2. The ``periodictable`` package data directory (offline capable).
      3. HTTP download from https://henke.lbl.gov/.

    Parameters
    ----------
    element : str
        Element symbol, e.g. 'si', 'au', 'c', 'h', 'o'.
        Case-insensitive.
    cache_dir : str
        Directory to cache downloaded files.

    Returns
    -------
    str
        Path to the .nff file.
    """
    element = element.lower()
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, f"{element}.nff")

    # 1. Already in local cache?
    if os.path.exists(local_path):
        return local_path

    # 2. Available in the periodictable package?
    pkg_path = _periodictable_nff_path(element)
    if pkg_path is not None:
        shutil.copy(pkg_path, local_path)
        print(f"  Copied {element.upper()} Henke data from periodictable → {local_path}")
        return local_path

    # 3. Download from CXRO
    try:
        import requests  # type: ignore
        url = HENKE_BASE_URL.format(element=element)
        print(f"Downloading Henke data for {element.upper()} from {url} ...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(local_path, "w") as f:
            f.write(response.text)
        print(f"  Saved to {local_path}")
        return local_path
    except Exception as exc:
        raise RuntimeError(
            f"Could not obtain Henke data for '{element}': {exc}\n"
            "Install the 'periodictable' package (pip install periodictable) "
            "or ensure internet access to henke.lbl.gov."
        ) from exc


def load_henke(element: str, e_min: float = 30.0, cache_dir: str = DATA_DIR):
    """
    Load Henke f1, f2 data for an element, filtering to E >= e_min eV.

    Parameters
    ----------
    element : str
        Element symbol, e.g. 'Si', 'Au'.
    e_min : float
        Minimum photon energy in eV (default 30 eV).
    cache_dir : str
        Directory to look for / cache .nff files.

    Returns
    -------
    energy : np.ndarray
        Photon energy in eV.
    f1 : np.ndarray
        Real part of atomic scattering factor f1.
    f2 : np.ndarray
        Imaginary part of atomic scattering factor f2.
    """
    local_path = download_henke(element, cache_dir=cache_dir)

    energy, f1, f2 = [], [], []
    with open(local_path, "r") as fh:
        lines = fh.readlines()

    # Skip first header line
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            e_val = float(parts[0])
            f1_val = float(parts[1])
            f2_val = float(parts[2])
        except ValueError:
            continue

        if e_val < e_min:
            continue
        # Skip sentinel f1 values used in some Henke files
        if f1_val < -9990:
            continue

        energy.append(e_val)
        f1.append(f1_val)
        f2.append(f2_val)

    energy = np.array(energy)
    f1 = np.array(f1)
    f2 = np.array(f2)

    # Sort by energy (should already be sorted, but just in case)
    idx = np.argsort(energy)
    return energy[idx], f1[idx], f2[idx]


def download_elements_for_materials(cache_dir: str = DATA_DIR):
    """
    Obtain Henke data for all elements needed for Si, SiO2, Au, and PMMA.
    """
    elements = ["si", "o", "au", "c", "h"]
    for el in elements:
        download_henke(el, cache_dir=cache_dir)
    print("All required Henke .nff files ready.")


if __name__ == "__main__":
    download_elements_for_materials()
    # Quick test: load Si data
    E, f1, f2 = load_henke("si")
    print(f"Si: {len(E)} data points, E range {E[0]:.1f} – {E[-1]:.1f} eV")
    print(f"  f1 range: {f1.min():.4f} – {f1.max():.4f}")
    print(f"  f2 range: {f2.min():.4e} – {f2.max():.4e}")
