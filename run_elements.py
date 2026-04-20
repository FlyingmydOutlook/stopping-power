"""
run_elements.py
---------------
Compute stopping power S(E) and IMFP for all 92 elements (Z=1 to Z=92)
using the Ashley optical-data model with Henke atomic scattering factors.

Each element is treated as its elemental solid/liquid (standard-state density).
For radioactive elements whose density is not tabulated in the ``periodictable``
package, engineering estimates from the literature are used.

Output
------
results/elements/{Symbol}_stopping_power.csv
    Three-column CSV: energy_eV, imfp_nm, stopping_power_eV_per_nm

results/elements/stopping_power_all_elements.png
results/elements/imfp_all_elements.png

Usage
-----
    python run_elements.py
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
OUT_DIR     = os.path.join(BASE_DIR, "results", "elements")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# All 92 elements: (symbol, Z)
# ---------------------------------------------------------------------------
ELEMENTS_Z1_92 = [
    ("H",  1), ("He", 2), ("Li", 3), ("Be", 4), ("B",  5),
    ("C",  6), ("N",  7), ("O",  8), ("F",  9), ("Ne", 10),
    ("Na", 11), ("Mg", 12), ("Al", 13), ("Si", 14), ("P",  15),
    ("S",  16), ("Cl", 17), ("Ar", 18), ("K",  19), ("Ca", 20),
    ("Sc", 21), ("Ti", 22), ("V",  23), ("Cr", 24), ("Mn", 25),
    ("Fe", 26), ("Co", 27), ("Ni", 28), ("Cu", 29), ("Zn", 30),
    ("Ga", 31), ("Ge", 32), ("As", 33), ("Se", 34), ("Br", 35),
    ("Kr", 36), ("Rb", 37), ("Sr", 38), ("Y",  39), ("Zr", 40),
    ("Nb", 41), ("Mo", 42), ("Tc", 43), ("Ru", 44), ("Rh", 45),
    ("Pd", 46), ("Ag", 47), ("Cd", 48), ("In", 49), ("Sn", 50),
    ("Sb", 51), ("Te", 52), ("I",  53), ("Xe", 54), ("Cs", 55),
    ("Ba", 56), ("La", 57), ("Ce", 58), ("Pr", 59), ("Nd", 60),
    ("Pm", 61), ("Sm", 62), ("Eu", 63), ("Gd", 64), ("Tb", 65),
    ("Dy", 66), ("Ho", 67), ("Er", 68), ("Tm", 69), ("Yb", 70),
    ("Lu", 71), ("Hf", 72), ("Ta", 73), ("W",  74), ("Re", 75),
    ("Os", 76), ("Ir", 77), ("Pt", 78), ("Au", 79), ("Hg", 80),
    ("Tl", 81), ("Pb", 82), ("Bi", 83), ("Po", 84), ("At", 85),
    ("Rn", 86), ("Fr", 87), ("Ra", 88), ("Ac", 89), ("Th", 90),
    ("Pa", 91), ("U",  92),
]

# Fallback densities (g/cm³) for elements whose density is None in periodictable
# Values taken from NIST WebElements / CRC Handbook estimates.
DENSITY_FALLBACK = {
    "At": 7.0,    # astatine — no reliable measurement; semi-metal estimate
    "Rn": 4.4,    # radon   — liquid at boiling point (211 K)
    "Fr": 1.87,   # francium — estimated from Cs/Rb trend
    "Ra": 5.0,    # radium  — CRC Handbook
    "Ac": 10.07,  # actinium — CRC Handbook
}


def get_density(symbol: str) -> float:
    """Return the standard-state density (g/cm³) for the given element."""
    try:
        import periodictable as _pt  # type: ignore
        val = getattr(_pt, symbol).density
        if val is not None:
            return float(val)
    except Exception:
        pass
    if symbol in DENSITY_FALLBACK:
        return DENSITY_FALLBACK[symbol]
    raise ValueError(
        f"No density found for element '{symbol}'. "
        "Add it to DENSITY_FALLBACK in run_elements.py."
    )


# ---------------------------------------------------------------------------
# Step 1 — Ensure Henke .nff files are available
# ---------------------------------------------------------------------------
print("=" * 60)
print("Step 1: Ensuring Henke .nff files are available")
print("=" * 60)

from henke_download import download_henke  # noqa: E402

for sym, Z in ELEMENTS_Z1_92:
    download_henke(sym, cache_dir=DATA_DIR)

print("All Henke .nff files ready.\n")

# ---------------------------------------------------------------------------
# Step 2 — Compute ELF and stopping power for every element
# ---------------------------------------------------------------------------
print("=" * 60)
print("Step 2: Computing ELF + stopping power for Z=1 to 92")
print("=" * 60)

from elf_from_henke import compute_elf_element, save_elf_dat  # noqa: E402
from stopping_power import compute_imfp_and_sp, default_energy_grid  # noqa: E402

K_grid = default_energy_grid(e_min=70.0, e_max=30000.0, n_points=200)

all_results = {}   # symbol → {"energy", "imfp", "sp", "density"}
failed = []

for sym, Z in ELEMENTS_Z1_92:
    t0 = time.time()
    try:
        density = get_density(sym)
        E_elf, elf, eps1, eps2 = compute_elf_element(sym, density)
        elf = np.clip(elf, 0.0, None)

        # Save ELF .dat (cstool format)
        save_elf_dat(sym, E_elf, elf, out_dir=DATA_DIR)

        # Compute IMFP and stopping power
        K_out, imfp_nm, sp, _ = compute_imfp_and_sp(K_grid, E_elf, elf)

        # Save CSV
        out_csv = os.path.join(OUT_DIR, f"{sym}_stopping_power.csv")
        np.savetxt(
            out_csv,
            np.column_stack([K_out, imfp_nm, sp]),
            delimiter=",",
            header="energy_eV,imfp_nm,stopping_power_eV_per_nm",
            comments="",
            fmt="%.6e",
        )

        sp_1kev = float(np.interp(1000.0, K_out, sp))
        lam_1kev = float(np.interp(1000.0, K_out, imfp_nm))
        elapsed = time.time() - t0
        print(
            f"  Z={Z:2d} {sym:2s} : ρ={density:7.3f} g/cm³ | "
            f"S(1keV)={sp_1kev:8.2f} eV/nm | λ(1keV)={lam_1kev:6.2f} nm "
            f"({elapsed:.1f}s)"
        )
        all_results[sym] = {
            "Z": Z,
            "density": density,
            "energy": K_out,
            "imfp": imfp_nm,
            "sp": sp,
        }
    except Exception as exc:
        print(f"  Z={Z:2d} {sym:2s} : FAILED — {exc}")
        failed.append((sym, Z, str(exc)))

# ---------------------------------------------------------------------------
# Step 3 — Summary plots
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 3: Generating summary plots")
print("=" * 60)

# Colour-map: one colour per element, grouped by row/period
cmap = cm.get_cmap("tab20", len(all_results))
colors = {sym: cmap(i) for i, sym in enumerate(all_results)}


def _plot_all(key: str, ylabel: str, title: str, fname: str, finite_only=False):
    fig, ax = plt.subplots(figsize=(12, 7))
    for sym, d in sorted(all_results.items(), key=lambda x: x[1]["Z"]):
        y = d[key]
        x = d["energy"]
        if finite_only:
            mask = np.isfinite(y) & (y < 1e6)
            x, y = x[mask], y[mask]
        ax.loglog(x, y, lw=0.8, color=colors[sym], label=sym)
    ax.set_xlabel("Electron kinetic energy (eV)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(70, 3e4)
    ax.grid(True, which="both", alpha=0.25)
    # Legend outside right
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.01, 1.0),
        fontsize=5, ncol=4, framealpha=0.7,
    )
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    out = os.path.join(OUT_DIR, fname)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


_plot_all(
    "sp", "Stopping power (eV/nm)",
    "Stopping Power — All Elements (Ashley Model)",
    "stopping_power_all_elements.png",
)
_plot_all(
    "imfp", "IMFP (nm)",
    "Inelastic Mean Free Path — All Elements (Ashley Model)",
    "imfp_all_elements.png",
    finite_only=True,
)

# ---------------------------------------------------------------------------
# Step 4 — Summary table at E = 1 keV
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 4: Saving summary table (S and IMFP at 1 keV and 10 keV)")
print("=" * 60)

summary_csv = os.path.join(OUT_DIR, "elements_summary.csv")
header = "Z,symbol,density_g_cm3,S_1keV_eV_per_nm,IMFP_1keV_nm,S_10keV_eV_per_nm,IMFP_10keV_nm"
rows = []
for sym, d in sorted(all_results.items(), key=lambda x: x[1]["Z"]):
    K_out  = d["energy"]
    sp     = d["sp"]
    imfp   = d["imfp"]
    s1  = float(np.interp(1000.0,  K_out, sp))
    l1  = float(np.interp(1000.0,  K_out, imfp))
    s10 = float(np.interp(10000.0, K_out, sp))
    l10 = float(np.interp(10000.0, K_out, imfp))
    rows.append(f"{d['Z']},{sym},{d['density']:.4f},{s1:.4f},{l1:.4f},{s10:.4f},{l10:.4f}")

with open(summary_csv, "w") as fh:
    fh.write(header + "\n")
    fh.write("\n".join(rows) + "\n")
print(f"  Saved {summary_csv}")

# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
print(f"  Computed: {len(all_results)} / 92 elements")
if failed:
    print(f"  Failed  : {len(failed)} elements:")
    for sym, Z, msg in failed:
        print(f"    Z={Z:2d} {sym:2s} — {msg}")
print(f"\n  Per-element CSVs : {OUT_DIR}/")
print(f"  Summary table    : {summary_csv}")
print(f"  Plots            : {OUT_DIR}/*.png")
