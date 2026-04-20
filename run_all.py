"""
run_all.py
----------
One-click script: download Henke data → compute ELF → compute stopping power
→ save CSV tables → generate plots.

Usage:
    python run_all.py

This script downloads atomic scattering factor data from the CXRO/Henke
database and computes stopping power for Si, SiO2, Au, and PMMA.
An internet connection is required for the initial download; subsequent
runs use cached .nff files in the data/ directory.
"""

import os
import sys
import time
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Step 1 — Download Henke .nff files
# ---------------------------------------------------------------------------
print("=" * 60)
print("Step 1: Downloading Henke atomic scattering factors")
print("=" * 60)

from henke_download import download_elements_for_materials
download_elements_for_materials(cache_dir=DATA_DIR)

# ---------------------------------------------------------------------------
# Step 2 — Compute ELF for all materials
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 2: Computing ELF (optical energy loss function)")
print("=" * 60)

from elf_from_henke import compute_all_materials
materials = compute_all_materials(out_dir=DATA_DIR)

# ---------------------------------------------------------------------------
# Step 3 — Compute IMFP and stopping power (Ashley model)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 3: Computing IMFP and stopping power (Ashley model)")
print("=" * 60)

from stopping_power import compute_imfp_and_sp, default_energy_grid

K_grid = default_energy_grid(e_min=70.0, e_max=30000.0, n_points=200)
all_results = {}

for name, (E_elf, elf, eps1, eps2) in materials.items():
    t0 = time.time()
    print(f"\n  {name} ...")
    K_out, imfp_nm, sp, imfp_inv = compute_imfp_and_sp(K_grid, E_elf, elf)
    elapsed = time.time() - t0

    out_csv = os.path.join(RESULTS_DIR, f"{name}_stopping_power.csv")
    header = "energy_eV,imfp_nm,stopping_power_eV_per_nm"
    np.savetxt(
        out_csv,
        np.column_stack([K_out, imfp_nm, sp]),
        delimiter=",",
        header=header,
        comments="",
        fmt="%.6e",
    )
    print(f"    Saved {out_csv} ({elapsed:.1f} s)")
    print(f"    S(E=1keV) ≈ {np.interp(1000, K_out, sp):.2f} eV/nm")
    print(f"    λ(E=1keV) ≈ {np.interp(1000, K_out, imfp_nm):.2f} nm")
    all_results[name] = {"energy": K_out, "imfp": imfp_nm, "sp": sp}

# ---------------------------------------------------------------------------
# Step 4 — Generate plots
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 4: Generating plots")
print("=" * 60)

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for headless environments

from plot_results import plot_stopping_power, plot_imfp, plot_elf

plot_stopping_power(all_results, out_dir=RESULTS_DIR)
plot_imfp(all_results, out_dir=RESULTS_DIR)
plot_elf(results_dir=RESULTS_DIR)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Pipeline complete!")
print("=" * 60)
print(f"  ELF .dat files : {DATA_DIR}/")
print(f"  CSV tables     : {RESULTS_DIR}/")
print(f"  Plots          : {RESULTS_DIR}/*.png")
print()
print("Generated files:")
for fname in sorted(os.listdir(RESULTS_DIR)):
    fpath = os.path.join(RESULTS_DIR, fname)
    size = os.path.getsize(fpath)
    print(f"  {fname:45s}  ({size/1024:.1f} kB)")
