"""
plot_results.py
---------------
Plot stopping power S(E) and inelastic mean free path (IMFP) curves for
all materials computed by the pipeline.

Reads CSV files from the results/ directory and produces two figures:
  - results/stopping_power.png
  - results/imfp.png
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Colors and styles for each material
STYLE = {
    "Si":   {"color": "#1f77b4", "ls": "-",  "lw": 2.0},
    "SiO2": {"color": "#ff7f0e", "ls": "--", "lw": 2.0},
    "Au":   {"color": "#d62728", "ls": "-.", "lw": 2.0},
    "PMMA": {"color": "#2ca02c", "ls": ":",  "lw": 2.5},
}


def load_results(results_dir: str = RESULTS_DIR) -> dict:
    """
    Load all CSV result files from the results directory.

    Returns
    -------
    dict
        {material_name: {"energy": ..., "imfp": ..., "sp": ...}}
    """
    data = {}
    csv_files = glob.glob(os.path.join(results_dir, "*_stopping_power.csv"))

    for csv_path in sorted(csv_files):
        basename = os.path.basename(csv_path)
        name = basename.replace("_stopping_power.csv", "")
        arr = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        if arr.ndim < 2 or arr.shape[1] < 3:
            print(f"  Warning: unexpected shape in {csv_path}, skipping.")
            continue
        data[name] = {
            "energy": arr[:, 0],
            "imfp":   arr[:, 1],
            "sp":     arr[:, 2],
        }
        print(f"  Loaded {name}: {len(arr)} points")

    return data


def plot_stopping_power(data: dict, out_dir: str = RESULTS_DIR) -> str:
    """
    Plot stopping power S(E) vs electron kinetic energy.

    Parameters
    ----------
    data : dict
        Output of load_results().
    out_dir : str
        Directory to save the figure.

    Returns
    -------
    str
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, d in sorted(data.items()):
        style = STYLE.get(name, {"color": "gray", "ls": "-", "lw": 1.5})
        ax.loglog(d["energy"], d["sp"],
                  label=name, **style)

    ax.set_xlabel("Electron kinetic energy (eV)", fontsize=13)
    ax.set_ylabel("Stopping power (eV/nm)", fontsize=13)
    ax.set_title("Electron Stopping Power — Ashley Optical-Data Model", fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(50, 3e4)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x:.0f}" if x < 1000 else f"{x/1000:.0f}k"))

    fig.tight_layout()
    out_path = os.path.join(out_dir, "stopping_power.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")
    return out_path


def plot_imfp(data: dict, out_dir: str = RESULTS_DIR) -> str:
    """
    Plot inelastic mean free path (IMFP) vs electron kinetic energy.

    Parameters
    ----------
    data : dict
        Output of load_results().
    out_dir : str
        Directory to save the figure.

    Returns
    -------
    str
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, d in sorted(data.items()):
        style = STYLE.get(name, {"color": "gray", "ls": "-", "lw": 1.5})
        imfp = d["imfp"]
        energy = d["energy"]
        # Filter out infinite / very large values for cleaner plots
        finite_mask = np.isfinite(imfp) & (imfp < 1e6)
        ax.loglog(energy[finite_mask], imfp[finite_mask],
                  label=name, **style)

    ax.set_xlabel("Electron kinetic energy (eV)", fontsize=13)
    ax.set_ylabel("IMFP (nm)", fontsize=13)
    ax.set_title("Inelastic Mean Free Path — Ashley Optical-Data Model", fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(50, 3e4)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x:.0f}" if x < 1000 else f"{x/1000:.0f}k"))

    fig.tight_layout()
    out_path = os.path.join(out_dir, "imfp.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")
    return out_path


def plot_elf(results_dir: str = RESULTS_DIR) -> str:
    """
    Plot ELF vs energy for all materials from their .dat files.

    Returns
    -------
    str
        Path to saved figure.
    """
    import os as _os
    data_dir = _os.path.join(_os.path.dirname(__file__), "data")
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, style in STYLE.items():
        dat_path = _os.path.join(data_dir, f"{name}.dat")
        if not _os.path.exists(dat_path):
            continue
        rows = []
        with open(dat_path) as fh:
            lines = fh.readlines()
        for line in lines[1:]:   # skip header
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                e_val = float(parts[0])
                elf_val = float(parts[1])
            except ValueError:
                continue
            if e_val < 0 or elf_val < 0:
                break
            rows.append((e_val, elf_val))
        if not rows:
            continue
        rows = np.array(rows)
        ax.loglog(rows[:, 0], rows[:, 1], label=name, **style)

    ax.set_xlabel("Photon energy (eV)", fontsize=13)
    ax.set_ylabel("ELF = Im[−1/ε(ω)]", fontsize=13)
    ax.set_title("Optical Energy Loss Function from Henke Data", fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out_path = _os.path.join(results_dir, "elf.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")
    return out_path


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Loading results ...")
    data = load_results()

    if not data:
        print("No result CSV files found. Run run_all.py first.")
    else:
        print("Plotting stopping power ...")
        plot_stopping_power(data)
        print("Plotting IMFP ...")
        plot_imfp(data)

    print("Plotting ELF ...")
    plot_elf()
    print("Done.")
