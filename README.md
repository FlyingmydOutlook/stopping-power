# stopping-power

A self-contained Python pipeline that downloads atomic scattering factors from
the [CXRO/Henke database](https://henke.lbl.gov/optical_constants/asf.html),
converts them into the optical Energy Loss Function (ELF), and computes
electron inelastic mean free paths (IMFP) and stopping powers using the
**Ashley optical-data model** (doi:10.1016/0368-2048(88)80019-7).

This is equivalent to the ELF-based stopping power used in CASINO Monte Carlo.

---

## Quick start

```bash
pip install -r requirements.txt
python run_all.py
```

This will:
1. Download Henke `.nff` files for Si, O, Au, C, H from CXRO (requires internet)
2. Compute the ELF for Si, SiO₂, Au, and PMMA
3. Compute IMFP and stopping power for each material
4. Save CSV tables in `results/`
5. Generate plots in `results/`

---

## File structure

```
stopping-power/
├── README.md                      # This file
├── requirements.txt               # numpy, scipy, matplotlib, requests
├── henke_download.py              # Download f1,f2 from CXRO for any element
├── elf_from_henke.py              # Convert f1,f2 → n,k → ε → ELF
├── stopping_power.py              # Ashley model: ELF → DIMFP → IMFP → S(E)
├── run_all.py                     # One-click pipeline script
├── plot_results.py                # Plot stopping power and IMFP curves
├── data/                          # Henke .nff cache + generated ELF .dat files
│   ├── si.nff
│   ├── au.nff
│   ├── ...
│   ├── Si.dat
│   ├── SiO2.dat
│   ├── Au.dat
│   └── PMMA.dat
└── results/                       # Output tables and plots
    ├── Si_stopping_power.csv
    ├── SiO2_stopping_power.csv
    ├── Au_stopping_power.csv
    ├── PMMA_stopping_power.csv
    ├── stopping_power.png
    ├── imfp.png
    └── elf.png
```

---

## Scripts

### `henke_download.py`

Downloads Henke `.nff` files from `https://henke.lbl.gov/optical_constants/sf/`.
Files are cached in `data/` and reused on subsequent runs.

```python
from henke_download import load_henke, download_henke

# Download and load Si data (energies 30 eV – 30 keV)
energy, f1, f2 = load_henke("Si")
```

### `elf_from_henke.py`

Converts f₁, f₂ → complex dielectric function ε(ω) → ELF Im[−1/ε(ω)].
Supports single elements and compounds via stoichiometric sums.

```python
from elf_from_henke import compute_elf_element, compute_elf_compound

# Single element
E, elf, eps1, eps2 = compute_elf_element("Si", density_g_cm3=2.329)

# Compound: SiO2
E, elf, eps1, eps2 = compute_elf_compound(
    "SiO2",
    formula={"Si": 1, "O": 2},
    density_g_cm3=2.2
)
```

The ELF is also saved in cstool-compatible `.dat` format:

```
99.2  149.7  1839.0  -1      ← outer-shell binding energies (eV), -1 sentinel
3.000000e+01  1.234567e-03   ← energy(eV)   ELF
...
-1  -1                        ← end sentinel
```

### `stopping_power.py`

Implements the Ashley optical-data model:

```
d(1/λ)/dω = ELF(ω) · L(K,ω) / (2π a₀ K β²)

L(K,ω) = ln[(1 − ω/2K + √(1 − ω/K)) / (1 − ω/2K − √(1 − ω/K))]

1/λ(K) = ∫₀^{K/2} d(1/λ)/dω dω        [nm⁻¹]
S(K)   = ∫₀^{K/2} ω · d(1/λ)/dω dω    [eV/nm]
```

A relativistic β² correction is included for high electron energies.

```python
from stopping_power import compute_imfp_and_sp, default_energy_grid
from elf_from_henke import compute_elf_element

E_elf, elf, _, _ = compute_elf_element("Si", density_g_cm3=2.329)
K_grid = default_energy_grid(e_min=50, e_max=30000)
K, imfp_nm, sp_eV_nm, imfp_inv = compute_imfp_and_sp(K_grid, E_elf, elf)
```

### `run_all.py`

One-click pipeline. Calls all the above scripts in sequence and saves results.

### `plot_results.py`

Reads CSV files from `results/` and produces publication-quality plots.

---

## Output CSV format

Each material produces `results/{name}_stopping_power.csv`:

```
energy_eV,imfp_nm,stopping_power_eV_per_nm
5.000000e+01,2.500000e+00,1.800000e+01
...
```

---

## Materials included

| Material | Formula     | Density (g/cm³) | Elements |
|----------|-------------|-----------------|----------|
| Si       | Si          | 2.329           | Si       |
| SiO₂     | SiO₂        | 2.200           | Si, O    |
| Au       | Au          | 19.32           | Au       |
| PMMA     | C₅H₈O₂     | 1.190           | C, H, O  |

---

## Energy range

- **Henke data**: 30 eV – 30 keV (data below 30 eV is not used)
- **Stopping power**: computed for electron kinetic energies 50 eV – 30 keV

---

## Physics references

- J.C. Ashley, *J. Electron Spectrosc. Relat. Phenom.* **46** (1988) 199–214.
  doi:[10.1016/0368-2048(88)80019-7](https://doi.org/10.1016/0368-2048(88)80019-7)
- B.L. Henke, E.M. Gullikson, J.C. Davis, *At. Data Nucl. Data Tables* **54** (1993) 181.
  https://henke.lbl.gov/optical_constants/

---

## Dependencies

```
numpy>=1.21
scipy>=1.7
matplotlib>=3.4
requests>=2.25
```