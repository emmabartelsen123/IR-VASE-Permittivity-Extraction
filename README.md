# IR-VASE Permittivity Extraction

A Python tool for extracting complex permittivity (ε₁ + iε₂) from infrared variable-angle spectroscopic ellipsometry (IR-VASE) data exported by WVASE32 / CompleteEASE. Designed for multilayer thin-film stacks grown on doped silicon substrates.

---

## Overview

This script processes `.dat` files from a J.A. Woollam IR-VASE instrument and extracts the complex dielectric function for up to four materials in a single automated pipeline:

| Sample | Method |
|---|---|
| Undoped Si substrate | Pseudo-dielectric inversion (exact, bulk) |
| Doped Si substrate | Pseudo-dielectric inversion (exact, bulk) |
| AlN thin film on doped Si | 3-angle Drude-Lorentz parametric fit |
| Ge thin film on doped Si | 3-angle Drude-Lorentz parametric fit |

All outputs are saved in the standard **e⁻ⁱωᵗ convention** (ε₂ ≥ 0 for absorbing media).

---

## Background

### Sign Convention

WVASE stores ellipsometric data using the **e⁺ⁱωᵗ** time convention, where:

```
ρ = tan(Ψ) · exp(iΔ)
ε = ε₁ + iε₂  with  ε₂ < 0 for absorbers
```

This script internally works in the WVASE convention throughout and converts outputs to the standard **e⁻ⁱωᵗ** physics convention by complex conjugation:

```
ε_standard = conj(ε_WVASE)   →   ε₂ ≥ 0 for absorbers
```

### Substrate Extraction Strategy

For bulk substrates (undoped Si, doped Si), the **Aspnes pseudo-dielectric formula** gives the exact permittivity with no model assumptions:

```
ε_pseudo = sin²(θ) · [1 + tan²(θ) · ((1 - ρ) / (1 + ρ))²]
```

A Drude-Lorentz model is then fit to the raw pseudo-eps for a smooth parametric form, but the raw point-by-point result is what gets saved to CSV.

### Film Extraction Strategy

For thin films (AlN, Ge), single-angle pseudo-dielectric inversion is ill-conditioned — the substrate reflection dominates and the film contribution is small. Instead, this script uses **simultaneous 3-angle TMM fitting** with a parametric Drude-Lorentz model:

- The substrate permittivity at each AOI is derived from the angle-specific pseudo-eps of the **bare doped Si** measurement — this is exact and introduces zero model error into the substrate
- The cost function minimises Ψ and Δ residuals across all three angles simultaneously using differential evolution + optional Nelder-Mead polish
- Delta comparison uses a proper **modular angular difference** to handle branch wrapping

**AlN model** — two Lorentz oscillators (E₁ TO ~667 cm⁻¹, A₁ TO ~611 cm⁻¹):
```
ε(ω) = ε∞ + Σⱼ Aⱼ·ω₀ⱼ² / (ω₀ⱼ² - ω² + iγⱼω)
```

**Ge model** — Cauchy dispersion + optional Drude free carriers:
```
ε(ω) = ε∞ + B·(ω/3000)² − ωₚ²/(ω² − iγ_D·ω)
```

---

## Requirements

```
numpy
scipy
matplotlib
```

Install with:

```bash
pip install numpy scipy matplotlib
```

No WVASE software required — the script reads the raw `.dat` export files directly.

---

## Input File Format

The script expects WVASE-format `.dat` files with data rows structured as:

```
E   wavenumber(cm⁻¹)   angle(°)   Psi(°)   Delta(°)   [N]   [C]
```

Files are identified by suffix matching. The expected filename suffixes are:

| Suffix | Sample |
|---|---|
| `Undoped_Si_VASE_55to75by10.dat` | Undoped Si (double-side polished) |
| `Doped_Si_VASE_55to75by10.dat` | Doped Si (single-side polished) |
| `AlN_doped_Si_VASE_55to75by10.dat` | AlN film on doped Si |
| `Ge_doped_Si_VASE_55to75by10.dat` | Ge film on doped Si |

Files can have any prefix (e.g. timestamps) — only the suffix must match.

Measurements at **three angles of incidence** are expected: 55°, 65°, 75°.

---

## Usage

```bash
python vase_permittivity_extraction.py <data_dir> <out_dir>
```

**Arguments:**

| Argument | Description | Default |
|---|---|---|
| `data_dir` | Directory containing the four `.dat` files | `.` (current directory) |
| `out_dir` | Directory for all output files | `./permittivity_output` |

**Example:**

```bash
python vase_permittivity_extraction.py ./VASE_data ./results
```

---

## Outputs

All files are written to `<out_dir>/`:

### Permittivity CSVs

Three-column CSV files: `wavenumber_cm-1, eps1, eps2`

| File | Contents |
|---|---|
| `permittivity_undoped_Si.csv` | Undoped Si — pseudo-dielectric (raw) |
| `permittivity_doped_Si.csv` | Doped Si — pseudo-dielectric (raw) |
| `permittivity_AlN.csv` | AlN film — DL parametric model |
| `permittivity_Ge.csv` | Ge film — DL parametric model |

All CSVs use the standard e⁻ⁱωᵗ convention (ε₂ ≥ 0). Comment lines begin with `#`.

### Drude-Lorentz Parameter Files

Plain text files listing the fitted oscillator parameters for each material:

| File | Contents |
|---|---|
| `undoped_Si_DL_params.txt` | ε∞, two Lorentz oscillators |
| `doped_Si_DL_params.txt` | ε∞, Drude (ωₚ, γ_D), two Lorentz oscillators |
| `AlN_DL_params.txt` | ε∞, two Lorentz oscillators |
| `Ge_DL_params.txt` | ε∞, Cauchy B term, Drude (ωₚ, γ_D) |

### Diagnostic Figures

| File | Contents |
|---|---|
| `fit_undoped_Si.png` | Ψ/Δ measured vs DL fit at 55°, 65°, 75° |
| `fit_doped_Si.png` | Ψ/Δ measured vs DL fit at 55°, 65°, 75° |
| `fit_AlN.png` | Ψ/Δ measured vs DL fit at 55°, 65°, 75° |
| `fit_Ge.png` | Ψ/Δ measured vs DL fit at 55°, 65°, 75° |
| `permittivity_undoped_Si.png` | ε₁ and ε₂ with raw pseudo-eps overlay |
| `permittivity_doped_Si.png` | ε₁ and ε₂ with raw pseudo-eps overlay |
| `permittivity_AlN.png` | ε₁ and ε₂ |
| `permittivity_Ge.png` | ε₁ and ε₂ |
| `permittivity_summary.png` | 2×2 overview of all four materials |

---

## Adapting to Different Samples

To change film thicknesses, edit the constants near the top of `run()`:

```python
ALN_D_NM = 45.0    # AlN film thickness in nm
GE_D_NM  = 207.0   # Ge film thickness in nm
```

To change the Drude-Lorentz parameter search bounds (e.g. for a different doping level or material), modify the `bounds_*` lists in `run()`. Each tuple is `(min, max)` for one parameter.

To add a new material, define a new `model_*` function using `eps_dl_wvase()` and add a corresponding fitting block in `run()`.

---

## Physical Notes

- The doped Si substrate used here is **heavily doped** (plasma frequency ωₚ ~ 3900 cm⁻¹, free carrier scattering rate γ_D ~ 380 cm⁻¹). The pseudo-dielectric formula remains valid for a bulk substrate regardless of doping level.
- AlN has two IR-active phonon modes in this geometry: **E₁(TO) at ~667 cm⁻¹** and **A₁(TO) at ~611 cm⁻¹**. The Reststrahlen band between the TO and LO frequencies produces a region of negative ε₁.
- Ge is monatomic and has no IR-active phonon, so ε₂ ≈ 0 across the mid-IR. The extracted ωₚ ≈ 0 is consistent with intrinsic (undoped) Ge.
- Data below ~500 cm⁻¹ is near the detector sensitivity cutoff and should be treated with caution in downstream TMM simulations.

---

## Citation

If you use this script in published work, please cite the associated paper and acknowledge the J.A. Woollam IR-VASE instrument used for data collection.

---

## License

MIT License — free to use, modify, and distribute with attribution.
