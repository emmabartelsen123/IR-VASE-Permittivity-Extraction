# IR-VASE Permittivity Extraction

Python tool for extracting complex permittivity (ε₁ + iε₂) from IR-VASE ellipsometry data. Supports pseudo-dielectric inversion for bulk substrates and multi-angle Drude-Lorentz fitting for thin films, with configurable materials and output CSVs ready for use in TMM simulations.

---

## Overview

This script processes `.dat` files exported from a J.A. Woollam IR-VASE instrument (WVASE32 / CompleteEASE) and extracts the complex dielectric function ε(ω) = ε₁ + iε₂ for multilayer thin-film stacks. It was developed for characterizing Ge/AlN distributed Bragg reflector (DBR) stacks grown on doped silicon substrates for infrared thermal emission sensing applications.

Two extraction strategies are implemented:

| Sample type | Method |
|---|---|
| Bulk substrate (Si, doped Si) | Exact pseudo-dielectric inversion — no model assumptions |
| Thin film on substrate (AlN, Ge) | Simultaneous 3-angle Drude-Lorentz parametric fit |

All outputs use the standard **e⁻ⁱωᵗ convention** (ε₂ ≥ 0 for absorbing media).

---

## Background

### Sign Convention

WVASE stores data using the **e⁺ⁱωᵗ** time convention:

```
ρ = tan(Ψ) · exp(iΔ)     →     ε₂ < 0 for absorbers
```

This script works internally in the WVASE convention and converts all outputs to the standard **e⁻ⁱωᵗ** physics convention via complex conjugation:

```
ε_output = conj(ε_WVASE)     →     ε₂ ≥ 0 for absorbers
```

### Substrate Extraction

For bulk substrates, the **Aspnes pseudo-dielectric formula** gives exact permittivity at every wavenumber with no model assumptions:

```
ε_pseudo = sin²(θ) · [1 + tan²(θ) · ((1 − ρ) / (1 + ρ))²]
```

A Drude-Lorentz model is also fit to the pseudo-eps for a smooth parametric form, but the raw point-by-point result is what gets saved to CSV.

### Film Extraction

Single-angle pseudo-dielectric inversion is ill-conditioned for thin films — the substrate reflection dominates and the film contribution is small. This script instead uses **simultaneous 3-angle TMM fitting** with a parametric Drude-Lorentz model:

- The substrate permittivity at each AOI is derived from the angle-specific pseudo-eps of the **bare substrate measurement** — exact, with zero model error propagated
- The cost function minimises Ψ and Δ residuals across all three angles simultaneously using differential evolution + optional Nelder-Mead polish
- Delta comparison uses a proper **modular angular difference** to handle branch wrapping

**Drude-Lorentz model** (WVASE e⁺ⁱωᵗ convention):

```
ε(ω) = ε∞  −  ωₚ² / (ω² − i·γ_D·ω)  +  Σⱼ Aⱼ·ω₀ⱼ² / (ω₀ⱼ² − ω² + i·γⱼ·ω)
```

---

## Requirements

```
numpy
scipy
matplotlib
```

```bash
pip install numpy scipy matplotlib
```

---

## Input File Format

The script reads WVASE `.dat` files with data rows in the format:

```
E   wavenumber(cm⁻¹)   angle(°)   Psi(°)   Delta(°)
```

Files are matched by suffix — any filename prefix (e.g. a timestamp) is allowed. Measurements at **three angles of incidence** are expected: 55°, 65°, 75°.

---

## Configuration

All user-facing settings are in the **SAMPLE CONFIGURATION** block near the top of the script. You do not need to edit the core physics functions.

### Global settings

```python
AOI_LIST    = [55.0, 65.0, 75.0]   # angles of incidence (degrees)
PRIMARY_AOI = 65.0                  # angle used for pseudo-eps and output grid
WV_MIN      = 500.0                 # wavenumber range for fitting (cm-1)
WV_MAX      = 5500.0
```

### Sample list

Each entry in `SAMPLES` defines one measurement:

```python
SAMPLES = [
    {
        'type'    : 'substrate',           # 'substrate' or 'film'
        'suffix'  : 'Undoped_Si_VASE.dat', # unique end of your .dat filename
        'label'   : 'undoped_Si',          # used in output filenames and plots
        'model'   : 'substrate_2osc',      # see Model Library below
        'bounds'  : [                      # (min, max) per parameter
            (10.5, 13.5),   # eps_inf
            (0.0,  1.5),    # A1
            (560,  650),    # w01  (cm-1)
            ...
        ],
    },
    {
        'type'      : 'film',
        'suffix'    : 'AlN_on_Si_VASE.dat',
        'label'     : 'AlN',
        'model'     : 'film_2osc',
        'thickness' : 100.0,               # film thickness in nm
        'substrate' : 'undoped_Si',        # must match a substrate label above
        'bounds'    : [...],
    },
]
```

Substrate samples must appear before any film that references them.

### Built-in models

| Model key | Parameters | Typical use |
|---|---|---|
| `substrate_2osc` | ε∞, A₁, ω₀₁, γ₁, A₂, ω₀₂, γ₂ | Undoped Si, phonon-active substrate |
| `substrate_drude` | ε∞, ωₚ, γ_D, A₁, ω₀₁, γ₁, A₂, ω₀₂, γ₂ | Doped semiconductor substrate |
| `film_2osc` | ε∞, A₁, ω₀₁, γ₁, A₂, ω₀₂, γ₂ | Phonon-active film (AlN, SiO₂, HfO₂) |
| `film_drude` | ε∞, ωₚ, γ_D | Free-carrier film (Ge, CdO) |
| `film_cauchy` | ε∞, B, ωₚ, γ_D | Transparent film with weak dispersion |

Custom models can be added to the **MODEL LIBRARY** section at the top of the script.

---

## Usage

```bash
python vase_permittivity_extraction.py [data_dir] [out_dir]
```

| Argument | Description | Default |
|---|---|---|
| `data_dir` | Folder containing `.dat` files | `.` (current directory) |
| `out_dir` | Folder for all output files | `./permittivity_output` |

**Example:**

```bash
python vase_permittivity_extraction.py ./VASE_data ./results
```

---

## Outputs

All files are written to `out_dir/`:

### Permittivity CSVs

Three-column files: `wavenumber_cm-1, eps1, eps2` (e⁻ⁱωᵗ standard, ε₂ ≥ 0).
Comment lines begin with `#`.

| File | Contents |
|---|---|
| `permittivity_<label>.csv` | Extracted permittivity (raw for substrates, DL model for films) |

### Drude-Lorentz parameter files

| File | Contents |
|---|---|
| `<label>_DL_params.txt` | Fitted oscillator parameters with labels and units |

### Diagnostic figures

| File | Contents |
|---|---|
| `fit_<label>.png` | Ψ/Δ measured vs fit at 55°, 65°, 75° |
| `permittivity_<label>.png` | ε₁ and ε₂ vs wavenumber |
| `permittivity_summary.png` | Overview of all extracted permittivities |

---

## Physical Notes

- **Doped Si substrate:** A heavily doped substrate will have a plasma frequency ωₚ in the mid-IR (typically 2000–5000 cm⁻¹). The pseudo-dielectric formula remains exact for a bulk substrate regardless of doping level.
- **AlN:** Two IR-active phonon modes appear in this geometry — E₁(TO) near 667 cm⁻¹ and A₁(TO) near 611 cm⁻¹. The Reststrahlen band between the TO and LO frequencies produces a region of negative ε₁.
- **Ge:** Monatomic, no IR-active phonon. ε₂ ≈ 0 across the mid-IR for intrinsic Ge; a Drude term can be included if the film is doped.
- **Spectral range:** Data below ~500 cm⁻¹ is near the detector sensitivity cutoff and should be treated with caution in downstream simulations.

---

## Citation

If you use this script in published work, please cite the associated paper:

> Bartelsen, E.; et al. "Multi-resonant non-dispersive infrared gas sensing: breaking the selectivity and sensitivity tradeoff." *ACS Photonics* (2025). DOI: [add when confirmed live]

You may also cite the repository directly:

> Bartelsen, E. *IR-VASE Permittivity Extraction* (2026). GitHub: https://github.com/emmabartelsen123/IR-VASE-Permittivity-Extraction

---

## Author

**Emma Bartelsen**  
PhD Candidate, Interdisciplinary Materials Science Program  
Vanderbilt University  
Caldwell Lab — Infrared Nanophotonics  
Email: emma.bartelsen@gmail.com

**Advisor:** Prof. Joshua D. Caldwell  
Department of Mechanical Engineering, Vanderbilt University

---

## Acknowledgments

This work was developed in the [Caldwell Lab](https://engineering.vanderbilt.edu/bio/joshua-caldwell) at Vanderbilt University.

---

## License

MIT License — see `LICENSE` for details.
