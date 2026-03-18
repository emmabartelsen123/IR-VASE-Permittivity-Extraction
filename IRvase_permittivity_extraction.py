"""
IR-VASE Permittivity Extraction
================================
Extracts complex permittivity (ε = ε₁ + iε₂) from infrared variable-angle
spectroscopic ellipsometry (IR-VASE) data exported by WVASE32 / CompleteEASE.

Two extraction strategies are supported:

  1. SUBSTRATE  (bulk, semi-infinite)
     Exact pseudo-dielectric inversion — no model assumptions. A Drude-Lorentz
     model is also fit to the raw result for a smooth parametric form.

  2. FILM  (thin film on a known substrate)
     Parametric Drude-Lorentz fit using all measured angles simultaneously.
     The substrate permittivity is derived from the bare-substrate measurement
     at each angle (exact, no model error propagation).

Sign convention
---------------
WVASE uses the e+iωt time convention:
  ρ = tan(Ψ)·exp(iΔ)   →   ε₂ < 0 for absorbers

All output CSVs use the standard physics e-iωt convention:
  ε_out = conj(ε_WVASE)   →   ε₂ ≥ 0 for absorbers

Usage
-----
  python vase_permittivity_extraction.py [data_dir] [out_dir]

  data_dir : folder containing .dat files (default: current directory)
  out_dir  : folder for output files     (default: ./permittivity_output)

Configuring for your samples
------------------------------
Edit the SAMPLE CONFIGURATION section below. Each entry in SAMPLES defines
one measurement. Set:

  'type'       : 'substrate' or 'film'
  'suffix'     : unique end of your .dat filename (can include a prefix wildcard)
  'label'      : human-readable name used in plots and output filenames
  'model'      : Drude-Lorentz model to fit (see MODEL LIBRARY below)
  'bounds'     : parameter search bounds for differential evolution
  'thickness'  : film thickness in nm (film samples only)
  'substrate'  : label of the substrate sample to use (film samples only)

Requirements
------------
  numpy, scipy, matplotlib

  pip install numpy scipy matplotlib

Output files (written to out_dir)
----------------------------------
  permittivity_<label>.csv     wavenumber_cm-1, eps1, eps2  (e-iωt standard)
  <label>_DL_params.txt        fitted Drude-Lorentz parameters
  fit_<label>.png              Ψ/Δ measured vs fit at each angle
  permittivity_<label>.png     ε₁ and ε₂ plot
  permittivity_summary.png     overview of all extracted permittivities
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# SAMPLE CONFIGURATION  –  edit this section for your measurements
# ══════════════════════════════════════════════════════════════════════════════

# Angles of incidence used in your VASE measurement (degrees)
AOI_LIST = [55.0, 65.0, 75.0]

# Primary angle used for pseudo-dielectric inversion and output grid
PRIMARY_AOI = 65.0

# Wavenumber range to include in fitting (cm-1)
WV_MIN = 500.0
WV_MAX = 5500.0

# ── Model library ─────────────────────────────────────────────────────────────
# Each model is a function  f(wv, params) -> complex permittivity array
# in the WVASE e+iωt convention.
#
# Built-in models (defined further below):
#   'substrate_2osc'  : ε∞ + 2 Lorentz oscillators  (undoped Si, AlN, etc.)
#   'substrate_drude' : ε∞ + Drude + 2 Lorentz osc  (doped semiconductor)
#   'film_drude'      : ε∞ + Drude                   (free-carrier film, Ge)
#   'film_cauchy'     : ε∞ + Cauchy dispersion + Drude (transparent film)
#   'film_2osc'       : ε∞ + 2 Lorentz oscillators  (phonon-active film)
#
# You can add custom models at the bottom of the MODEL LIBRARY section.

# ── Sample list ───────────────────────────────────────────────────────────────
# Processed in order. Substrate samples must appear before any film that
# references them.

SAMPLES = [
    # ── Example 1: undoped Si substrate ──────────────────────────────────────
    # Replace 'suffix' with the end of your actual filename.
    # bounds: [(eps_inf_min, max), (A1_min, max), (w01_min, max), (g1_min, max),
    #          (A2_min, max), (w02_min, max), (g2_min, max)]
    {
        'type'    : 'substrate',
        'suffix'  : 'Undoped_Si_VASE.dat',
        'label'   : 'undoped_Si',
        'model'   : 'substrate_2osc',
        'bounds'  : [
            (10.5, 13.5),   # eps_inf
            (0.0,  1.5),    # A1  (oscillator strength)
            (560,  650),    # w01 (cm-1)  Si TO phonon ~610 cm-1
            (5,    100),    # g1  (cm-1)
            (0.0,  0.8),    # A2
            (430,  560),    # w02 (cm-1)
            (5,    120),    # g2  (cm-1)
        ],
    },

    # ── Example 2: doped Si substrate ────────────────────────────────────────
    # bounds: [eps_inf, wp, gamma_D, A1, w01, g1, A2, w02, g2]
    {
        'type'    : 'substrate',
        'suffix'  : 'Doped_Si_VASE.dat',
        'label'   : 'doped_Si',
        'model'   : 'substrate_drude',
        'bounds'  : [
            (10.0, 13.5),   # eps_inf
            (500,  6000),   # wp   (cm-1)  plasma frequency
            (5,    1500),   # gD   (cm-1)  Drude scattering rate
            (0.0,  2.0),    # A1
            (570,  650),    # w01  (cm-1)
            (5,    100),    # g1   (cm-1)
            (0.0,  0.8),    # A2
            (430,  570),    # w02  (cm-1)
            (5,    120),    # g2   (cm-1)
        ],
    },

    # ── Example 3: AlN film on doped Si ──────────────────────────────────────
    # bounds: [eps_inf, A1, w01, g1, A2, w02, g2]
    {
        'type'       : 'film',
        'suffix'     : 'AlN_on_Si_VASE.dat',
        'label'      : 'AlN',
        'model'      : 'film_2osc',
        'thickness'  : 100.0,       # <-- set your film thickness in nm
        'substrate'  : 'doped_Si',  # must match the 'label' of a substrate above
        'bounds'     : [
            (3.0,  7.0),    # eps_inf   AlN optical ~4.6
            (0.5,  6.0),    # A1
            (630,  730),    # w01 (cm-1) E1(TO) ~667 cm-1
            (2,    50),     # g1
            (0.3,  5.0),    # A2
            (560,  640),    # w02 (cm-1) A1(TO) ~611 cm-1
            (2,    60),     # g2
        ],
    },

    # ── Example 4: Ge film on doped Si ───────────────────────────────────────
    # bounds: [eps_inf, wp, gamma_D]
    {
        'type'       : 'film',
        'suffix'     : 'Ge_on_Si_VASE.dat',
        'label'      : 'Ge',
        'model'      : 'film_drude',
        'thickness'  : 200.0,       # <-- set your film thickness in nm
        'substrate'  : 'doped_Si',
        'bounds'     : [
            (13.0, 19.0),   # eps_inf   Ge ~16-17 in mid-IR
            (0,    1500),   # wp  (cm-1)  set upper bound to 0 if undoped
            (5,    800),    # gD  (cm-1)
        ],
    },

    # ── Add more samples here following the same pattern ─────────────────────
    # To add a custom material, define a new model function in the
    # MODEL LIBRARY section below and reference it by name here.
]

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LIBRARY  –  add custom models here if needed
# ══════════════════════════════════════════════════════════════════════════════

def _eps_dl_wvase(wv, eps_inf, wp, gD, lorentz):
    """
    Drude-Lorentz permittivity in the WVASE e+iωt convention.

    ε(ω) = ε∞
           − ωₚ² / (ω² − i·γ_D·ω)                     [Drude; skip if wp=0]
           + Σⱼ Aⱼ·ω₀ⱼ² / (ω₀ⱼ² − ω² + i·γⱼ·ω)       [Lorentz oscillators]

    All frequencies in cm-1.  Returns complex array (ε₂ < 0 for absorbers).
    """
    w   = np.asarray(wv, float)
    eps = np.full(len(w), eps_inf, dtype=complex)
    if wp > 0:
        eps -= wp**2 / (w**2 - 1j * gD * w)
    for (A, w0, g) in lorentz:
        eps += A * w0**2 / (w0**2 - w**2 + 1j * g * w)
    return eps


def _model_substrate_2osc(wv, p):
    """ε∞ + 2 Lorentz oscillators (undoped semiconductor or phonon-active substrate).
    params: [eps_inf, A1, w01, g1, A2, w02, g2]"""
    ei, A1, w01, g1, A2, w02, g2 = p
    return _eps_dl_wvase(wv, ei, 0, 0, [(A1, w01, g1), (A2, w02, g2)])


def _model_substrate_drude(wv, p):
    """ε∞ + Drude + 2 Lorentz oscillators (doped semiconductor substrate).
    params: [eps_inf, wp, gD, A1, w01, g1, A2, w02, g2]"""
    ei, wp, gD, A1, w01, g1, A2, w02, g2 = p
    return _eps_dl_wvase(wv, ei, wp, gD, [(A1, w01, g1), (A2, w02, g2)])


def _model_film_drude(wv, p):
    """ε∞ + Drude (free-carrier film; set wp=0 bound if intrinsic).
    params: [eps_inf, wp, gD]"""
    ei, wp, gD = p
    return _eps_dl_wvase(wv, ei, wp, gD, [])


def _model_film_cauchy(wv, p):
    """ε∞ + Cauchy dispersion + Drude (transparent/weakly-absorbing film).
    params: [eps_inf, B_cauchy, wp, gD]
    Cauchy term: B * (ω / 3000)²  — adjust the reference wavenumber if needed."""
    ei, B, wp, gD = p
    w   = np.asarray(wv, float)
    eps = ei + B * (w / 3000.0)**2 + 0j
    if wp > 0:
        eps -= wp**2 / (w**2 - 1j * gD * w)
    return eps


def _model_film_2osc(wv, p):
    """ε∞ + 2 Lorentz oscillators (phonon-active film, e.g. AlN, SiO₂, HfO₂).
    params: [eps_inf, A1, w01, g1, A2, w02, g2]"""
    ei, A1, w01, g1, A2, w02, g2 = p
    return _eps_dl_wvase(wv, ei, 0, 0, [(A1, w01, g1), (A2, w02, g2)])


# ── Model dispatch table ──────────────────────────────────────────────────────
_MODEL_FNS = {
    'substrate_2osc'  : _model_substrate_2osc,
    'substrate_drude' : _model_substrate_drude,
    'film_drude'      : _model_film_drude,
    'film_cauchy'     : _model_film_cauchy,
    'film_2osc'       : _model_film_2osc,
    # Add custom entries here:  'my_model': my_model_function
}

# Parameter name lists for output .txt files (must match bounds length)
_PARAM_NAMES = {
    'substrate_2osc'  : ['eps_inf', 'A1', 'w01_cm-1', 'gamma1_cm-1',
                         'A2', 'w02_cm-1', 'gamma2_cm-1'],
    'substrate_drude' : ['eps_inf', 'wp_cm-1', 'gamma_D_cm-1',
                         'A1', 'w01_cm-1', 'gamma1_cm-1',
                         'A2', 'w02_cm-1', 'gamma2_cm-1'],
    'film_drude'      : ['eps_inf', 'wp_cm-1', 'gamma_D_cm-1'],
    'film_cauchy'     : ['eps_inf', 'B_cauchy', 'wp_cm-1', 'gamma_D_cm-1'],
    'film_2osc'       : ['eps_inf', 'A1', 'w01_cm-1', 'gamma1_cm-1',
                         'A2', 'w02_cm-1', 'gamma2_cm-1'],
}


# ══════════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS  –  no need to edit below this line
# ══════════════════════════════════════════════════════════════════════════════

# ── I/O ───────────────────────────────────────────────────────────────────────

def load_vase(path):
    """Parse a WVASE .dat file.  Returns array [N, 4]: wv, angle, Psi, Delta."""
    rows = []
    with open(path, encoding='latin-1') as fh:
        for line in fh:
            p = line.split()
            if len(p) >= 5 and p[0] == 'E':
                try:
                    rows.append([float(p[1]), float(p[2]),
                                 float(p[3]), float(p[4])])
                except ValueError:
                    pass
    if not rows:
        raise ValueError(f"No data rows found in {path}. "
                         "Check that the file is a WVASE .dat export.")
    return np.array(rows)


def get_spectrum(data, angle):
    """Return (wv, Psi, Delta) sorted ascending for one angle of incidence."""
    mask = np.abs(data[:, 1] - angle) < 0.01
    d    = data[mask]
    if len(d) == 0:
        raise ValueError(f"No data found for angle {angle}°. "
                         f"Available angles: {np.unique(data[:, 1])}")
    idx = np.argsort(d[:, 0])
    return d[idx, 0], d[idx, 2], d[idx, 3]


def find_file(data_dir, suffix):
    """Find a file whose name ends with *suffix* inside data_dir."""
    matches = [fn for fn in os.listdir(data_dir) if fn.endswith(suffix)]
    if not matches:
        raise FileNotFoundError(
            f"No file ending with '{suffix}' found in '{data_dir}'.\n"
            f"Files present: {os.listdir(data_dir)}")
    if len(matches) > 1:
        print(f"  Warning: multiple files match '{suffix}': {matches}. "
              f"Using {matches[0]}.")
    return os.path.join(data_dir, matches[0])


def save_csv(wv, eps_wvase, path, label):
    """Save permittivity in standard e-iωt convention (conj of WVASE)."""
    eps_out = np.conj(eps_wvase)
    arr = np.column_stack([wv, eps_out.real, eps_out.imag])
    hdr = (f"# {label}\n"
           "# Convention: e^-iwt standard  (eps2 >= 0 for absorbers)\n"
           "# wavenumber_cm-1,eps1,eps2")
    np.savetxt(path, arr, delimiter=',', header=hdr, comments='')
    print(f"  Saved: {os.path.basename(path)}")


def save_params_txt(path, label, model_name, params):
    """Save Drude-Lorentz parameters to a plain-text file."""
    names = _PARAM_NAMES.get(model_name,
                              [f'p{i}' for i in range(len(params))])
    with open(path, 'w') as fh:
        fh.write(f"# Drude-Lorentz parameters: {label}\n")
        fh.write(f"# Model: {model_name}\n")
        fh.write("# All frequencies in cm-1\n")
        for name, val in zip(names, params):
            fh.write(f"{name:25s} = {val:.6f}\n")
    print(f"  Params: {os.path.basename(path)}")


# ── Pseudo-dielectric (WVASE convention) ──────────────────────────────────────

def pseudo_eps(psi_deg, delta_deg, theta_deg):
    """
    Aspnes pseudo-dielectric formula.
    Returns ε in WVASE e+iωt convention (ε₂ < 0 for absorbers).
    Exact for bulk semi-infinite substrates.
    """
    psi   = np.deg2rad(psi_deg)
    delta = np.deg2rad(delta_deg)
    theta = np.deg2rad(theta_deg)
    rho   = np.tan(psi) * np.exp(1j * delta)
    return (np.sin(theta)**2 *
            (1 + np.tan(theta)**2 * ((1 - rho) / (1 + rho))**2))


# ── Thin-film Fresnel (air | film | substrate, WVASE convention) ──────────────

def fresnel_rho(eps_film, eps_sub, d_nm, wv, theta_deg):
    """
    Compute ρ = rp/rs for a single film on a substrate.
    All ε values in WVASE e+iωt convention.
    Returns ρ directly comparable to tan(Ψ)·exp(iΔ) from the .dat file.
    """
    lam = 1e7 / np.asarray(wv, float)    # wavelength in nm
    t0  = np.deg2rad(theta_deg)
    n1  = np.sqrt(eps_film + 0j)
    n2  = np.sqrt(eps_sub  + 0j)
    s0, c0 = np.sin(t0), np.cos(t0)
    c1  = np.sqrt(1 - (s0 / n1)**2 + 0j)
    c2  = np.sqrt(1 - (s0 / n2)**2 + 0j)

    delta1 = 2 * np.pi * n1 * c1 * d_nm / lam   # phase thickness
    eP     = np.exp(-2j * delta1)

    r01s = (c0 - n1*c1) / (c0 + n1*c1)
    r12s = (n1*c1 - n2*c2) / (n1*c1 + n2*c2)
    rs   = (r01s + r12s * eP) / (1 + r01s * r12s * eP)

    r01p = (n1*c0 - c1) / (n1*c0 + c1)
    r12p = (n2*c1 - n1*c2) / (n2*c1 + n1*c2)
    rp   = (r01p + r12p * eP) / (1 + r01p * r12p * eP)

    return rp / rs


def rho_to_psi_delta(rho):
    psi   = np.rad2deg(np.arctan(np.abs(rho)))
    delta = np.rad2deg(np.angle(rho))
    return psi, delta


def delta_diff(a, b):
    """Signed angular difference (a - b) wrapped to (-180, 180]."""
    return ((a - b) + 180) % 360 - 180


# ── Substrate: exact pseudo-eps interpolant for use as film substrate ─────────

def make_substrate_fn(data, angles=None):
    """
    Build a callable  eps_sub(wv, angle)  that returns the substrate
    permittivity (WVASE convention) by interpolating the pseudo-eps.
    This is exact — it reproduces the measured Ψ/Δ of the bare substrate.
    """
    if angles is None:
        angles = AOI_LIST
    interps = {}
    for ang in angles:
        wv, psi, dlt = get_spectrum(data, ang)
        eps = pseudo_eps(psi, dlt, ang)
        interps[ang] = (
            interp1d(wv, eps.real, bounds_error=False, fill_value='extrapolate'),
            interp1d(wv, eps.imag, bounds_error=False, fill_value='extrapolate'),
        )
    def fn(wv, angle):
        wv = np.asarray(wv, float)
        fr, fi = interps[angle]
        return fr(wv) + 1j * fi(wv)
    return fn


# ── Fitting ───────────────────────────────────────────────────────────────────

def fit_dl_to_pseudo(wv, eps_raw, model_fn, bounds, maxiter=800):
    """Fit a Drude-Lorentz model to a raw pseudo-eps array (substrate)."""
    def cost(p):
        return np.mean(np.abs(model_fn(wv, p) - eps_raw)**2)
    res = differential_evolution(cost, bounds, maxiter=maxiter,
                                 seed=42, tol=1e-10, polish=True, workers=1)
    return res.x, res.fun


def fit_film_dl(film_data, sub_fn, d_nm, model_fn, bounds,
                angles=None, wv_min=None, wv_max=None, maxiter=1500):
    """
    Fit a Drude-Lorentz model for a thin film by minimising Ψ/Δ residuals
    across all angles simultaneously.

    sub_fn  : callable(wv, angle) → ε_sub (WVASE convention, from bare substrate)
    d_nm    : film thickness in nm
    """
    if angles is None: angles = AOI_LIST
    if wv_min is None: wv_min = WV_MIN
    if wv_max is None: wv_max = WV_MAX

    angle_spectra = []
    for ang in angles:
        wv, psi_m, del_m = get_spectrum(film_data, ang)
        mask = (wv >= wv_min) & (wv <= wv_max)
        angle_spectra.append((wv[mask], psi_m[mask], del_m[mask], ang))

    def cost(p):
        tot = 0.0
        for wv, psi_m, del_m, ang in angle_spectra:
            eps_f = model_fn(wv, p)
            eps_s = sub_fn(wv, ang)
            rho_c = fresnel_rho(eps_f, eps_s, d_nm, wv, ang)
            psi_c, del_c = rho_to_psi_delta(rho_c)
            tot += np.sum((psi_m - psi_c)**2 + delta_diff(del_m, del_c)**2)
        return tot / sum(2 * len(a[0]) for a in angle_spectra)

    print("  Running differential evolution …")
    res = differential_evolution(cost, bounds, maxiter=maxiter,
                                 seed=42, tol=1e-10, polish=True,
                                 workers=1, popsize=18)
    print(f"  converged={res.success}  MSE={res.fun:.5f}")
    if res.fun > 5.0:
        print("  MSE > 5 — running Nelder-Mead polish …")
        res2 = minimize(cost, res.x, method='Nelder-Mead',
                        options={'maxiter': 100000, 'xatol': 1e-8,
                                 'fatol': 1e-10, 'adaptive': True})
        if res2.fun < res.fun:
            print(f"  Polish improved MSE to {res2.fun:.5f}")
            return res2.x, res2.fun
    return res.x, res.fun


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_substrate_fit(label, data, model_fn, params, out_path,
                       angles=None, wv_min=None, wv_max=None):
    if angles is None: angles = AOI_LIST
    if wv_min is None: wv_min = WV_MIN
    if wv_max is None: wv_max = WV_MAX

    fig, axes = plt.subplots(2, len(angles), figsize=(5 * len(angles), 8),
                              sharex=True)
    for col, ang in enumerate(angles):
        wv, psi_m, del_m = get_spectrum(data, ang)
        mask = (wv >= wv_min) & (wv <= wv_max)
        wv = wv[mask]; psi_m = psi_m[mask]; del_m = del_m[mask]
        eps_c = model_fn(wv, params)
        n_c   = np.sqrt(eps_c)
        t0    = np.deg2rad(ang)
        ct    = np.sqrt(1 - (np.sin(t0) / n_c)**2 + 0j)
        rp    = (n_c * np.cos(t0) - ct) / (n_c * np.cos(t0) + ct)
        rs    = (np.cos(t0) - n_c * ct)  / (np.cos(t0) + n_c * ct)
        psi_c, del_c = rho_to_psi_delta(rp / rs)

        ax_p = axes[0, col]; ax_d = axes[1, col]
        ax_p.plot(wv, psi_m, 'k.', ms=1.5, label='Measured')
        ax_p.plot(wv, psi_c, 'r-', lw=1.5, label='DL fit')
        ax_p.set_title(f'{ang}°'); ax_p.set_ylabel('Ψ (deg)'); ax_p.legend(fontsize=7)
        ax_d.plot(wv, del_m, 'k.', ms=1.5)
        ax_d.plot(wv, del_c, 'r-', lw=1.5)
        ax_d.set_ylabel('Δ (deg)'); ax_d.set_xlabel('Wavenumber (cm⁻¹)')

    fig.suptitle(f'{label}  –  Ψ/Δ DL fit', fontweight='bold')
    plt.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  Plot: {os.path.basename(out_path)}")


def plot_film_fit(label, film_data, model_fn, params, sub_fn, d_nm, out_path,
                  angles=None, wv_min=None, wv_max=None):
    if angles is None: angles = AOI_LIST
    if wv_min is None: wv_min = WV_MIN
    if wv_max is None: wv_max = WV_MAX

    fig, axes = plt.subplots(2, len(angles), figsize=(5 * len(angles), 8),
                              sharex=True)
    for col, ang in enumerate(angles):
        wv, psi_m, del_m = get_spectrum(film_data, ang)
        mask = (wv >= wv_min) & (wv <= wv_max)
        wv = wv[mask]; psi_m = psi_m[mask]; del_m = del_m[mask]
        eps_f = model_fn(wv, params)
        eps_s = sub_fn(wv, ang)
        psi_c, del_c = rho_to_psi_delta(fresnel_rho(eps_f, eps_s, d_nm, wv, ang))

        ax_p = axes[0, col]; ax_d = axes[1, col]
        ax_p.plot(wv, psi_m, 'k.', ms=1.5, label='Measured')
        ax_p.plot(wv, psi_c, 'r-', lw=1.5, label='Fit')
        ax_p.set_title(f'{ang}°'); ax_p.set_ylabel('Ψ (deg)'); ax_p.legend(fontsize=7)
        ax_d.plot(wv, del_m, 'k.', ms=1.5)
        ax_d.plot(wv, del_c, 'r-', lw=1.5)
        ax_d.set_ylabel('Δ (deg)'); ax_d.set_xlabel('Wavenumber (cm⁻¹)')

    fig.suptitle(f'{label}  –  Ψ/Δ fit ({d_nm} nm film)', fontweight='bold')
    plt.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  Plot: {os.path.basename(out_path)}")


def plot_permittivity(label, wv, eps_wvase, out_path, eps_raw=None):
    eps_std = np.conj(eps_wvase)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)
    for ax, attr, ylab in [(a1, np.real, 'ε₁ (real)'),
                           (a2, np.imag, 'ε₂ (imaginary)')]:
        if eps_raw is not None:
            er = np.conj(eps_raw)
            ax.plot(wv, attr(er), '.', ms=1.5, color='#bbbbbb',
                    label='Point-by-point', zorder=1)
        ax.plot(wv, attr(eps_std), '-', lw=2.0, color='royalblue',
                label='DL model' if eps_raw is not None else 'Extracted', zorder=2)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xlabel('Wavenumber (cm⁻¹)'); ax.set_ylabel(ylab)
        ax.legend(fontsize=8); ax.set_xlim(WV_MIN, WV_MAX)
    fig.suptitle(label, fontweight='bold')
    plt.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  Plot: {os.path.basename(out_path)}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run(data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Storage for substrate functions (needed by film samples)
    substrate_fns = {}   # label -> callable(wv, angle) -> eps (WVASE conv.)

    # Collect results for summary figure
    summary_entries = []  # (label, wv, eps_smooth, eps_raw_or_None)

    for cfg in SAMPLES:
        label     = cfg['label']
        model_key = cfg['model']
        bounds    = cfg['bounds']
        stype     = cfg['type']

        if model_key not in _MODEL_FNS:
            raise ValueError(f"Unknown model '{model_key}' for sample '{label}'. "
                             f"Available: {list(_MODEL_FNS.keys())}")
        model_fn = _MODEL_FNS[model_key]

        print(f"\n{'══'*30}")
        print(f"  {label.upper()}  ({stype})")
        print(f"{'══'*30}")

        dat_path = find_file(data_dir, cfg['suffix'])
        data = load_vase(dat_path)

        if stype == 'substrate':
            # ── Pseudo-dielectric inversion ───────────────────────────────────
            wv, psi, dlt = get_spectrum(data, PRIMARY_AOI)
            eps_raw = pseudo_eps(psi, dlt, PRIMARY_AOI)

            print("  Fitting DL model to pseudo-eps …")
            params, mse = fit_dl_to_pseudo(wv, eps_raw, model_fn, bounds)
            eps_dl = model_fn(wv, params)
            print(f"  MSE = {mse:.5f}")

            save_csv(wv, eps_raw,
                     os.path.join(out_dir, f'permittivity_{label}.csv'),
                     f'{label} – pseudo-dielectric (raw, point-by-point)')
            save_params_txt(os.path.join(out_dir, f'{label}_DL_params.txt'),
                            label, model_key, params)
            plot_substrate_fit(label, data, model_fn, params,
                               os.path.join(out_dir, f'fit_{label}.png'))
            plot_permittivity(label, wv, eps_dl,
                              os.path.join(out_dir, f'permittivity_{label}.png'),
                              eps_raw=eps_raw)

            # Store exact substrate function for downstream film samples
            substrate_fns[label] = make_substrate_fn(data)
            summary_entries.append((label, wv, eps_dl, eps_raw))

        elif stype == 'film':
            # ── 3-angle parametric DL fit ─────────────────────────────────────
            sub_label = cfg.get('substrate')
            d_nm      = cfg.get('thickness')
            if sub_label is None:
                raise ValueError(f"Film sample '{label}' must specify 'substrate'.")
            if d_nm is None:
                raise ValueError(f"Film sample '{label}' must specify 'thickness'.")
            if sub_label not in substrate_fns:
                raise ValueError(
                    f"Substrate '{sub_label}' not yet processed. "
                    "Make sure it appears before '{label}' in SAMPLES.")

            sub_fn = substrate_fns[sub_label]
            params, mse = fit_film_dl(data, sub_fn, d_nm, model_fn, bounds,
                                      maxiter=cfg.get('maxiter', 1500))

            wv_out = np.sort(get_spectrum(data, PRIMARY_AOI)[0])
            eps_out = model_fn(wv_out, params)

            save_csv(wv_out, eps_out,
                     os.path.join(out_dir, f'permittivity_{label}.csv'),
                     f'{label} {d_nm} nm – 3-angle DL parametric fit')
            save_params_txt(os.path.join(out_dir, f'{label}_DL_params.txt'),
                            f'{label} ({d_nm} nm)', model_key, params)
            plot_film_fit(f'{label} ({d_nm} nm)', data, model_fn, params,
                          sub_fn, d_nm,
                          os.path.join(out_dir, f'fit_{label}.png'))
            plot_permittivity(f'{label} ({d_nm} nm)', wv_out, eps_out,
                              os.path.join(out_dir, f'permittivity_{label}.png'))
            summary_entries.append((f'{label} ({d_nm} nm)', wv_out, eps_out, None))

        else:
            raise ValueError(f"Unknown sample type '{stype}' for '{label}'. "
                             "Use 'substrate' or 'film'.")

    # ── Summary figure ────────────────────────────────────────────────────────
    if summary_entries:
        n = len(summary_entries)
        ncols = min(n, 2); nrows = (n + 1) // 2
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(7 * ncols, 5 * nrows),
                                  squeeze=False)
        for idx, (lbl, wv, eps_sm, eps_rw) in enumerate(summary_entries):
            ax = axes[idx // ncols][idx % ncols]
            eps_sm_std = np.conj(eps_sm)
            if eps_rw is not None:
                eps_rw_std = np.conj(eps_rw)
                ax.plot(wv, eps_rw_std.real, '.', ms=1.2, color='#cccccc', zorder=1)
                ax.plot(wv, eps_rw_std.imag, '.', ms=1.2, color='#ffcccc', zorder=1)
            ax.plot(wv, eps_sm_std.real, '-', lw=2.0, color='royalblue', label='ε₁', zorder=2)
            ax.plot(wv, eps_sm_std.imag, '-', lw=2.0, color='tomato',    label='ε₂', zorder=2)
            ax.axhline(0, color='k', lw=0.6)
            ax.set_title(lbl, fontweight='bold', fontsize=12)
            ax.set_xlabel('Wavenumber (cm⁻¹)'); ax.set_ylabel('Permittivity')
            ax.legend(fontsize=9); ax.set_xlim(WV_MIN, WV_MAX)
        # Hide any unused subplots
        for idx in range(n, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)
        fig.suptitle('Extracted IR Permittivities', fontsize=14, fontweight='bold')
        plt.tight_layout()
        sp = os.path.join(out_dir, 'permittivity_summary.png')
        fig.savefig(sp, dpi=180); plt.close(fig)
        print(f"\n  Summary figure: {os.path.basename(sp)}")

    print("\n══ All done ══")


if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    out_dir  = sys.argv[2] if len(sys.argv) > 2 else './permittivity_output'
    run(data_dir, out_dir)
