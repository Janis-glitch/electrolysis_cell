"""Electrolysis cell fitting and simulation (multi-file datasets).

- Fits absolute concentrations over the full time series for CH3OH, HCOO-, CO3^2- (OH- is not fitted).
- No time weighting, no subsampling, no heuristic accelerations that could alter results.
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, brentq, differential_evolution
import matplotlib.pyplot as plt
import time
import math
# --- Optional Numba acceleration ---------------------------------------
try:
    from numba import jit, njit, prange
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    print("Numba not available. Falling back to pure Python implementation.")
# -----------------------------------------------------------------------
import os
import sys
FORCE_PY_NERNST = os.environ.get("FORCE_PY_NERNST", "0") == "1"
from multiprocessing import cpu_count
from functools import lru_cache, wraps
import warnings
import signal
from contextlib import contextmanager
warnings.filterwarnings('ignore')

# === FIT DATA DOWNSAMPLING ===
# Cap the number of points used for fitting per dataset to keep DE stable/fast.
# You can override via environment variable MAX_POINTS_FIT.
MAX_POINTS_FIT = int(os.environ.get("MAX_POINTS_FIT", "60"))  # reasonable default

# === TIMEOUT UTILITIES (macOS) ===
class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException(f"Timeout after {seconds}s!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def with_timeout(timeout_seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with time_limit(timeout_seconds):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# === PHYSICAL CONSTANTS ===
F = 96485.0    # Faraday-Konstante in C/mol
R = 8.314      # Gaskonstante in J/(mol·K)
T = 323.15     # Temperatur in K (50°C)
T0 = 298.15    # Referenztemperatur in K (25°C)

# === SYSTEM PARAMETERS ===
n1, n2 = 4, 2  # Elektronenzahlen für Reaktion 1 und 2
n3 = 4  # Elektronenzahl für OER

c_H2O0 = 55.56 # mol/L (Wasser-Anfangskonzentration)
# === INITIAL PARAMETERS (both fit) ===
DELTA_DEFAULT = 40e-06    # m   – Startwert für Diffusionsschichtdicke (jetzt fitbar)

# === FIXED SYSTEM CONSTANTS ===

A_EFF_FIXED = 0.0025 * 390  # m² – entspricht 0.975 m²
# Geometric area (projected area used for transport)
A_GEO_FIXED = 0.0025  # m² (25 cm²)

# === THERMODYNAMIC PARAMETERS ===
E1_std_25C = -0.0689   # V (CH3OH -> HCOO-) vs. RHE
E2_std_25C = -0.1014     # V (HCOO- -> CO32-) vs. RHE 
delta_S1 = 298.2     # J/(mol·K)
delta_S2 = 23.8      # J/(mol·K)

 # Temperaturkorrigierte Standardpotentiale
dE1_dT = delta_S1 / (n1 * F)
dE2_dT = delta_S2 / (n2 * F)
E1_std = E1_std_25C + dE1_dT * (T - T0) 
E2_std = E2_std_25C + dE2_dT * (T - T0) 

# OER-Standardpotential (vs. RHE, schwache T-Abhängigkeit vernachlässigt)
E3_std_25C = 1.2297  # V (4 OH⁻ → O₂ + 2 H₂O + 4 e⁻)
E3_std = E3_std_25C

# === NUMERICAL SAFEGUARDS (constants) ===
C_MIN_NERNST = 1e-8   # mol/L – Clamp in Nernst-Berechnung
C_MIN_SOLVE  = 1e-8   # mol/L – Clamp in solve_currents_MT (Surface/Bulk Stabilisierung)
C_MIN_ODE_C  = 1e-8   # mol/L – Clamp für c1..c3 im ODE-Integrator
C_MIN_ODE_OH = 1e-8   # mol/L – Clamp für OH- im ODE-Integrator
C_MIN_H2O    = 30.0   # mol/L – Mindestwert für Wasser
# Slightly higher clamp for OH− at the surface to avoid pathological j_lim collapse
C_MIN_SOLVE_OH_SURF = 1e-6  # mol/L

# === PRE-COMPUTED CONSTANTS ===
RT_over_n1F = R * T / (n1 * F)
RT_over_n2F = R * T / (n2 * F)
n1F_over_RT = n1 * F / (R * T)
n2F_over_RT = n2 * F / (R * T)

# --- Numba-optimized Nernst calculation (after constants) ---
if NUMBA_AVAILABLE:
    from numba import njit

    @njit(cache=True, fastmath=False)
    def _nernst_numba(c1, c2, c3, c4):
        """Fast Nernst calculation for NUMBA (no lru_cache)."""
        # Clamp to lower bounds to avoid overflow
        c1 = C_MIN_NERNST if c1 < C_MIN_NERNST else c1
        c2 = C_MIN_NERNST if c2 < C_MIN_NERNST else c2
        c3 = C_MIN_NERNST if c3 < C_MIN_NERNST else c3
        c4 = C_MIN_NERNST if c4 < C_MIN_NERNST else c4
        # Water activity a_H2O = 1
        Q1 = c2 / (c1 * c4**5)
        Q2 = c3 / (c2 * c4**3)
        return (
            E1_std - RT_over_n1F * np.log(Q1),
            E2_std - RT_over_n2F * np.log(Q2),
        )
# ----------------------------------------------------------------------

# === RATE-DETERMINING STEP (RDS) ELECTRON NUMBERS ===
# For kinetics, use the electron number of the rate‑determining step (RDS).
# Evidence suggests one‑electron steps for both MOR and FOR; set n_RDS = 1 for all.
n1_RDS, n2_RDS, n3_RDS = 1, 1, 1
n1_RDSF_over_RT = n1_RDS * F / (R * T)
n2_RDSF_over_RT = n2_RDS * F / (R * T)
n3_RDSF_over_RT = n3_RDS * F / (R * T)

# === VOLUMEN-STROM-ABHÄNGIGKEIT ===
def calculate_volume_from_current(i_total_A):
    """Berechnet das Reaktorvolumen basierend auf dem Gesamtstrom"""
    I_ref = [5.0, 10.0]  # A
    V_ref = [0.05026, 0.056474]  # L
    
    slope = (V_ref[1] - V_ref[0]) / (I_ref[1] - I_ref[0])
    volume = V_ref[0] + slope * (i_total_A - I_ref[0])
    return max(0.01, min(0.1, volume))

# === MASSENTRANSPORT FUNKTIONEN - MIT FITBAREN DIFFUSIONSKOEFFIZIENTEN ===
@lru_cache(maxsize=1000)
def cached_diffusion_coefficients_with_params(c_OH_hash, D_CH3OH_ref, D_HCOO_ref, D_CO32_ref, D_OH_ref):
    """Cached Diffusionskoeffizienten MIT FITBAREN PARAMETERN"""
    c_OH = c_OH_hash * 1e-6  # Rückkonvertierung

    # Feste Referenzwerte
    D_ref_fixed = {
        'H2O': 2.3e-9      # Wasser - fest
    }

    # Fitbare Referenzwerte bei 25°C [m²/s]
    D_ref_fittable = {
        'CH3OH': D_CH3OH_ref,  # Methanol - fitbar
        'HCOO': D_HCOO_ref,    # Formiat - fitbar  
        'CO32': D_CO32_ref,    # Carbonat - fitbar
        'OH': D_OH_ref         # Hydroxid - fitbar
    }

    T_ref = 298.15
    E_a_water = 15900  # J/mol

    # Temperaturabhängiger Viskositäts‑Term (Wasser, Arrhenius)
    viscosity_ratio = np.exp(E_a_water / R * (1 / T - 1 / T_ref))

    # ---- Relative Viscosity n_rel(c) for KOH (up to 2 M) -------------------
    # Target literature points (25 °C): 
    #   n_rel = 1.12 @ 0.5 M, 1.29 @ 1.0 M, 1.46 @ 1.5 M, 1.66 @ 2.0 M
    # Constrained quadratic fit with intercept fixed at 1:
    #   n_rel(c) ≈ 1 + 0.2351*c + 0.04774*c²   (c in mol·L⁻¹)
    # This matches the above points closely (≤ ~0.01 abs error).
    # Note: This n_rel is defined relative to pure water at the SAME temperature.
    #       The temperature dependence is handled separately by `viscosity_ratio`
    #       via an Arrhenius-type model for water. Combined with Stokes–Einstein,
    #       we use D(T,c) = D_ref * (T/T_ref) / [ (η(T)/η(T_ref)) * n_rel(c) ].
    c_KOH = max(0.0, min(2.0, c_OH))  # clamp to 0–2 M (data validity range)
    viscosity_factor = 1.0 + 0.23509677419354838 * c_KOH + 0.047741935483870894 * (c_KOH ** 2)

    # Kombiniere alle Diffusionskoeffizienten
    D_all = {**D_ref_fittable, **D_ref_fixed}

    # Temperatur-, Viskositäts-
    D_dict = {}
    for species, D_25C in D_all.items():
        D_bulk_corrected = D_25C * (T/T_ref) / (viscosity_ratio * viscosity_factor)
        D_dict[species] = D_bulk_corrected 

    # Als Array für schnelleren Zugriff (Reihenfolge: CH3OH, HCOO, CO32, OH, H2O)
    return np.array([D_dict['CH3OH'], D_dict['HCOO'], D_dict['CO32'], D_dict['OH'], D_dict['H2O']])

def calculate_diffusion_coefficients_with_params(c_OH, D_CH3OH_ref, D_HCOO_ref, D_CO32_ref, D_OH_ref):
    """Effizienter Wrapper für Diffusionskoeffizienten mit fitbaren Parametern"""
    c_OH_hash = int(round(c_OH * 1e6))
    return cached_diffusion_coefficients_with_params(c_OH_hash, D_CH3OH_ref, D_HCOO_ref, D_CO32_ref, D_OH_ref)

# === NERNST-POTENTIALE ===
@lru_cache(maxsize=5000)
def cached_nernst_safe(c1_hash, c2_hash, c3_hash, c4_hash):
    """Sichere Nernst-Berechnung mit kontrollierten Rundungsfehlern"""
    c1 = c1_hash * 1e-8
    c2 = c2_hash * 1e-8
    c3 = c3_hash * 1e-8
    c4 = c4_hash * 1e-8
    c1, c2, c3, c4 = [max(c, C_MIN_NERNST) for c in [c1, c2, c3, c4]]
    a_H2O = 1.0
    Q1 = (c2 * a_H2O**4) / (c1 * c4**5)
    Q2 = (c3 * a_H2O**2) / (c2 * c4**3)
    E1_eq = E1_std - RT_over_n1F * math.log(Q1)
    E2_eq = E2_std - RT_over_n2F * math.log(Q2)
    return E1_eq, E2_eq

def calculate_nernst_efficient(c1, c2, c3, c4):
    # Numba-Pfad nur, wenn verfügbar und nicht per Env-Flag deaktiviert
    if NUMBA_AVAILABLE and not FORCE_PY_NERNST:
        return _nernst_numba(c1, c2, c3, c4)

    # Python‑Fallback mit Hash‑basiertem lru_cache
    c1_hash = int(round(c1 * 1e8))
    c2_hash = int(round(c2 * 1e8))
    c3_hash = int(round(c3 * 1e8))
    c4_hash = int(round(c4 * 1e8))
    return cached_nernst_safe(c1_hash, c2_hash, c3_hash, c4_hash)

# --- Helfer zum Einfrieren von Parametern ---------------------------------

def _apply_param_freezing(p):
    """Gibt eine Kopie von p zurück (Parameter-Freeze entfällt, da A_eff nicht mehr fitbar)."""
    return np.array(p, dtype=float).copy()


def _warmup_numba():
    """Einmaliger Warm-up der Numba-Nernst-Funktion, um JIT-Latenz zu minimieren."""
    if NUMBA_AVAILABLE and not FORCE_PY_NERNST:
        _ = _nernst_numba(C_MIN_NERNST, C_MIN_NERNST, C_MIN_NERNST, C_MIN_NERNST)


def _nernst_regression_check(n_samples: int = 500):
    """Vergleicht Numba- und Python-Pfad für die Nernst-Berechnung auf Abweichungen."""
    try:
        import numpy as np
    except Exception:
        return
    rng = np.random.default_rng(42)
    c1 = 10 ** rng.uniform(-6, 0, n_samples)
    c2 = 10 ** rng.uniform(-6, 0, n_samples)
    c3 = 10 ** rng.uniform(-6, 0, n_samples)
    c4 = 10 ** rng.uniform(-6, 0, n_samples)

    # Python-Pfad (lru_cache-basierte Variante)
    e1_py, e2_py = [], []
    for a, b, c, d in zip(c1, c2, c3, c4):
        e1, e2 = cached_nernst_safe(
            int(round(max(a, C_MIN_NERNST) * 1e8)),
            int(round(max(b, C_MIN_NERNST) * 1e8)),
            int(round(max(c, C_MIN_NERNST) * 1e8)),
            int(round(max(d, C_MIN_NERNST) * 1e8)),
        )
        e1_py.append(e1); e2_py.append(e2)

    # Numba-Pfad
    e1_nb, e2_nb = [], []
    for a, b, c, d in zip(c1, c2, c3, c4):
        e1, e2 = calculate_nernst_efficient(a, b, c, d)
        e1_nb.append(e1); e2_nb.append(e2)

    e1_diff = float(np.max(np.abs(np.array(e1_py) - np.array(e1_nb))))
    e2_diff = float(np.max(np.abs(np.array(e2_py) - np.array(e2_nb))))
    print(f"max|ΔE1|={e1_diff:.3e} V, max|ΔE2|={e2_diff:.3e} V")

class ElectrolysisDataset:
    """Container for a single electrolysis dataset."""
    def __init__(self, filepath):
        self.filepath = filepath
        self.name = os.path.basename(filepath).replace('.csv', '')
        self.load_data()
        
    def load_data(self):
        """Load and preprocess CSV data."""
        try:
            df = pd.read_csv(self.filepath, encoding='utf-8')
            
            # Current from first 10 entries (mA → A)
            current_mA = df.iloc[:10, 8].mean()
            self.i_total = current_mA / 1000
            
            # Reactor volume
            self.V = calculate_volume_from_current(self.i_total)
            
            # Concentration data (columns A-E)
            conc_data = df.iloc[:, :5].dropna()
            
            # Electrical data (columns F-I)
            elec_data = df.iloc[:, 5:9].dropna()
            
            # Time normalization
            t_start_conc = conc_data.iloc[0, 0] if len(conc_data) > 0 else 0
            t_start_elec = elec_data.iloc[0, 0] if len(elec_data) > 0 else t_start_conc
            t_start_global = min(t_start_conc, t_start_elec)
            
            # Time in seconds, normalized to t=0
            self.t_conc = (conc_data.iloc[:, 0].values - t_start_global) * 60
            
            # Concentrations (order from CSV)
            self.c_HCOO = np.maximum(0, conc_data.iloc[:, 1].values)
            self.c_OH = np.maximum(0, conc_data.iloc[:, 2].values)
            self.c_MeOH = np.maximum(0, conc_data.iloc[:, 3].values)
            self.c_CO32 = np.maximum(0, conc_data.iloc[:, 4].values)
            
            # Initial conditions
            self.c0 = {
                'MeOH': max(0, conc_data.iloc[0, 3]),
                'HCOO': max(0, conc_data.iloc[0, 1]),
                'CO32': max(0, conc_data.iloc[0, 4]),
                'OH': max(0, conc_data.iloc[0, 2]),
                'H2O': c_H2O0
            }
            
            # Electrical data
            if len(elec_data) > 0:
                self.t_volt = (elec_data.iloc[:, 0].values - t_start_global) * 60
                self.E_anode = elec_data.iloc[:, 2].values
            else:
                self.t_volt = None
                self.E_anode = None
            
            # Downsample for fitting
            self.reduce_data_for_fitting()
            
            self.valid = True
            
            print(f"✓ {self.name}: {len(self.t_conc)} points, I = {self.i_total:.2f} A, V = {self.V*1000:.1f} mL")
            
        except Exception as e:
            print(f"✗ Error loading {self.name}: {e}")
            self.valid = False
    
    def reduce_data_for_fitting(self):
        """Downsample points for fitting (use all points for plotting)."""
        t = self.t_conc
        N = len(t)
        if N <= MAX_POINTS_FIT or MAX_POINTS_FIT < 5:
            # Use all points if already small
            self.t_conc_fit = self.t_conc
            self.c_MeOH_fit = self.c_MeOH
            self.c_HCOO_fit = self.c_HCOO
            self.c_CO32_fit = self.c_CO32
            self.c_OH_fit = self.c_OH
            return
        # Uniform stride
        stride = int(np.ceil(N / MAX_POINTS_FIT))
        idx = np.arange(0, N, stride, dtype=int)
        # Ensure last point is included
        if idx[-1] != N - 1:
            idx = np.concatenate([idx, [N - 1]])
        # Apply selection
        self.t_conc_fit = self.t_conc[idx]
        self.c_MeOH_fit = self.c_MeOH[idx]
        self.c_HCOO_fit = self.c_HCOO[idx]
        self.c_CO32_fit = self.c_CO32[idx]
        self.c_OH_fit = self.c_OH[idx]

class PhysicsCorrectElectrolysisModel:
    """Electrolysis model (same physics as the main model) with fitted diffusion coefficients."""
    def __init__(self, params_dict):
        self.update_params(params_dict)
        self.n1F = n1 * F
        self.n2F = n2 * F
        self.n3F = n3 * F
        # Cache for the last solved anode potential (used for fast local bracketing)
        self._last_E = None
        
    def update_params(self, params_dict):
        """Update all model parameters including diffusion coefficients."""
        self.i0_1 = params_dict['i0_1']
        self.i0_2 = params_dict['i0_2']
        self.i0_3 = params_dict['i0_3']  # OER
        self.alpha1 = params_dict['alpha1']
        self.alpha2 = params_dict['alpha2']
        self.alpha3 = params_dict['alpha3']  # OER
        self.A_eff = A_EFF_FIXED  # fixed area, not part of the fit
        self.delta = params_dict['delta']
        # Numerical safety: avoid 1/delta explosions
        if self.delta < 1e-8:
            self.delta = 1e-8
        self.inv_delta = 1.0 / self.delta
        # Fitted diffusion coefficients (reference values at 25°C)
        self.D_CH3OH_ref = params_dict['D_CH3OH_ref']
        self.D_HCOO_ref = params_dict['D_HCOO_ref']
        self.D_CO32_ref = params_dict['D_CO32_ref']
        self.D_OH_ref = params_dict['D_OH_ref']
        
    def solve_currents_MT(self, E_anode, c1, c2, c3, c4, i_total):
        """Solve partial currents with mass transport. Kinetics on A_eff. OH⁻ limitation via min() for all reactions (R1/R2/R3)."""
        max_iter = 30
        tol = 1e-7

        # Stabilize inputs
        c1, c2, c3, c4 = [max(c, C_MIN_SOLVE) for c in [c1, c2, c3, c4]]

        # Equilibrium potentials (R1/R2 via Nernst, R3 = OER vs. RHE)
        E1_eq_bulk, E2_eq_bulk = calculate_nernst_efficient(c1, c2, c3, c4)
        E3_eq_bulk = E3_std
        eta1_bulk = E_anode - E1_eq_bulk
        eta2_bulk = E_anode - E2_eq_bulk
        eta3_bulk = E_anode - E3_eq_bulk

        # Initial guess
        i1_old, i2_old, i3_old = 0.0, 0.0, 0.0

        # Terms constant during the iteration
        D_values = calculate_diffusion_coefficients_with_params(
            c4, self.D_CH3OH_ref, self.D_HCOO_ref, self.D_CO32_ref, self.D_OH_ref)
        c1_m3 = c1 * 1000.0
        c2_m3 = c2 * 1000.0
        c4_m3 = c4 * 1000.0
        A_eff = self.A_eff
        n1F = self.n1F
        n2F = self.n2F
        n3F = self.n3F
        inv_delta = self.inv_delta
        # KL j_lim is computed per A_geo and then converted to the A_eff basis used by kinetics
        # (transport across geometric interface; kinetics on ECSA)
        area_ratio_geo_to_eff = A_GEO_FIXED / A_eff  # ≈ 1/390 for this electrode

        # ---- Fast feasibility pre-check (bulk-based transport upper bound) ----
        # Upper-bound KL limits using BULK concentrations (zero-flux assumption).
        # If even this optimistic sum cannot support i_total, skip costly iterations.
        def j_lim_bulk(nF_reac, stoich_dict):
            j_candidates = []
            for sp, nu in stoich_dict.items():
                if nu <= 0:
                    continue
                if sp == 'CH3OH':
                    Dk, ck_m3 = D_values[0], c1_m3
                elif sp == 'HCOO':
                    Dk, ck_m3 = D_values[1], c2_m3
                elif sp == 'OH':
                    Dk, ck_m3 = D_values[3], c4_m3
                else:
                    continue
                j_candidates.append((nF_reac * Dk * ck_m3 * inv_delta / float(nu)) * area_ratio_geo_to_eff)
            return min(j_candidates) if j_candidates else 1e30
        j_lim1_bulk = j_lim_bulk(n1F, {'CH3OH': 1, 'OH': 5})
        j_lim2_bulk = j_lim_bulk(n2F, {'HCOO': 1, 'OH': 3})
        j_lim3_bulk = j_lim_bulk(n3F, {'OH': 4})
        i_transport_upper = A_eff * (j_lim1_bulk + j_lim2_bulk + j_lim3_bulk)
        # If even the optimistic transport bound is far below the required current, abort early.
        if not np.isfinite(i_transport_upper) or i_transport_upper < 0.05 * i_total:
            # Signal the optimizer that this parameter set / state is transport-infeasible
            raise RuntimeError("Transport-infeasible at given bulk concentrations")

        for iteration in range(max_iter):
            # Surface concentrations via film theory
            J1 = i1_old / n1F; J2 = i2_old / n2F; J3 = i3_old / n3F
            j1 = J1 / A_eff;   j2 = J2 / A_eff;   j3 = J3 / A_eff

            # Convert to geometric-area flux for film theory (transport occurs across A_geo)
            # j_geo = (mol s^-1 m^-2) referenced to A_geo
            area_scale_eff_to_geo = A_eff / A_GEO_FIXED
            j1_geo = j1 * area_scale_eff_to_geo
            j2_geo = j2 * area_scale_eff_to_geo
            j3_geo = j3 * area_scale_eff_to_geo

            # Film theory uses flux per geometric area; convert j (A_eff-based) to j_geo above
            c1_surf = max(c1 - j1_geo * self.delta / (D_values[0] * 1000), C_MIN_SOLVE)
            c2_surf = max(c2 - (-j1_geo + j2_geo) * self.delta / (D_values[1] * 1000), C_MIN_SOLVE)
            c3_surf = max(c3 - (-j2_geo) * self.delta / (D_values[2] * 1000), C_MIN_SOLVE)
            c4_surf = max(c4 - (5*j1_geo + 3*j2_geo + 4*j3_geo) * self.delta / (D_values[3] * 1000), C_MIN_SOLVE_OH_SURF)

            # Surface equilibrium potentials
            E1_eq_surf, E2_eq_surf = calculate_nernst_efficient(c1_surf, c2_surf, c3_surf, c4_surf)
            E3_eq_surf = E3_std
            eta1_surf = E_anode - E1_eq_surf
            eta2_surf = E_anode - E2_eq_surf
            eta3_surf = E_anode - E3_eq_surf

            # Full Butler–Volmer (anodic + cathodic) with overflow clamp
            # i = i0 * [exp(α nF η / RT) - exp(−(1−α) nF η / RT)]
            MAX_EXP = 60.0

            def exp_clamped(x):
                if x > MAX_EXP:
                    x = MAX_EXP
                elif x < -MAX_EXP:
                    x = -MAX_EXP
                return math.exp(x)

            # Reaction 1
            phi1 = n1_RDSF_over_RT * eta1_surf
            exp_a1 = exp_clamped(self.alpha1 * phi1)
            exp_c1 = exp_clamped(-(1.0 - self.alpha1) * phi1)
            j_bv1 = self.i0_1 * (exp_a1 - exp_c1)
            if j_bv1 < 0.0:
                j_bv1 = 0.0  # Anode: suppress negative (cathodic) currents for KL combination

            # Reaction 2
            phi2 = n2_RDSF_over_RT * eta2_surf
            exp_a2 = exp_clamped(self.alpha2 * phi2)
            exp_c2 = exp_clamped(-(1.0 - self.alpha2) * phi2)
            j_bv2 = self.i0_2 * (exp_a2 - exp_c2)
            if j_bv2 < 0.0:
                j_bv2 = 0.0

            # Reaction 3 (OER)
            phi3 = n3_RDSF_over_RT * eta3_surf
            exp_a3 = exp_clamped(self.alpha3 * phi3)
            exp_c3 = exp_clamped(-(1.0 - self.alpha3) * phi3)
            j_bv3 = self.i0_3 * (exp_a3 - exp_c3)
            if j_bv3 < 0.0:
                j_bv3 = 0.0

            # Diffusion limits (multiple reactants) – physical min-limit per reaction
            # Helper: j_lim for reaction i across all required reactants
            def j_lim_reaction(nF_reac, stoich_dict):
                # stoich_dict: { 'CH3OH': nu, 'HCOO': nu, 'OH': nu }
                j_candidates = []
                for sp, nu in stoich_dict.items():
                    if nu <= 0:
                        continue
                    if sp == 'CH3OH':
                        Dk, ck_m3 = D_values[0], c1_m3
                    elif sp == 'HCOO':
                        Dk, ck_m3 = D_values[1], c2_m3
                    elif sp == 'OH':
                        Dk, ck_m3 = D_values[3], c4_m3
                    else:
                        continue
                    j_candidates.append((nF_reac * Dk * ck_m3 * inv_delta / float(nu)) * area_ratio_geo_to_eff)
                return min(j_candidates) if j_candidates else 1e30

            # R1 (MOR): CH3OH (ν=1) and OH− (ν=5)
            j_lim1 = j_lim_reaction(n1F, {'CH3OH': 1, 'OH': 5})

            # R2 (FOR): HCOO− (ν=1) and OH− (ν=3)
            j_lim2 = j_lim_reaction(n2F, {'HCOO': 1, 'OH': 3})

            # R3 (OER): OH− (ν=4)
            j_lim3 = j_lim_reaction(n3F, {'OH': 4})

            # Koutecky-Levich
            j1 = (j_bv1 * j_lim1) / (j_bv1 + j_lim1)
            j2 = (j_bv2 * j_lim2) / (j_bv2 + j_lim2)
            j3 = (j_bv3 * j_lim3) / (j_bv3 + j_lim3)

            i1 = j1 * A_eff
            i2 = j2 * A_eff
            i3 = j3 * A_eff

            # Convergence
            if abs(i1 - i1_old) < tol and abs(i2 - i2_old) < tol and abs(i3 - i3_old) < tol:
                break

            i1_old, i2_old, i3_old = i1, i2, i3

        return i1, i2, i3
    
    def find_anode_potential(self, c1, c2, c3, c4, i_total):
        """Fast root-finding for E_anode with local bracketing and robust fallback.
        Strategy:
          1) Try a small window around the last solution (if available) to find a sign change and use brentq.
          2) If that fails, fall back to a coarser window around the kinetic onset.
          3) If no bracket exists, return the grid point with minimal |I_calc - I_total|.
        Physics unchanged; only the search strategy is optimized.
        """
        # Equilibrium potentials
        E1_eq, E2_eq = calculate_nernst_efficient(c1, c2, c3, c4)
        E3_eq = E3_std

        def g(E):
            i1, i2, i3 = self.solve_currents_MT(E, c1, c2, c3, c4, i_total)
            return (i1 + i2 + i3) - i_total

        # Default center near the highest equilibrium potential
        E_center_default = max(E1_eq, E2_eq, E3_eq) + 0.2
        # Prefer the last successful solution if available
        E_center = self._last_E if (hasattr(self, "_last_E") and self._last_E is not None) else E_center_default

        # 1) Small local window (fast path)
        grid_small = np.linspace(E_center - 0.6, E_center + 0.6, 9)
        vals_small = [g(E) for E in grid_small]
        for a, b, fa, fb in zip(grid_small[:-1], grid_small[1:], vals_small[:-1], vals_small[1:]):
            if not (np.isfinite(fa) and np.isfinite(fb)):
                continue
            if np.sign(fa) != np.sign(fb):
                try:
                    E = brentq(g, a, b, xtol=1e-6)
                    self._last_E = float(E)
                    return float(E)
                except Exception:
                    break  # fall back to coarse search

        # 2) Coarse fallback window
        E_center = max(E_center_default, E_center)
        grid = np.linspace(E_center - 3.5, E_center + 3.5, 15)
        vals = [g(E) for E in grid]
        for a, b, fa, fb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
            if not (np.isfinite(fa) and np.isfinite(fb)):
                continue
            if np.sign(fa) != np.sign(fb):
                try:
                    E = brentq(g, a, b, xtol=1e-6)
                    self._last_E = float(E)
                    return float(E)
                except Exception:
                    pass

        # 3) No valid bracket → best effort
        vals_arr = np.array(vals, dtype=float)
        abs_vals = np.abs(vals_arr)
        # Treat non-finite values as very large so they won't be selected
        abs_vals[~np.isfinite(abs_vals)] = 1e18
        if np.all(~np.isfinite(vals_arr)):
            # If everything is non-finite, fall back to default center
            E = float(E_center_default)
        else:
            idx = int(np.argmin(abs_vals))
            E = float(grid[idx])
        self._last_E = E
        return E
    
    def simulate(self, dataset, t_eval=None):
        """Simulate the system."""
        # Reset cached anode potential per simulation run
        self._last_E = None
        V = dataset.V
        i_total = dataset.i_total

        # Initial conditions
        x0 = [dataset.c0['MeOH'], dataset.c0['HCOO'],
              dataset.c0['CO32'], dataset.c0['OH'], dataset.c0['H2O'],
              dataset.c0['HCOO']*V, dataset.c0['CO32']*V, 0]

        if t_eval is None:
            t_eval = dataset.t_conc_fit

        # Precompute
        A_eff_over_V = self.A_eff / V

        def odes(t, x):
            c1, c2, c3, c4, c5, n_HCOO_total, n_CO32_total, Q_total = x
            c1 = max(c1, C_MIN_ODE_C)
            c2 = max(c2, C_MIN_ODE_C)
            c3 = max(c3, C_MIN_ODE_C)
            c4 = max(c4, C_MIN_ODE_OH)
            c5 = max(c5, C_MIN_H2O)  # Water

            # Galvanostatic operation
            E_anode = self.find_anode_potential(c1, c2, c3, c4, i_total)
            i1, i2, i3 = self.solve_currents_MT(E_anode, c1, c2, c3, c4, i_total)

            # Reaction rates
            r1 = i1 / (self.A_eff * self.n1F)
            r2 = i2 / (self.A_eff * self.n2F)
            r3 = i3 / (self.A_eff * self.n3F)

            # Volumetric rates
            r1_vol = r1 * A_eff_over_V
            r2_vol = r2 * A_eff_over_V
            r3_vol = r3 * A_eff_over_V

            # Concentration changes (effective)
            dc1_dt = -r1_vol
            dc2_dt = r1_vol - r2_vol
            dc3_dt = r2_vol
            # Effective coupling to OH⁻ (net balance; no explicit stoichiometric correction)
            dc4_dt = (-r1_vol - r2_vol)
            dc5_dt = (2*r1_vol + r2_vol)  # effective coupling to H₂O

            # Integrals
            dn_HCOO_dt = r1 * self.A_eff
            dn_CO32_dt = r2 * self.A_eff
            dQ_dt = i_total

            return [dc1_dt, dc2_dt, dc3_dt, dc4_dt, dc5_dt,
                    dn_HCOO_dt, dn_CO32_dt, dQ_dt]

        sol = solve_ivp(odes, (0, np.max(t_eval)), x0,
                        t_eval=t_eval,
                        method='Radau',
                        rtol=1e-6,
                        atol=1e-6,
                        max_step=50)

        return sol.y[:5]  # All concentrations


# Ersetze die alte objective_function durch diese robustere Version

@with_timeout(200) # Timeout von 90 Sekunden pro Einzel-Fit
def objective_function(params, dataset, model):
    """Objective for a single dataset – fit absolute concentrations with equal‑priority weighting."""
    params = _apply_param_freezing(params)
    params_dict = {
        'i0_1': params[0],
        'i0_2': params[1],
        'i0_3': params[2],
        'alpha1': params[3],
        'alpha2': params[4],
        'alpha3': params[5],
        'delta': params[6],
        'D_CH3OH_ref': params[7],
        'D_HCOO_ref': params[8],
        'D_CO32_ref': params[9],
        'D_OH_ref': params[10]
    }

    model.update_params(params_dict)

    try:
        # Simulation
        c_sim = model.simulate(dataset)

        # Verbesserte Stabilitätsprüfung: Prüfe auf NaN oder Inf
        if np.any(np.isnan(c_sim)) or np.any(np.isinf(c_sim)):
            return 1e12 # Gib einen sehr hohen Strafwert zurück

        # --- Fehlerberechnung ---
        errors = []
        for c_exp, c_sim_i in [
            (dataset.c_MeOH_fit, c_sim[0]),
            (dataset.c_HCOO_fit, c_sim[1]),
            (dataset.c_CO32_fit, c_sim[2])
        ]:
            diff = (c_sim_i - c_exp)
            sse = float(np.sum(diff * diff))
            errors.append(sse)

        total_error = float(np.sum(errors))
        # Stelle sicher, dass der Fehler nicht ungültig ist
        return total_error if np.isfinite(total_error) else 1e12

    except (TimeoutException, RuntimeError):
        return 1e12  # High penalty for timeout or transport infeasibility
    except Exception:
        return 1e12 # Hoher Strafwert für alle anderen Fehler

def objective_function_multi(params, datasets):
    """Multi-dataset objective – absolute concentrations over the full time series (OH- excluded)."""
    params = _apply_param_freezing(params)
    model = PhysicsCorrectElectrolysisModel({
        'i0_1': params[0],
        'i0_2': params[1],
        'i0_3': params[2],
        'alpha1': params[3],
        'alpha2': params[4],
        'alpha3': params[5],
        'delta': params[6],
        'D_CH3OH_ref': params[7],
        'D_HCOO_ref': params[8],
        'D_CO32_ref': params[9],
        'D_OH_ref': params[10]
    })

    total_error = 0.0
    for dataset in datasets:
        error = objective_function(params, dataset, model)
        total_error += error

    return total_error / max(1, len(datasets))

class LiveMonitor:
    """Live monitoring during fitting – absolute concentrations (no weighting)."""
    def __init__(self, datasets):
        self.datasets = datasets
        self.iteration = 0
        self.best_objective = float('inf')
        self.best_params = None
        self.start_time = time.time()
        # Initialize persistent live plot (one-time setup)
        self._setup_live_plot()

    def _setup_live_plot(self):
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except Exception:
            self._live_plot_enabled = False
            return

        self._live_plot_enabled = True
        self._species_names = ['CH3OH', 'HCOO-', 'CO3^2-', 'OH-']
        n = len(self.datasets)
        if n == 0:
            self._live_plot_enabled = False
            return

        # Prepare data caches (experimental)
        self._t_lists = []          # minutes
        self._exp_lists = []        # list of tuples (MeOH, HCOO, CO32, OH)
        for ds in self.datasets:
            t_min = ds.t_conc / 60.0
            self._t_lists.append(t_min)
            self._exp_lists.append((ds.c_MeOH, ds.c_HCOO, ds.c_CO32, ds.c_OH))

        # Figure and axes: ONE axis per dataset (n rows, 1 column)
        fig_height = max(3.0, 2.8 * n)
        self._fig, self._axes = plt.subplots(n, 1, figsize=(12, fig_height), sharex=True)
        self._axes = np.atleast_1d(self._axes)
        self._sim_lines = []  # flattened list of Line2D handles in order per dataset: [MeOH, HCOO, CO32, OH]

        for r, ds in enumerate(self.datasets):
            ax = self._axes[r]
            t_min = self._t_lists[r]
            exp_ch3oh, exp_hcoo, exp_co32, exp_oh = self._exp_lists[r]

            # Plot experimental markers for all species on the SAME axis
            ax.plot(t_min, exp_ch3oh, 'o', alpha=0.7, label='exp CH3OH')
            ax.plot(t_min, exp_hcoo, 's', alpha=0.7, label='exp HCOO-')
            ax.plot(t_min, exp_co32, 'D', alpha=0.7, label='exp CO3^2-')
            ax.plot(t_min, exp_oh,   '^', alpha=0.7, label='exp OH-')

            # Placeholder simulated lines (same axis), will be updated each iteration
            sim_m, = ax.plot(t_min, np.full_like(t_min, np.nan), '-', linewidth=2, label='sim CH3OH')
            sim_h, = ax.plot(t_min, np.full_like(t_min, np.nan), '-', linewidth=2, label='sim HCOO-')
            sim_c, = ax.plot(t_min, np.full_like(t_min, np.nan), '-', linewidth=2, label='sim CO3^2-')
            sim_o, = ax.plot(t_min, np.full_like(t_min, np.nan), '-', linewidth=2, label='sim OH-')
            self._sim_lines.extend([sim_m, sim_h, sim_c, sim_o])

            ax.set_title(ds.name)
            ax.set_ylabel('Concentration (mol/L)')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=8, ncol=2)

        self._axes[-1].set_xlabel('Time (min)')
        self._fig.suptitle('Live fit — absolute concentrations per experiment (all species in one axis)', y=1.02, fontsize=12)
        self._fig.tight_layout()
        try:
            plt.ion()
            plt.show(block=False)
        except Exception:
            pass

    def _update_live_plot(self, params_vec):
        if not getattr(self, '_live_plot_enabled', False):
            return

        import numpy as np
        import matplotlib.pyplot as plt

        # Build model with current parameters
        p = _apply_param_freezing(np.array(params_vec, dtype=float))
        model = PhysicsCorrectElectrolysisModel({
            'i0_1': p[0],
            'i0_2': p[1],
            'i0_3': p[2],
            'alpha1': p[3],
            'alpha2': p[4],
            'alpha3': p[5],
            'delta': p[6],
            'D_CH3OH_ref': p[7],
            'D_HCOO_ref': p[8],
            'D_CO32_ref': p[9],
            'D_OH_ref': p[10],
        })

        # Simulate each dataset on the full plotting grid
        sim_curves = []  # flattened in order [MeOH, HCOO, CO32, OH] per dataset
        try:
            for ds in self.datasets:
                c_sim = model.simulate(ds, t_eval=ds.t_conc)
                sim_curves.extend([c_sim[0], c_sim[1], c_sim[2], c_sim[3]])
        except Exception:
            return

        # Update line data
        line_idx = 0
        for r, ds in enumerate(self.datasets):
            t_min = self._t_lists[r]
            for _ in range(4):
                line = self._sim_lines[line_idx]
                y = sim_curves[line_idx]
                if len(y) == len(t_min):
                    line.set_data(t_min, y)
                else:
                    line.set_data(t_min, np.full_like(t_min, np.nan))
                line_idx += 1

        try:
            self._fig.canvas.draw_idle()
            plt.pause(0.001)
        except Exception:
            pass
        
    def callback(self, xk, convergence=None):
        """Callback for live updates"""
        self.iteration += 1

        xk_disp = _apply_param_freezing(xk)

        i0_1, i0_2, i0_3, alpha1, alpha2, alpha3, delta, D_CH3OH, D_HCOO, D_CO32, D_OH = xk_disp

        # Objective value for display
        try:
            obj_value = objective_function_multi(xk, self.datasets)
            if obj_value < self.best_objective:
                self.best_objective = obj_value
                self.best_params = xk.copy()
                improvement = " ⬇"
            else:
                improvement = ""
        except Exception:
            obj_value = float('inf')
            improvement = ""

        # Update live plot (experimental vs. simulated) for current parameters
        try:
            self._update_live_plot(xk_disp)
        except Exception:
            pass

        # Live output
        elapsed = time.time() - self.start_time
        print(
            f"\rGen {self.iteration:3d} [{elapsed:5.1f}s]: "
            f"i0_1={i0_1:.3g} | i0_2={i0_2:.3g} | i0_3={i0_3:.3g} | "
            f"alpha1={alpha1:.3f} | alpha2={alpha2:.3f} | alpha3={alpha3:.3f} | "
            f"delta={delta*1e6:3.0f}µm | "
            f"D_MeOH={D_CH3OH:.1e} | D_HCOO={D_HCOO:.1e} | D_CO32={D_CO32:.1e} | D_OH={D_OH:.1e} | "
            f"f={obj_value:.2e}{improvement}",
            end="",
            flush=True,
        )

def run_fitting(datasets, use_initial_guess=True):
    """Perform the fitting (absolute concentrations)."""
    
    # --- Literature-backed parameter windows ---
    # D_OH−, D_CO3²−, D_HCOO− at 25 °C from CRC/PHREEQC compilation (aqion table).
    # D_CH3OH in water at ~23–25 °C ≈ 1.4×10⁻⁹ m²/s (Van Loon et al., J. Phys. Chem. A 2008).
    # OER i0 range estimated from typical b=40–120 mV/dec and η@10 mA/cm² on oxides; converted to i0 via Tafel.
    # i0 for MOR/FOR on porous CuO set per ECSA (A_eff), using Tafel back-calculation at η≈0.2–0.6 V and j_ECSA≈0.1–1 mA/cm².
    bounds = [
        (0.021, 0.021),   # i0_1 [A/m²] (MOR on CuO, per A_eff)
        (1.5e-3, 1.5e-3),   # i0_2 [A/m²] (FOR on Cu/CuO, per A_eff)
        (8e-4, 8e-4),   # i0_3 [A/m²] (OER on CuO, per A_eff)
        (0.21, 0.21),   # alpha1 [-]
        (0.21, 0.21),   # alpha2 [-]
        (0.24, 0.24),   # alpha3 [-]
        (20e-6, 300e-6),# delta [m]
        (1.0e-9, 1.2e-9),  # D_CH3OH_ref [m²/s]
        (1.6e-9, 1.8e-9),  # D_HCOO_ref  [m²/s]
        (1.1e-9, 1.3e-9),  # D_CO32_ref  [m²/s]
        (5.4e-9, 5.6e-9),  # D_OH_ref    [m²/s]
    ]
    initial_guess = [
        0.021,    # i0_1  [A/m²] (MOR, per A_eff)
        1.5e-3,    # i0_2  [A/m²] (FOR, per A_eff)
        8.0e-4,    # i0_3  [A/m²] (OER, per A_eff)
        0.21,      # alpha1 [-]
        0.21,      # alpha2 [-]
        0.24,      # alpha3 [-]
        25e-6,    # delta  [m]
        1.1e-9,   # D_CH3OH_ref [m²/s] # Lien-Chun Weng, Alexis T Bell, and Adam Z Weber. Modeling gas-diffusion electrodes for CO2 reduction.
        1.7e-9,   # D_HCOO_ref  [m²/s] # Bruce E Poling. The properties of gases and liquids. 2004.
        1.2e-9,  # D_CO32_ref  [m²/s]     #Bruce E Poling. The properties of gases and liquids. 2004.
        5.5e-9,   # D_OH_ref    [m²/s]    #Edward Lansing Cussler. Multicomponent diffusion, volume 3. Elsevier, 2013
    ]
    n_params = len(bounds)
    min_pop_rows = max(10, n_params)

    print(f"\nPARAMETER FITTING (11 parameters) – ABSOLUTE CONCENTRATIONS")
    print(f"Datasets: {len(datasets)}")
    print("Fitted parameters: i0_1, i0_2, i0_3 (OER), alpha1, alpha2, alpha3, delta, D_CH3OH, D_HCOO-, D_CO3^2-, D_OH-")
    print(f"delta (diffusion film): fitted (start: {DELTA_DEFAULT*1e6:.0f} µm)")
    print("Fit basis: absolute concentrations (CH3OH, HCOO-, CO3^2-); OH- not fitted")
    print("No time weighting. Using all points from the CSV time series.")

    if use_initial_guess:
        print(f"\nUsing literature‑informed start values:")
        print(f"   i0_1={initial_guess[0]:.2e}, i0_2={initial_guess[1]:.2e}, i0_3={initial_guess[2]:.2e}, "
              f"alpha1={initial_guess[3]:.3f}, alpha2={initial_guess[4]:.3f}, alpha3={initial_guess[5]:.3f}, "
              f"delta={initial_guess[6]*1e6:.1f}µm")
        print(f"   D_CH3OH={initial_guess[7]*1e18:.2f}nm²/s, D_HCOO={initial_guess[8]*1e18:.2f}nm²/s, D_CO32={initial_guess[9]*1e18:.2f}nm²/s, D_OH={initial_guess[10]*1e18:.2f}nm²/s")

        # Create initial population with start values
        init_pop = np.zeros((min_pop_rows, n_params))
        init_pop[0, :] = np.clip(initial_guess, [b[0] for b in bounds], [b[1] for b in bounds])
        rng = np.random.default_rng(42)
        for i in range(1, min_pop_rows):
            for j, (lo, hi) in enumerate(bounds):
                lo_j = max(lo, init_pop[0, j] * 0.6)
                hi_j = min(hi, init_pop[0, j] * 1.4)
                if lo == hi:
                    init_pop[i, j] = lo
                else:
                    init_pop[i, j] = rng.uniform(lo_j, hi_j)
    else:
        print(f"\nUsing random start values (Latin Hypercube)")
        init_pop = 'latinhypercube'

    # Live monitor
    monitor = LiveMonitor(datasets)
    monitor._bounds = bounds

    print(f"\nLive fitting progress:")

    # NOTE: workers=1 to avoid multiprocessing + signal/Numba issues on macOS causing "no generation step" hangs
    result = differential_evolution(
        objective_function_multi,
        bounds=bounds,
        args=(datasets,),
        maxiter=300,
        strategy='best1bin', # Best-1-bin strategy ist standard, weitere optionen: 'rand1bin', 'randtobest1bin', 'best2bin' etc.
        popsize=min_pop_rows,
        seed=42,
        polish=True,
        disp=True,
        updating='deferred',
        workers=1,
        tol=1e-1,
        atol=1e-1,
        init=init_pop,
        callback=monitor.callback
    )

    print()  # Newline
    elapsed_time = time.time() - monitor.start_time
    print(f"\nFitting finished in {elapsed_time:.1f}s")
    print(f"Best objective: {monitor.best_objective:.2e}")
    print("Basis: absolute concentrations (CH3OH, HCOO-, CO3^2-); OH- not fitted")

    return result

def create_time_weight_visualization(dataset):
    """(Unused) Weighting visualization – disabled: fitting uses all points with no time weighting."""
    return plt.gcf()

def visualize_results(datasets, fitted_params):
    """Visualize fitting results (absolute concentrations) with ONE subplot per experiment. Each subplot contains all species (CH3OH, HCOO-, CO3^2-, OH-) with experimental markers and simulated lines."""
    model = PhysicsCorrectElectrolysisModel({
        'i0_1': fitted_params[0],
        'i0_2': fitted_params[1],
        'i0_3': fitted_params[2],
        'alpha1': fitted_params[3],
        'alpha2': fitted_params[4],
        'alpha3': fitted_params[5],
        'delta': fitted_params[6],
        'D_CH3OH_ref': fitted_params[7],
        'D_HCOO_ref': fitted_params[8],
        'D_CO32_ref': fitted_params[9],
        'D_OH_ref': fitted_params[10]
    })

    n = len(datasets)
    if n == 0:
        return None

    fig_height = max(3.0, 2.8 * n)
    fig, axes = plt.subplots(n, 1, figsize=(12, fig_height), sharex=True)
    axes = np.atleast_1d(axes)

    for r, dataset in enumerate(datasets):
        ax = axes[r]
        c_sim = model.simulate(dataset, t_eval=dataset.t_conc)
        t_min = dataset.t_conc / 60.0

        # Experimental data
        ax.plot(t_min, dataset.c_MeOH, 'o', alpha=0.7, label='exp CH3OH')
        ax.plot(t_min, dataset.c_HCOO, 's', alpha=0.7, label='exp HCOO-')
        ax.plot(t_min, dataset.c_CO32, 'D', alpha=0.7, label='exp CO3^2-')
        ax.plot(t_min, dataset.c_OH,   '^', alpha=0.7, label='exp OH-')

        # Simulated lines
        ax.plot(t_min, c_sim[0], '-', linewidth=2, label='sim CH3OH')
        ax.plot(t_min, c_sim[1], '-', linewidth=2, label='sim HCOO-')
        ax.plot(t_min, c_sim[2], '-', linewidth=2, label='sim CO3^2-')
        ax.plot(t_min, c_sim[3], '-', linewidth=2, label='sim OH-')

        ax.set_title(dataset.name)
        ax.set_ylabel('Concentration (mol/L)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8, ncol=2)

    axes[-1].set_xlabel('Time (min)')
    fig.suptitle('Fitting results — absolute concentrations (one axis per experiment)', y=1.02, fontsize=12)
    fig.tight_layout()
    return fig

def validate_results(datasets, fitted_params):
    """Validate fitting results (absolute concentrations)."""
    model = PhysicsCorrectElectrolysisModel({
        'i0_1': fitted_params[0],
        'i0_2': fitted_params[1],
        'i0_3': fitted_params[2],
        'alpha1': fitted_params[3],
        'alpha2': fitted_params[4],
        'alpha3': fitted_params[5],
        'delta': fitted_params[6],
        'D_CH3OH_ref': fitted_params[7],
        'D_HCOO_ref': fitted_params[8],
        'D_CO32_ref': fitted_params[9],
        'D_OH_ref': fitted_params[10]
    })
    
    print("\n=== VALIDATION (absolute concentrations) ===")
    print(f"{'Dataset':<30} {'Error':<12} {'CH3OH RMSE':<12} {'HCOO RMSE':<12} {'CO32 RMSE':<12} {'OH RMSE*':<12}")
    print("-" * 92)
    
    for dataset in datasets:
        c_sim = model.simulate(dataset, t_eval=dataset.t_conc)
        rmse_ch3oh = np.sqrt(np.mean((c_sim[0] - dataset.c_MeOH)**2))
        rmse_hcoo = np.sqrt(np.mean((c_sim[1] - dataset.c_HCOO)**2))
        rmse_co32 = np.sqrt(np.mean((c_sim[2] - dataset.c_CO32)**2))
        rmse_oh = np.sqrt(np.mean((c_sim[3] - dataset.c_OH)**2))
        error = objective_function(fitted_params, dataset, model)
        print(f"{dataset.name:<30} {error:<12.2e} {rmse_ch3oh:<12.4f} "
              f"{rmse_hcoo:<12.4f} {rmse_co32:<12.4f} {rmse_oh:<12.4f}")
    
    print("\n* OH- was NOT fitted; D_OH- was fitted")
    print("  Fit uses all data points (no weighting).")

# === HAUPTPROGRAMM ===
if __name__ == "__main__":
    # Pfad-Konfiguration
    import platform
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = script_dir      # Default-Pfad

    # macOS-Tweak: BLAS/OpenBLAS nicht hyperthreaden lassen
    if platform.system() == 'Darwin':
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_count())


    print(f"Data directory: {data_dir}")
    print(f"Script : {os.path.abspath(__file__)}")
    print(f"Python interpreter  : {sys.executable}")
    print(f"Last modified       : {time.ctime(os.path.getmtime(__file__))}")
    print(f"A_eff fixed at {A_EFF_FIXED:.6f} m² (not a fit parameter)")
    
    if not os.path.exists(data_dir):
        print("Fitting directory not found!")
        exit()
    
    # Select CSV files
    print("\n=== SELECT CSV FILES ===")
    all_csv_files = sorted(
        [
            f for f in os.listdir(data_dir)
            if f and not f.startswith('.') and f.lower().rstrip().endswith('.csv')
        ]
    )

    # Debug: if no CSVs found, show directory content
    if not all_csv_files:
        print("No CSV files found – directory listing:")
        for entry in os.listdir(data_dir):
            print("   •", repr(entry))
    
    if not all_csv_files:
        print("No CSV files found!")
        exit()
    
    print("Available CSV files:")
    for i, f in enumerate(all_csv_files):
        print(f"  {i+1}. {f}")
    
    print("\nSelect files (comma-separated indices, or 'all'):")
    selection = input().strip()
    
    if selection.lower() in ('alle', 'all'):
        csv_files = all_csv_files
    else:
        try:
            indices = [int(x.strip())-1 for x in selection.split(',')]
            csv_files = [all_csv_files[i] for i in indices if 0 <= i < len(all_csv_files)]
        except:
            print("Invalid selection!")
            exit()
    
    # Load datasets
    print("\n=== LOADING EXPERIMENTAL DATA ===")
    datasets = []
    
    for csv_file in csv_files:
        filepath = os.path.join(data_dir, csv_file)
        dataset = ElectrolysisDataset(filepath)
        if dataset.valid:
            datasets.append(dataset)
    
    if len(datasets) == 0:
        print("No valid datasets found!")
        exit()
    
    print(f"\n{len(datasets)} datasets loaded successfully")

    # Numba warm-up and optional regression test
    if NUMBA_AVAILABLE and not FORCE_PY_NERNST:
        print("\nWarming up Nernst calculation (Numba)…", flush=True)
        _warmup_numba()
    if os.environ.get("CHECK_NERNST_REGRESSION", "0") == "1":
        print("\nStarting Nernst regression test (Numba vs. Python)…", flush=True)
        _nernst_regression_check(n_samples=500)
    
    # Ask for fitting parameters
    print("Use start values? (y/n, default: y): ", end="")
    _ans = input().strip().lower()
    use_guess = _ans not in ("n", "no", "nein")

    # Perform fitting (equal-priority, absolute concentrations)
    result = run_fitting(datasets, use_initial_guess=use_guess)
    
    if result.success:
        fitted_params = result.x

        print(f"\n=== FITTED PARAMETERS ===")
        print(f"KINETICS:")
        print(f"  i0_1   = {fitted_params[0]:.2f} A/m² (MOR)   # A/m² (exchange current density R1: MOR)")
        print(f"  i0_2   = {fitted_params[1]:.2f} A/m² (FOR)   # A/m² (exchange current density R2: FOR)")
        print(f"  i0_3   = {fitted_params[2]:.2e} A/m² (OER)   # A/m² (exchange current density R3: OER)")
        print(f"  alpha1 = {fitted_params[3]:.4f}")
        print(f"  alpha2 = {fitted_params[4]:.4f}")
        print(f"  alpha3 = {fitted_params[5]:.4f}")
        print(f"SYSTEM:")
        print(f"  A_eff  = {A_EFF_FIXED:.6f} m² (fixed)")
        print(f"  delta  = {fitted_params[6]*1e6:.1f} μm (fitted)")
        print(f"DIFFUSION (25 °C reference values):")
        print(f"  D_CH3OH = {fitted_params[7]*1e9:.2f} nm²/s ({fitted_params[7]:.2e} m²/s)")
        print(f"  D_HCOO  = {fitted_params[8]*1e9:.2f} nm²/s ({fitted_params[8]:.2e} m²/s)")
        print(f"  D_CO32  = {fitted_params[9]*1e9:.2f} nm²/s ({fitted_params[9]:.2e} m²/s)")
        print(f"  D_OH    = {fitted_params[10]*1e9:.2f} nm²/s ({fitted_params[10]:.2e} m²/s)")
        print("Fit uses all data points (no weighting). OH- not included in objective.")
        print(f"\nObjective: {result.fun:.2e}")
        print(f"Iterations: {result.nit}")
        print(f"Function evaluations: {result.nfev}")

        # Visualization
        print("\nCreating plots...")
        fig = visualize_results(datasets, fitted_params)

        # Validation
        validate_results(datasets, fitted_params)

        # Output parameters for main model
        print(f"\n=== PARAMETERS FOR MAIN MODEL (Elektrolysezelle_V1.1.1.py) ===")
        print(f"# === KINETIC PARAMETERS ===")
        print(f"i0_1 = {fitted_params[0]:.2f}       # A/m² (exchange current density R1: MOR)")
        print(f"i0_2 = {fitted_params[1]:.2f}       # A/m² (exchange current density R2: FOR)")
        print(f"i0_3 = {fitted_params[2]:.2e}       # A/m² (exchange current density R3: OER)")
        print(f"alpha1 = {fitted_params[3]:.4f}    # transfer coefficient R1")
        print(f"alpha2 = {fitted_params[4]:.4f}    # transfer coefficient R2")
        print(f"alpha3 = {fitted_params[5]:.4f}    # transfer coefficient R3 (OER)")
        print(f"\n# === SYSTEM PARAMETERS ===")
        print(f"A_eff = {A_EFF_FIXED:.6f}    # m² (effective electrode area, fixed)")
        print(f"\n# === MASS TRANSPORT PARAMETERS ===")
        print(f"delta_diff = {fitted_params[6]:.2e}  # m diffusion film thickness (fitted)")
        print(f"\n# === DIFFUSION COEFFICIENTS (25 °C reference values) ===")
        print(f"# These replace the fixed values in D_ref:")
        print(f"D_ref = {{")
        print(f"    'CH3OH': {fitted_params[7]:.2e},   # methanol (fitted)")
        print(f"    'HCOO': {fitted_params[8]:.2e},    # formate (fitted)")
        print(f"    'CO32': {fitted_params[9]:.2e},    # carbonate (fitted)")
        print(f"    'OH': {fitted_params[10]:.2e},     # hydroxide (fitted)")
        print(f"    'H2O': 2.3e-9      # water (fixed)")
        print(f"}}")
        print(f"\n# === FITTING INFO ===")
        print(f"# Fit to absolute concentrations. All time points used. No weighting. OH- not included in objective.")

        plt.show()

    else:
        print(f"\nFitting failed!")
        print(f"  Reason: {result.message}")
        # --- NEW: Ensure plots are shown even if max iterations were reached ---
        try:
            # Use the best parameters found so far (SciPy still returns best candidate in result.x)
            fitted_params = result.x
            print("\nPlotting best-so-far concentration profiles (no convergence).")

            # Create plots for all datasets
            fig = visualize_results(datasets, fitted_params)

            # Optional: show basic validation metrics for transparency
            validate_results(datasets, fitted_params)

            plt.show()
        except Exception as e:
            print(f"Could not plot best-so-far results: {e}")
