"""
Stand‑alone electrolysis cell model (no fitting).

- Implements the exact same physics as the former fitting script (MOR, FOR, OER; BV + KL with film theory; Nernst with clamping; ECSA vs A_geo handling; temperature/viscosity‑corrected diffusion).
- Simple, functional structure. No optimization, no timeouts, no DE, no weighting/subsampling.
- Use this to simulate concentration trajectories from fixed parameters and CSV datasets.
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
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
warnings.filterwarnings('ignore')


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
            self.valid = True
            print(f"✓ {self.name}: {len(self.t_conc)} points, I = {self.i_total:.2f} A, V = {self.V*1000:.1f} mL")
        except Exception as e:
            print(f"✗ Error loading {self.name}: {e}")
            self.valid = False
    
    # reduce_data_for_fitting removed

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
            t_eval = dataset.t_conc

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


def visualize_results(datasets, params_dict):
    """Visualize simulation results (absolute concentrations) in ONE figure.
    Rows = experiments (datasets), columns = species (CH3OH, HCOO-, CO3^2-, OH-).
    """
    model = PhysicsCorrectElectrolysisModel(params_dict)

    n = len(datasets)
    if n == 0:
        return None

    species_names = ['CH3OH', 'HCOO-', 'CO3^2-', 'OH-']

    fig_height = max(3.0, 2.8 * n)
    fig, axes = plt.subplots(n, 4, figsize=(16, fig_height), sharex='col')
    axes = np.atleast_2d(axes)

    for r, dataset in enumerate(datasets):
        c_sim = model.simulate(dataset, t_eval=dataset.t_conc)
        t_min = dataset.t_conc / 60.0

        data_pairs = [
            (dataset.c_MeOH, c_sim[0]),
            (dataset.c_HCOO, c_sim[1]),
            (dataset.c_CO32, c_sim[2]),
            (dataset.c_OH,   c_sim[3]),
        ]

        for c, (exp_arr, sim_arr) in enumerate(data_pairs):
            ax = axes[r, c]
            ax.plot(t_min, exp_arr, 'o', alpha=0.7, label='exp')
            ax.plot(t_min, sim_arr, '-', linewidth=2, label='sim')
            if r == 0:
                ax.set_title(species_names[c])
            if c == 0:
                ax.set_ylabel(f"{dataset.name}\nConcentration (mol/L)")
            if r == n - 1:
                ax.set_xlabel('Time (min)')
            ax.grid(True, alpha=0.3)

    axes[0, 0].legend(loc='best', fontsize=8)
    fig.suptitle('Cell model — absolute concentrations (rows = experiments, columns = species)', y=1.02, fontsize=12)
    fig.tight_layout()
    return fig

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

    # --- Fixed parameter set for the cell model (edit as needed) ---
    PARAMS = {
        'i0_1': 0.021,       # A/m² (MOR, per A_eff)
        'i0_2': 1.5e-3,      # A/m² (FOR, per A_eff)
        'i0_3': 8.0e-4,      # A/m² (OER, per A_eff)
        'alpha1': 0.21,
        'alpha2': 0.21,
        'alpha3': 0.24,
        'delta': 28e-6,      # m (diffusion film thickness)
        'D_CH3OH_ref': 1.0e-9,
        'D_HCOO_ref': 1.7e-9,
        'D_CO32_ref': 1.1e-9,
        'D_OH_ref': 5.5e-9,
    }

    print("\n=== RUNNING CELL MODEL (no fitting) ===")
    print("Parameters:")
    for k, v in PARAMS.items():
        if 'D_' in k:
            print(f"  {k:12s} = {v:.2e} m²/s")
        elif k == 'delta':
            print(f"  {k:12s} = {v*1e6:.1f} µm")
        else:
            print(f"  {k:12s} = {v}")

    try:
        fig = visualize_results(datasets, PARAMS)
        plt.show()
    except Exception as e:
        print(f"Simulation/plotting error: {e}")
