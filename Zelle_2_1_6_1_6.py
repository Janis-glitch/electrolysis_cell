"""
Electrolysis cell — simulation with fixed parameters

- Fixed parameters (edit in the CONFIG section below).
- User can freely choose initial concentrations for CH3OH, HCOO-, CO3^2-, OH-,
  and the galvanostatic current (1–20 A).
- Computes concentration time series:
  Nernst, Butler–Volmer with RDS n=1, mass transport with δ(I) scaling,
  geometric vs effective area treatment, and temperature-corrected thermodynamics.

 Optional Numba for Nernst.
"""

import os
import sys
import math
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import lru_cache

# --- Optional Numba acceleration for Nernst (safe to disable) ------------
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    print("Numba not available. Using pure Python for Nernst.")

warnings.filterwarnings("ignore")

# =========================================================================
# CONFIG — FIXED MODEL PARAMETERS (EDIT ME IF NEEDED)
# =========================================================================

# Temperature and constants
F  = 96485.0        # C/mol
R  = 8.314          # J/(mol·K)
T  = 323.15         # K (50 °C)
T0 = 298.15         # K

# Stoichiometry electron numbers
n1, n2, n3 = 4, 2, 4           # MOR, FOR, OER (net)
n1_RDS, n2_RDS, n3_RDS = 1, 1, 1  # RDS electrons for BV

# Water initial concentration (activity ~ 1)
c_H2O0 = 55.56  # mol/L

# Effective vs geometric area (ECSA factor ~390)
A_GEO_FIXED = 0.0025             # m² (25 cm²)
A_EFF_FIXED = A_GEO_FIXED * 390  # m²

# Standard potentials at 25 °C (vs RHE) and entropy terms for T‑correction
E1_std_25C = -0.0689  # V (MOR)
E2_std_25C = -0.1014  # V (FOR)
delta_S1   = 298.2    # J/(mol·K)
delta_S2   = 23.8     # J/(mol·K)

dE1_dT = delta_S1 / (n1 * F)
dE2_dT = delta_S2 / (n2 * F)
E1_std = E1_std_25C + dE1_dT * (T - T0)
E2_std = E2_std_25C + dE2_dT * (T - T0)

# OER standard potential (weak T dependency neglected, vs RHE)
E3_std = 1.2297  # V

# ---- Kinetic parameters (per A_eff). These are FIXED (not fitted). ----
# Defaults are taken from your latest "initial_guess" in the former fitter.
I0_1   = 0.90       # A/m² (MOR)
I0_2   = 0.22       # A/m² (FOR)
I0_3   = 7.15e-4    # A/m² (OER)
ALPHA1 = 0.3450
ALPHA2 = 0.2963
ALPHA3 = 0.5158

# ---- Mass-transport parameters (fixed) ----
DELTA_BASE      = 16.6e-6  # m — base diffusion film at Iref
IREF_DELTA_A    = 5.0      # A — reference current for δ(I)
BETA_DELTA      = 0.20     # exponent in δ(I) = δ_base * (Iref/I)^beta, clamped 5–200 µm

# ---- Diffusion coefficients (reference values at 25 °C, fixed) ----
D_CH3OH_ref = 1.20e-9
D_HCOO_ref  = 1.79e-9
D_CO32_ref  = 1.05e-9
D_OH_ref    = 5.25e-9
D_H2O_ref   = 2.30e-9  # fixed (not used for transport limitation)

# ---- Numerical clamps/safety ----
C_MIN_NERNST      = 1e-8   # mol/L
C_MIN_SOLVE       = 1e-8   # mol/L
C_MIN_SOLVE_OH    = 1e-3   # mol/L (slightly higher at surface for stability)
C_MIN_ODE_C       = 1e-8   # mol/L
C_MIN_ODE_OH      = 1e-8   # mol/L
C_MIN_H2O         = 30.0   # mol/L

# ---- Minimum allowed initial concentration for user inputs -------------
MIN_INPUT_CONC = 1e-3  # mol/L — hard lower bound for any provided initial concentration

# ---- i0 concentration scaling (kept minimal and fixed exponents) ----
I0_CREF    = 1.0   # mol/L
I0_M1_REACT = 1.0  # order for CH3OH in MOR
I0_P1_OH    = 1.0  # order for OH- in MOR
I0_M2_REACT = 1.0  # order for HCOO- in FOR
I0_P2_OH    = 1.0  # order for OH- in FOR
I0_P3_OH    = 1.0  # order for OH- in OER

# =========================================================================
# HELPER FUNCTIONS — transport, thermodynamics, and kinetics helpers
# =========================================================================

RT_over_n1F = R * T / (n1 * F)
RT_over_n2F = R * T / (n2 * F)
n1_RDSF_over_RT = n1_RDS * F / (R * T)
n2_RDSF_over_RT = n2_RDS * F / (R * T)
n3_RDSF_over_RT = n3_RDS * F / (R * T)

def calculate_volume_from_current(i_total_A: float) -> float:
    """
    Reactor volume vs current (empirical linear map, clamped).
    Returns liters.
    """
    I_ref = [5.0, 10.0]           # A
    V_ref = [0.05026, 0.056474]   # L
    slope = (V_ref[1] - V_ref[0]) / (I_ref[1] - I_ref[0])
    volume = V_ref[0] + slope * (i_total_A - I_ref[0])
    return max(0.01, min(0.1, volume))

@lru_cache(maxsize=1000)
def _cached_diffusion_coeffs(c_OH_hash: int,
                             D_CH3OH: float, D_HCOO: float, D_CO32: float, D_OH: float):
    """
    Cached concentration‑ and temperature‑corrected diffusion coefficients.
    Returns array [D_CH3OH, D_HCOO, D_CO32, D_OH, D_H2O].
    c_OH_hash encodes c_OH in µmol/L to keep cache size modest.
    """
    c_OH = c_OH_hash * 1e-6  # back to mol/L

    # Relative viscosity of KOH at 25 °C (fit, valid up to 2 M)
    # n_rel(c) ≈ 1 + 0.2351*c + 0.04774*c², c in mol/L
    c_KOH = max(0.0, min(2.0, c_OH))
    viscosity_factor_rel = 1.0 + 0.23509677419354838 * c_KOH + 0.047741935483870894 * (c_KOH ** 2)

    # Temperature factor via Arrhenius-like ratio for water viscosity
    T_ref = 298.15
    E_a_water = 15900.0
    viscosity_ratio_T = math.exp(E_a_water / R * (1.0 / T - 1.0 / T_ref))

    def correct(D25):
        return D25 * (T / T_ref) / (viscosity_ratio_T * viscosity_factor_rel)

    return np.array([
        correct(D_CH3OH), correct(D_HCOO), correct(D_CO32), correct(D_OH), correct(D_H2O_ref)
    ])

def diffusion_coefficients(c_OH: float) -> np.ndarray:
    c_hash = int(round(c_OH * 1e6))
    return _cached_diffusion_coeffs(c_hash, D_CH3OH_ref, D_HCOO_ref, D_CO32_ref, D_OH_ref)

# --- Nernst potentials (Numba path if available) -------------------------
if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=False)
    def _nernst_numba(c1, c2, c3, c4):
        c1 = C_MIN_NERNST if c1 < C_MIN_NERNST else c1
        c2 = C_MIN_NERNST if c2 < C_MIN_NERNST else c2
        c3 = C_MIN_NERNST if c3 < C_MIN_NERNST else c3
        c4 = C_MIN_NERNST if c4 < C_MIN_NERNST else c4
        Q1 = c2 / (c1 * c4**5)   # MOR
        Q2 = c3 / (c2 * c4**3)   # FOR
        E1 = E1_std - RT_over_n1F * np.log(Q1)
        E2 = E2_std - RT_over_n2F * np.log(Q2)
        return E1, E2

@lru_cache(maxsize=5000)
def _nernst_py_cached(c1h, c2h, c3h, c4h):
    c1 = c1h * 1e-8; c2 = c2h * 1e-8; c3 = c3h * 1e-8; c4 = c4h * 1e-8
    c1 = max(c1, C_MIN_NERNST); c2 = max(c2, C_MIN_NERNST)
    c3 = max(c3, C_MIN_NERNST); c4 = max(c4, C_MIN_NERNST)
    Q1 = c2 / (c1 * c4**5)
    Q2 = c3 / (c2 * c4**3)
    E1 = E1_std - RT_over_n1F * math.log(Q1)
    E2 = E2_std - RT_over_n2F * math.log(Q2)
    return E1, E2

def nernst(c1, c2, c3, c4):
    if NUMBA_AVAILABLE:
        return _nernst_numba(c1, c2, c3, c4)
    return _nernst_py_cached(
        int(round(c1 * 1e8)),
        int(round(c2 * 1e8)),
        int(round(c3 * 1e8)),
        int(round(c4 * 1e8)),
    )

# =========================================================================
# CORE MODEL
# =========================================================================

class PhysicsCorrectElectrolysisModel:
    """
    Electrolysis model with fixed parameters.
    """
    def __init__(self):
        self.i0_1 = I0_1
        self.i0_2 = I0_2
        self.i0_3 = I0_3
        self.alpha1 = ALPHA1
        self.alpha2 = ALPHA2
        self.alpha3 = ALPHA3
        self.A_eff = A_EFF_FIXED
        self.delta_base = DELTA_BASE
        self.beta_delta = BETA_DELTA
        self.Iref_delta = IREF_DELTA_A

        self.n1F = n1 * F
        self.n2F = n2 * F
        self.n3F = n3 * F
        self._last_E = None

    def _delta_eff(self, i_total):
        I = max(1e-6, float(i_total))
        d = self.delta_base * (self.Iref_delta / I) ** self.beta_delta
        # Clamp δ between 5 µm and 200 µm
        if d < 5e-6: d = 5e-6
        if d > 200e-6: d = 200e-6
        return d

    def _solve_currents(self, E_anode, c1, c2, c3, c4, i_total):
        """
        Solve partial currents with Koutecky–Levich + BV, including
        multi‑reactant transport limits and surface concentration effects.
        """
        max_iter = 30
        tol = 1e-7

        delta_eff = self._delta_eff(i_total)
        inv_delta = 1.0 / delta_eff

        c1 = max(c1, C_MIN_SOLVE); c2 = max(c2, C_MIN_SOLVE)
        c3 = max(c3, C_MIN_SOLVE); c4 = max(c4, C_MIN_SOLVE)

        # Equilibrium potentials and overpotentials (bulk)
        E1_eq_b, E2_eq_b = nernst(c1, c2, c3, c4)
        eta1_b = E_anode - E1_eq_b
        eta2_b = E_anode - E2_eq_b
        eta3_b = E_anode - E3_std

        # i0 effective (bulk-based, minimal-invasive)
        c_ref = I0_CREF
        i0_1_eff = self.i0_1 * (c1 / c_ref)**I0_M1_REACT * (c4 / c_ref)**I0_P1_OH
        i0_2_eff = self.i0_2 * (c2 / c_ref)**I0_M2_REACT * (c4 / c_ref)**I0_P2_OH
        i0_3_eff = self.i0_3 * (c4 / c_ref)**I0_P3_OH

        # constants
        Dvals = diffusion_coefficients(c4)
        c1_m3, c2_m3, c4_m3 = c1 * 1000.0, c2 * 1000.0, c4 * 1000.0
        A_eff = self.A_eff
        area_geo_to_eff = A_GEO_FIXED / A_eff
        area_eff_to_geo = A_eff / A_GEO_FIXED

        def j_lim_bulk(nF_reac, stoich_dict):
            cand = []
            for sp, nu in stoich_dict.items():
                if nu <= 0: continue
                if sp == 'CH3OH':
                    Dk, ck_m3 = Dvals[0], c1_m3
                elif sp == 'HCOO':
                    Dk, ck_m3 = Dvals[1], c2_m3
                elif sp == 'OH':
                    Dk, ck_m3 = Dvals[3], c4_m3
                else:
                    continue
                cand.append((nF_reac * Dk * ck_m3 * inv_delta / float(nu)) * area_geo_to_eff)
            return min(cand) if cand else 1e30

        # optimistic bulk KL caps (rarely used, but cheap to compute)
        _ = j_lim_bulk(self.n1F, {'CH3OH': 1, 'OH': 5})
        _ = j_lim_bulk(self.n2F, {'HCOO': 1, 'OH': 3})
        _ = j_lim_bulk(self.n3F, {'OH': 4})

        i1_old = i2_old = i3_old = 0.0

        MAX_EXP = 60.0
        def exp_clamped(x):
            if x > MAX_EXP: x = MAX_EXP
            elif x < -MAX_EXP: x = -MAX_EXP
            return math.exp(x)

        for _iter in range(max_iter):
            J1 = i1_old / self.n1F; J2 = i2_old / self.n2F; J3 = i3_old / self.n3F
            j1 = J1 / A_eff; j2 = J2 / A_eff; j3 = J3 / A_eff

            # Convert to geometric for film theory
            j1_geo = j1 * area_eff_to_geo
            j2_geo = j2 * area_eff_to_geo
            j3_geo = j3 * area_eff_to_geo

            # Surface concentrations
            c1_s = max(c1 - j1_geo * delta_eff / (Dvals[0] * 1000.0), C_MIN_SOLVE)
            c2_s = max(c2 - (-j1_geo + j2_geo) * delta_eff / (Dvals[1] * 1000.0), C_MIN_SOLVE)
            c3_s = max(c3 - (-j2_geo) * delta_eff / (Dvals[2] * 1000.0), C_MIN_SOLVE)
            c4_s = max(c4 - (5*j1_geo + 3*j2_geo + 4*j3_geo) * delta_eff / (Dvals[3] * 1000.0), C_MIN_SOLVE_OH)

            # Surface potentials/etas
            E1_s, E2_s = nernst(c1_s, c2_s, c3_s, c4_s)
            eta1 = E_anode - E1_s
            eta2 = E_anode - E2_s
            eta3 = E_anode - E3_std

            # BV currents (anodic only; suppress cathodic)
            phi1 = n1_RDSF_over_RT * eta1
            phi2 = n2_RDSF_over_RT * eta2
            phi3 = n3_RDSF_over_RT * eta3
            j_bv1 = i0_1_eff * (exp_clamped(ALPHA1 * phi1) - exp_clamped(-(1.0 - ALPHA1) * phi1))
            j_bv2 = i0_2_eff * (exp_clamped(ALPHA2 * phi2) - exp_clamped(-(1.0 - ALPHA2) * phi2))
            j_bv3 = i0_3_eff * (exp_clamped(ALPHA3 * phi3) - exp_clamped(-(1.0 - ALPHA3) * phi3))
            if j_bv1 < 0.0: j_bv1 = 0.0
            if j_bv2 < 0.0: j_bv2 = 0.0
            if j_bv3 < 0.0: j_bv3 = 0.0

            # Diffusion limits for each reaction at bulk (multi-reactant min rule)
            def j_lim_reaction(nF_reac, stoich_dict):
                cand = []
                for sp, nu in stoich_dict.items():
                    if nu <= 0: continue
                    if sp == 'CH3OH':
                        Dk, ck_m3 = Dvals[0], c1_m3
                    elif sp == 'HCOO':
                        Dk, ck_m3 = Dvals[1], c2_m3
                    elif sp == 'OH':
                        Dk, ck_m3 = Dvals[3], c4_m3
                    else:
                        continue
                    cand.append((nF_reac * Dk * ck_m3 * inv_delta / float(nu)) * area_geo_to_eff)
                return min(cand) if cand else 1e30

            j_lim1 = j_lim_reaction(self.n1F, {'CH3OH': 1, 'OH': 5})
            j_lim2 = j_lim_reaction(self.n2F, {'HCOO': 1, 'OH': 3})
            j_lim3 = j_lim_reaction(self.n3F, {'OH': 4})

            j1_eff = (j_bv1 * j_lim1) / (j_bv1 + j_lim1)
            j2_eff = (j_bv2 * j_lim2) / (j_bv2 + j_lim2)
            j3_eff = (j_bv3 * j_lim3) / (j_bv3 + j_lim3)

            i1 = j1_eff * A_eff
            i2 = j2_eff * A_eff
            i3 = j3_eff * A_eff

            if abs(i1 - i1_old) < tol and abs(i2 - i2_old) < tol and abs(i3 - i3_old) < tol:
                break
            i1_old, i2_old, i3_old = i1, i2, i3

        return i1, i2, i3

    def _find_E_anode(self, c1, c2, c3, c4, i_total):
        """
        Root-find E_anode such that i1+i2+i3 = i_total (robust bracketing).
        """
        E1_eq, E2_eq = nernst(c1, c2, c3, c4)
        E3_eq = E3_std

        def g(E):
            i1, i2, i3 = self._solve_currents(E, c1, c2, c3, c4, i_total)
            return (i1 + i2 + i3) - i_total

        # Prefer last solution if available, else center above max equilibrium potential
        E_center = self._last_E if self._last_E is not None else max(E1_eq, E2_eq, E3_eq) + 0.2

        # Local small bracket
        grid_small = np.linspace(E_center - 0.7, E_center + 0.7, 9)
        vals_small = [g(E) for E in grid_small]
        for a, b, fa, fb in zip(grid_small[:-1], grid_small[1:], vals_small[:-1], vals_small[1:]):
            if np.isfinite(fa) and np.isfinite(fb) and np.sign(fa) != np.sign(fb):
                try:
                    from scipy.optimize import brentq
                    E = float(brentq(g, a, b, xtol=1e-6))
                    self._last_E = E
                    return E
                except Exception:
                    pass

        # Coarse fallback
        grid = np.linspace(max(E_center, max(E1_eq, E2_eq, E3_eq)+0.1) - 3.5,
                           max(E_center, max(E1_eq, E2_eq, E3_eq)+0.1) + 3.5, 15)
        vals = [g(E) for E in grid]
        for a, b, fa, fb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
            if np.isfinite(fa) and np.isfinite(fb) and np.sign(fa) != np.sign(fb):
                try:
                    from scipy.optimize import brentq
                    E = float(brentq(g, a, b, xtol=1e-6))
                    self._last_E = E
                    return E
                except Exception:
                    pass

        # Best-effort fallback: choose grid point with minimal |g|
        vals_arr = np.array(vals_small if len(vals_small) else vals, dtype=float)
        grids    = np.array(grid_small if len(vals_small) else grid, dtype=float)
        bad = ~np.isfinite(vals_arr)
        vals_arr[bad] = 1e18
        idx = int(np.argmin(np.abs(vals_arr)))
        E = float(grids[idx])
        self._last_E = E
        return E

    def simulate(self, i_total_A: float,
                 c0_MeOH: float, c0_HCOO: float, c0_CO32: float, c0_OH: float,
                 t_end_s: float = 3600.0, n_points: int = 301):
        """
        Run a simulation with given current and initial concentrations.
        Returns time (s) and concentrations array [CH3OH, HCOO-, CO3^2-, OH-, H2O].
        """
        from scipy.integrate import solve_ivp

        # Reset last potential
        self._last_E = None

        # Volume from current
        V_L = calculate_volume_from_current(i_total_A)
        V_m3 = V_L / 1000.0

        # Enforce hard lower bound on provided initial concentrations
        c0_MeOH = max(c0_MeOH, MIN_INPUT_CONC)
        c0_HCOO = max(c0_HCOO, MIN_INPUT_CONC)
        c0_CO32 = max(c0_CO32, MIN_INPUT_CONC)
        c0_OH   = max(c0_OH,   MIN_INPUT_CONC)

        # Initial state
        x0 = [
            max(c0_MeOH, C_MIN_ODE_C),
            max(c0_HCOO, C_MIN_ODE_C),
            max(c0_CO32, C_MIN_ODE_C),
            max(c0_OH,   C_MIN_ODE_OH),
            c_H2O0,
            c0_HCOO * V_L,  # n_HCOO_total (mol) [not used externally]
            c0_CO32 * V_L,  # n_CO32_total (mol)
            0.0             # Q_total (C)
        ]
        t_eval = np.linspace(0.0, float(t_end_s), int(n_points))

        A_eff_over_V = self.A_eff / V_L  # V in L because concentrations are mol/L

        def odes(t, x):
            c1, c2, c3, c4, c5, n_HCOO_tot, n_CO32_tot, Q_tot = x
            c1 = max(c1, C_MIN_ODE_C)
            c2 = max(c2, C_MIN_ODE_C)
            c3 = max(c3, C_MIN_ODE_C)
            c4 = max(c4, C_MIN_ODE_OH)
            c5 = max(c5, C_MIN_H2O)

            E_an = self._find_E_anode(c1, c2, c3, c4, i_total_A)
            i1, i2, i3 = self._solve_currents(E_an, c1, c2, c3, c4, i_total_A)

            r1 = i1 / (self.A_eff * self.n1F)
            r2 = i2 / (self.A_eff * self.n2F)
            r3 = i3 / (self.A_eff * self.n3F)

            r1_v = r1 * A_eff_over_V
            r2_v = r2 * A_eff_over_V
            r3_v = r3 * A_eff_over_V

            # Concentration rates (effective balance)
            dc1_dt = -r1_v
            dc2_dt =  r1_v - r2_v
            dc3_dt =  r2_v
            dc4_dt = (-r1_v - r2_v)
            dc5_dt = ( 2*r1_v + r2_v)

            dn_HCOO_dt = r1 * self.A_eff
            dn_CO32_dt = r2 * self.A_eff
            dQ_dt = i_total_A

            return [dc1_dt, dc2_dt, dc3_dt, dc4_dt, dc5_dt,
                    dn_HCOO_dt, dn_CO32_dt, dQ_dt]

        sol = solve_ivp(odes, (0.0, float(t_end_s)), x0,
                        method="Radau", t_eval=t_eval,
                        rtol=1e-6, atol=1e-6, max_step=50.0)

        if not sol.success:
            print("Warning: ODE solver reported:", sol.message)

        return sol.t, sol.y[:5]  # (time_s, concentrations array)

# =========================================================================
# CLI —  interactive runner
# =========================================================================

def _prompt_float(msg, default, vmin=None, vmax=None):
    try:
        raw = input(f"{msg} [{default}]: ").strip()
        val = float(raw) if raw else float(default)
        if vmin is not None and val < vmin:
            print(f"Value too small, clamped to {vmin}."); val = vmin
        if vmax is not None and val > vmax:
            print(f"Value too large, clamped to {vmax}."); val = vmax
        return val
    except Exception:
        print("Invalid input, using default.")
        return float(default)

def main():
    import platform

    if platform.system() == 'Darwin':
        # avoid over-threading on macOS Accelerate/veclib
        os.environ['VECLIB_MAXIMUM_THREADS'] = "1"

    print("=== Electrolysis Simulation (no fitting) ===")
    print(f"Script : {os.path.abspath(__file__)}")
    print(f"Python : {sys.executable}")
    print(f"A_eff fixed at {A_EFF_FIXED:.6f} m²; A_geo = {A_GEO_FIXED:.6f} m²")
    print("Enter initial concentrations (mol/L) and current (A). Minimum initial concentration is 1e-3 mol/L.")

    I = _prompt_float("Current I_total (A, 1–20)", 5.0, 1.0, 20.0)
    c1 = _prompt_float("c0 CH3OH (mol/L)", 1.0, MIN_INPUT_CONC, 10.0)
    c2 = _prompt_float("c0 HCOO- (mol/L)", 0.0, MIN_INPUT_CONC, 10.0)
    c3 = _prompt_float("c0 CO3^2- (mol/L)", 0.0, MIN_INPUT_CONC, 10.0)
    c4 = _prompt_float("c0 OH- (mol/L)", 1.0, MIN_INPUT_CONC, 10.0)
    tend = _prompt_float("Simulation time (s)", 3600.0, 1.0, 1e6)
    npts = int(_prompt_float("Number of time points", 301, 5, 10001))

    model = PhysicsCorrectElectrolysisModel()
    t_s, C = model.simulate(I, c1, c2, c3, c4, tend, npts)

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True)
    labels = ["CH3OH", "HCOO-", "CO3^2-", "OH-"]
    for i, ax in enumerate(axes):
        ax.plot(t_s/60.0, C[i], '-', linewidth=2)
        ax.set_title(labels[i])
        ax.set_xlabel("Time (min)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel("Concentration (mol/L)")
    fig.suptitle(f"Simulation — I={I:.2f} A, V={calculate_volume_from_current(I)*1000:.1f} mL", y=1.02)
    fig.tight_layout()
    plt.show()

    # Optional CSV export
    ans = input("Export CSV of the simulated profiles? (y/n) [n]: ").strip().lower()
    if ans in ("y", "yes", "j", "ja"):
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"sim_I{I:.2f}A_{int(time.time())}.csv")
        df = pd.DataFrame({
            "Time_s": t_s,
            "Time_min": t_s/60.0,
            "CH3OH_sim_molL": C[0],
            "HCOO_sim_molL":  C[1],
            "CO32_sim_molL":  C[2],
            "OH_sim_molL":    C[3],
            "H2O_sim_molL":   C[4],
        })
        try:
            df.to_csv(out_path, index=False)
            print(f"✓ Exported: {out_path}")
        except Exception as e:
            print(f"✗ Could not export CSV: {e}")

if __name__ == "__main__":
    main()