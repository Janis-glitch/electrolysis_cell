# Multi-File Elektrolysemodell - MIT TIMEOUT UND FITBAREN DIFFUSIONSKOEFFIZIENTEN
# Erweitert um D_CH3OH, D_HCOO, D_CO32, D_OH als fitbare Parameter
# NEU: Zeitgewichtung für Transportlimitierung im letzten Viertel

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, brentq, differential_evolution
import matplotlib.pyplot as plt
import time
import os
from multiprocessing import cpu_count
from functools import lru_cache, wraps
import warnings
import signal
from contextlib import contextmanager
warnings.filterwarnings('ignore')

# === TIMEOUT DECORATOR FÜR MACOS ===
class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException(f"Timeout nach {seconds}s!")
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

# === PHYSIKALISCHE KONSTANTEN ===
F = 96485.0    # Faraday-Konstante in C/mol
R = 8.314      # Gaskonstante in J/(mol·K)
T = 323.15     # Temperatur in K (50°C)
T0 = 298.15    # Referenztemperatur in K (25°C)

# === SYSTEM-PARAMETER ===
n1, n2 = 4, 2  # Elektronenzahlen für Reaktion 1 und 2
c_H2O0 = 55.56 # mol/L (Wasser-Anfangskonzentration)

# === THERMODYNAMISCHE PARAMETER ===
E1_std_25C = -0.04    # V (CH3OH -> HCOO-) vs. RHE
E2_std_25C = -0.05     # V (HCOO- -> CO32-) vs. RHE 
delta_S1 = 297.9     # J/(mol·K)
delta_S2 = 23.4      # J/(mol·K)

# Temperaturkorrigierte Standardpotentiale
dE1_dT = delta_S1 / (n1 * F)
dE2_dT = delta_S2 / (n2 * F)
E1_std = E1_std_25C + dE1_dT * (T - T0)
E2_std = E2_std_25C + dE2_dT * (T - T0)

# === PRE-COMPUTED CONSTANTS ===
RT_over_n1F = R * T / (n1 * F)
RT_over_n2F = R * T / (n2 * F)
n1F_over_RT = n1 * F / (R * T)
n2F_over_RT = n2 * F / (R * T)

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
    """Cached Diffusionskoeffizienten MIT FITBAREN PARAMETERN und CuO-Poren-Effekt"""
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
    
    # Viskositätsverhältnis
    viscosity_ratio = np.exp(E_a_water/R * (1/T - 1/T_ref))
    
    # Korrektur für hohe OH- Konzentration
    if c_OH > 1.5:
        viscosity_factor = 1 + 0.15 * (c_OH - 1.5)
    else:
        viscosity_factor = 1.0
    
    # CuO-Poren-Effekt wie im Hauptmodell
    porosity = 0.7        # Porosität CuO-Elektroden
    tortuosity = 1.6       # Tortuosität
    constriction = 0.8     # Verengungsfaktor
    
    # CuO-Poren-Effektivitätsfaktor nach Bruggeman-Relation
    cuo_pore_factor = (porosity**1.5) / tortuosity * constriction
    
    # Kombiniere alle Diffusionskoeffizienten
    D_all = {**D_ref_fittable, **D_ref_fixed}
    
    # Temperatur-, Viskositäts- und CuO-Korrektur
    D_dict = {}
    for species, D_25C in D_all.items():
        D_bulk_corrected = D_25C * (T/T_ref) / (viscosity_ratio * viscosity_factor)
        D_dict[species] = D_bulk_corrected * cuo_pore_factor
    
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
    
    c1, c2, c3, c4 = [max(c, 3e-8) for c in [c1, c2, c3, c4]]
    
    a_H2O = 1.0
    Q1 = (c2 * a_H2O**4) / (c1 * c4**5)
    Q2 = (c3 * a_H2O**2) / (c2 * c4**3)
    
    E1_eq = E1_std - RT_over_n1F * np.log(Q1)
    E2_eq = E2_std - RT_over_n2F * np.log(Q2)
    
    return E1_eq, E2_eq

def calculate_nernst_efficient(c1, c2, c3, c4):
    """Effizienter Wrapper für Nernst-Berechnung"""
    c1_hash = int(np.round(c1 * 1e8))
    c2_hash = int(np.round(c2 * 1e8))
    c3_hash = int(np.round(c3 * 1e8))
    c4_hash = int(np.round(c4 * 1e8))
    return cached_nernst_safe(c1_hash, c2_hash, c3_hash, c4_hash)

class ElectrolysisDataset:
    """Container für einen Elektrolyse-Datensatz"""
    def __init__(self, filepath):
        self.filepath = filepath
        self.name = os.path.basename(filepath).replace('.csv', '')
        self.load_data()
        
    def load_data(self):
        """Lädt und verarbeitet CSV-Daten"""
        try:
            df = pd.read_csv(self.filepath, encoding='utf-8')
            
            # Stromstärke aus ersten 10 Einträgen (mA → A)
            current_mA = df.iloc[:10, 8].mean()
            self.i_total = current_mA / 1000
            
            # Reaktorvolumen
            self.V = calculate_volume_from_current(self.i_total)
            
            # Konzentrationsdaten (Spalten A-E)
            conc_data = df.iloc[:, :5].dropna()
            
            # Elektrische Daten (Spalten F-I)
            elec_data = df.iloc[:, 5:9].dropna()
            
            # Zeitnormierung
            t_start_conc = conc_data.iloc[0, 0] if len(conc_data) > 0 else 0
            t_start_elec = elec_data.iloc[0, 0] if len(elec_data) > 0 else t_start_conc
            t_start_global = min(t_start_conc, t_start_elec)
            
            # Zeit in Sekunden, normiert auf t=0
            self.t_conc = (conc_data.iloc[:, 0].values - t_start_global) * 60
            
            # Konzentrationen (Reihenfolge aus CSV)
            self.c_HCOO = np.maximum(0, conc_data.iloc[:, 1].values)
            self.c_OH = np.maximum(0, conc_data.iloc[:, 2].values)
            self.c_MeOH = np.maximum(0, conc_data.iloc[:, 3].values)
            self.c_CO32 = np.maximum(0, conc_data.iloc[:, 4].values)
            
            # Anfangsbedingungen
            self.c0 = {
                'MeOH': max(0, conc_data.iloc[0, 3]),
                'HCOO': max(0, conc_data.iloc[0, 1]),
                'CO32': max(0, conc_data.iloc[0, 4]),
                'OH': max(0, conc_data.iloc[0, 2]),
                'H2O': c_H2O0
            }
            
            # Elektrische Daten
            if len(elec_data) > 0:
                self.t_volt = (elec_data.iloc[:, 0].values - t_start_global) * 60
                self.E_anode = elec_data.iloc[:, 2].values
            else:
                self.t_volt = None
                self.E_anode = None
            
            # Datenreduktion für Fitting
            self.reduce_data_for_fitting()
            
            self.valid = True
            
            print(f"✓ {self.name}: {len(self.t_conc)} Punkte, I = {self.i_total:.2f} A, V = {self.V*1000:.1f} mL")
            
        except Exception as e:
            print(f"✗ Fehler beim Laden von {self.name}: {e}")
            self.valid = False
    
    def reduce_data_for_fitting(self):
        """Datenreduktion für schnelleres Fitting"""
        max_points = 70
        
        if len(self.t_conc) > max_points:
            indices = np.linspace(0, len(self.t_conc)-1, max_points, dtype=int)
            self.t_conc_fit = self.t_conc[indices]
            self.c_MeOH_fit = self.c_MeOH[indices]
            self.c_HCOO_fit = self.c_HCOO[indices]
            self.c_CO32_fit = self.c_CO32[indices]
            self.c_OH_fit = self.c_OH[indices]
        else:
            self.t_conc_fit = self.t_conc
            self.c_MeOH_fit = self.c_MeOH
            self.c_HCOO_fit = self.c_HCOO
            self.c_CO32_fit = self.c_CO32
            self.c_OH_fit = self.c_OH

class PhysicsCorrectElectrolysisModel:
    """Physikalisch identisches Modell zum Hauptmodell mit fitbaren Diffusionskoeffizienten"""
    def __init__(self, params_dict):
        self.update_params(params_dict)
        self.n1F = n1 * F
        self.n2F = n2 * F
        
    def update_params(self, params_dict):
        """Aktualisiert alle Modellparameter inkl. Diffusionskoeffizienten"""
        self.i0_1 = params_dict['i0_1']
        self.i0_2 = params_dict['i0_2']
        self.alpha1 = params_dict['alpha1']
        self.alpha2 = params_dict['alpha2']
        self.A_eff = params_dict['A_eff']
        self.delta = params_dict['delta']
        self.inv_delta = 1.0 / self.delta
        
        # Fitbare Diffusionskoeffizienten (Referenzwerte bei 25°C)
        self.D_CH3OH_ref = params_dict['D_CH3OH_ref']
        self.D_HCOO_ref = params_dict['D_HCOO_ref']
        self.D_CO32_ref = params_dict['D_CO32_ref']
        self.D_OH_ref = params_dict['D_OH_ref']
        
    def solve_currents_MT(self, E_anode, c1, c2, c3, c4, i_total):
        """Löst Ströme mit Massentransport - MIT FITBAREN DIFFUSIONSKOEFFIZIENTEN"""
        max_iter = 30  
        tol = 1e-7
        
        # Eingangsdaten stabilisieren
        c1, c2, c3, c4 = [max(c, 5e-4) for c in [c1, c2, c3, c4]]
        
        # Gleichgewichtspotentiale
        E1_eq_bulk, E2_eq_bulk = calculate_nernst_efficient(c1, c2, c3, c4)
        eta1_bulk = E_anode - E1_eq_bulk
        eta2_bulk = E_anode - E2_eq_bulk
        
        # NUMERISCHE STABILITÄT: Prüfe auf unrealistische Überspannungen
        if abs(eta1_bulk) > 1.5 or abs(eta2_bulk) > 1.5:
            return 0.0, 0.0
        
        # Startschätzung
        i1_old, i2_old = 0.0, 0.0
        
        for iteration in range(max_iter):
            # Diffusionskoeffizienten MIT FITBAREN PARAMETERN
            D_values = calculate_diffusion_coefficients_with_params(
                c4, self.D_CH3OH_ref, self.D_HCOO_ref, self.D_CO32_ref, self.D_OH_ref)
            
            # Oberflächenkonzentrationen
            J1 = i1_old / self.n1F
            J2 = i2_old / self.n2F
            j1 = J1 / self.A_eff
            j2 = J2 / self.A_eff
            
            # Film-Theorie
            c1_surf = max(c1 - j1 * self.delta / (D_values[0] * 1000), 5e-7)
            c2_surf = max(c2 - (-j1 + j2) * self.delta / (D_values[1] * 1000), 5e-7)
            c3_surf = max(c3 - (-j2) * self.delta / (D_values[2] * 1000), 5e-7)
            c4_surf = max(c4 - (5*j1 + 3*j2) * self.delta / (D_values[3] * 1000), 5e-10)
            
            # Oberflächenpotentiale
            E1_eq_surf, E2_eq_surf = calculate_nernst_efficient(c1_surf, c2_surf, c3_surf, c4_surf)
            eta1_surf = E_anode - E1_eq_surf
            eta2_surf = E_anode - E2_eq_surf
            
            # NUMERISCHE STABILITÄT: Butler-Volmer mit Overflow-Schutz
            exp_arg1 = self.alpha1 * n1F_over_RT * eta1_surf
            exp_arg2 = self.alpha2 * n2F_over_RT * eta2_surf
            
            MAX_EXP = 60.0  # stabilisiert
            exp_arg1_c = np.clip(exp_arg1, -MAX_EXP, MAX_EXP)
            exp_arg2_c = np.clip(exp_arg2, -MAX_EXP, MAX_EXP)
            
            j_bv1 = self.i0_1 * np.exp(exp_arg1_c)
            j_bv2 = self.i0_2 * np.exp(exp_arg2_c)
            
            # Diffusionslimits
            c1_m3 = c1 * 1000
            c2_m3 = c2 * 1000
            j_lim1 = self.n1F * D_values[0] * c1_m3 * self.inv_delta
            j_lim2 = self.n2F * D_values[1] * c2_m3 * self.inv_delta
            
            # Koutecky-Levich
            j1 = (j_bv1 * j_lim1) / (j_bv1 + j_lim1)
            j2 = (j_bv2 * j_lim2) / (j_bv2 + j_lim2)
            
            i1 = j1 * self.A_eff
            i2 = j2 * self.A_eff
            
            # Konvergenz
            if abs(i1 - i1_old) < tol and abs(i2 - i2_old) < tol:
                break
                
            i1_old, i2_old = i1, i2
        
        return i1, i2
    
    def find_anode_potential(self, c1, c2, c3, c4, i_total):
        """Findet Anodenpotential für gegebenen Strom"""
        E1_eq, E2_eq = calculate_nernst_efficient(c1, c2, c3, c4)
        
        def objective(E):
            i1, i2 = self.solve_currents_MT(E, c1, c2, c3, c4, i_total)
            return (i1 + i2) - i_total
        
        try:
            E_min = min(E1_eq, E2_eq) - 0.2
            E_max = max(E1_eq, E2_eq) + 1.2
            
            f_min = objective(E_min)
            f_max = objective(E_max)
            
            if f_min * f_max < 0:
                E_anode = brentq(objective, E_min, E_max, xtol=1e-6)
            else:
                E_guess = max(E1_eq, E2_eq) + 0.2
                E_anode = fsolve(objective, E_guess, xtol=1e-6)[0]
                
        except:
            E_anode = max(E1_eq, E2_eq) + 0.2
            
        return E_anode
    
    @with_timeout(90)
    def simulate(self, dataset, t_eval=None):
        """Simuliert das System """
        V = dataset.V
        i_total = dataset.i_total
        
        # Anfangsbedingungen
        x0 = [dataset.c0['MeOH'], dataset.c0['HCOO'], 
              dataset.c0['CO32'], dataset.c0['OH'], dataset.c0['H2O'],
              dataset.c0['HCOO']*V, dataset.c0['CO32']*V, 0]
        
        if t_eval is None:
            t_eval = dataset.t_conc_fit
        
        # Pre-compute
        A_eff_over_V = self.A_eff / V
        inv_A_eff_n1F = 1.0 / (self.A_eff * self.n1F)
        inv_A_eff_n2F = 1.0 / (self.A_eff * self.n2F)
        
        def odes(t, x):
            c1, c2, c3, c4, c5, n_HCOO_total, n_CO32_total, Q_total = x
            c1 = max(c1, 5e-4)
            c2 = max(c2, 5e-4)
            c3 = max(c3, 5e-4) 
            c4 = max(c4, 5e-8)
            c5 = max(c5, 30.0)  # Wasser
            
            # Galvanostatisch
            E_anode = self.find_anode_potential(c1, c2, c3, c4, i_total)
            i1, i2 = self.solve_currents_MT(E_anode, c1, c2, c3, c4, i_total)
            
            # Reaktionsraten
            r1 = i1 * inv_A_eff_n1F
            r2 = i2 * inv_A_eff_n2F
            
            # Konzentrationsänderungen (BPM-korrigiert wie im Hauptmodell)
            r1_vol = r1 * A_eff_over_V
            r2_vol = r2 * A_eff_over_V
                        
            dc1_dt = -r1_vol
            dc2_dt = r1_vol - r2_vol
            dc3_dt = r2_vol
            dc4_dt = (-5*r1_vol - 3*r2_vol) + i_total / (F * V) 
            dc5_dt = 2*r1_vol + r2_vol
            
            # Integrale
            dn_HCOO_dt = r1 * self.A_eff
            dn_CO32_dt = r2 * self.A_eff
            dQ_dt = i_total
            
            return [dc1_dt, dc2_dt, dc3_dt, dc4_dt, dc5_dt,
                   dn_HCOO_dt, dn_CO32_dt, dQ_dt]
        
        # Solver mit angepassten Toleranzen für Speed
        sol = solve_ivp(odes, (0, np.max(t_eval)), x0,
                       t_eval=t_eval,
                       method='Radau',
                       rtol=1e-6,
                       atol=1e-7,
                       max_step=10)
        
        return sol.y[:5]  # Alle Konzentrationen

def objective_function(params, dataset, model, transport_weight_factor=3.0):
    """Zielfunktion für ein Dataset - MIT DIFFUSIONSKOEFFIZIENTEN UND ZEITGEWICHTUNG"""
    params_dict = {
        'i0_1': params[0],
        'i0_2': params[1],
        'alpha1': params[2],
        'alpha2': params[3],
        'A_eff': params[4],
        'delta': params[5],
        'D_CH3OH_ref': params[6],
        'D_HCOO_ref': params[7],
        'D_CO32_ref': params[8],
        'D_OH_ref': params[9]
    }
    
    model.update_params(params_dict)
    
    try:
        # Simulation
        c_sim = model.simulate(dataset)
        
        # NUMERISCHE STABILITÄT: Prüfe auf ungültige Ergebnisse
        if np.any(np.isnan(c_sim)) or np.any(np.isinf(c_sim)):
            return 1e10
        
        # === ZEITGEWICHTUNG: Letztes Viertel höher gewichten ===
        n_points = len(dataset.t_conc_fit)
        time_weights = np.ones(n_points)
        
        # Letztes Viertel der Zeitpunkte identifizieren
        last_quarter_start = int(0.75 * n_points)
        
        # Graduelle Gewichtung im letzten Viertel für Transportlimitierung
        for i in range(last_quarter_start, n_points):
            relative_pos = (i - last_quarter_start) / (n_points - last_quarter_start)
            time_weights[i] = 1.0 + relative_pos * (transport_weight_factor - 1.0)
        
        # Fehlerberechnung NUR für CH3OH, HCOO⁻, CO₃²⁻ (ohne OH⁻)
        errors = []
        species_weights = [1.0, 1.3, 1.3]  # Gewichte für CH₃OH, HCOO⁻, CO₃²⁻
        
        for i, (c_exp, c_sim_i, w) in enumerate([
            (dataset.c_MeOH_fit, c_sim[0], species_weights[0]),
            (dataset.c_HCOO_fit, c_sim[1], species_weights[1]),
            (dataset.c_CO32_fit, c_sim[2], species_weights[2])
        ]):
            # Relativer Fehler MIT ZEITGEWICHTUNG
            rel_error_squared = ((c_sim_i - c_exp) / (c_exp + 0.01))**2
            weighted_error = np.sum(rel_error_squared * time_weights)
            errors.append(weighted_error * w)
        
        # Constraints
        penalty = 0
        if params[0] < 0.1 or params[1] < 0.1:
            penalty += 100
        if params[4] < 0.001 or params[4] > 0.5:
            penalty += 100
        if params[5] < 1e-6 or params[5] > 5e-4:
            penalty += 100
        # Diffusionskoeffizienten-Constraints
        if params[6] < 1e-12 or params[6] > 1e-8:  # D_CH3OH_ref
            penalty += 100
        if params[7] < 1e-12 or params[7] > 1e-8:  # D_HCOO_ref
            penalty += 100
        if params[8] < 1e-12 or params[8] > 1e-8:  # D_CO32_ref
            penalty += 100
        if params[9] < 1e-12 or params[9] > 1e-8:  # D_OH_ref
            penalty += 100
            
        return np.sum(errors) + penalty
        
    except TimeoutException as e:
        print(f"\n⏱️ {e} für {dataset.name}")
        return 1e10
    except Exception as e:
        print(f"\n❌ Fehler bei {dataset.name}: {str(e)[:50]}")
        return 1e10

def objective_function_multi(params, datasets, transport_weight_factor=3.0):
    """Multi-Dataset Zielfunktion mit Diffusionskoeffizienten und Zeitgewichtung"""
    model = PhysicsCorrectElectrolysisModel({
        'i0_1': params[0],
        'i0_2': params[1],
        'alpha1': params[2],
        'alpha2': params[3],
        'A_eff': params[4],
        'delta': params[5],
        'D_CH3OH_ref': params[6],
        'D_HCOO_ref': params[7],
        'D_CO32_ref': params[8],
        'D_OH_ref': params[9]
    })
    
    total_error = 0
    for dataset in datasets:
        error = objective_function(params, dataset, model, transport_weight_factor)
        total_error += error
    
    return total_error / len(datasets)

class LiveMonitorTimeWeighted:
    """Live-Monitoring während des Fittings mit Diffusionskoeffizienten und Zeitgewichtung"""
    def __init__(self, datasets, transport_weight_factor):
        self.datasets = datasets
        self.transport_weight_factor = transport_weight_factor
        self.iteration = 0
        self.best_objective = float('inf')
        self.best_params = None
        self.start_time = time.time()
        
    def callback(self, xk, convergence=None):
        """Callback für Live-Updates mit Diffusionskoeffizienten und Zeitgewichtung"""
        self.iteration += 1
        
        # Parameter
        i0_1, i0_2, alpha1, alpha2, A_eff, delta, D_CH3OH, D_HCOO, D_CO32, D_OH = xk
        
        # Zielfunktion
        try:
            obj_value = objective_function_multi(xk, self.datasets, self.transport_weight_factor)
            
            if obj_value < self.best_objective:
                self.best_objective = obj_value
                self.best_params = xk.copy()  # Speichere beste Parameter
                improvement = " ⬇"
            else:
                improvement = ""
                
        except:
            obj_value = float('inf')
            improvement = ""
        
        # Live-Output
        elapsed = time.time() - self.start_time
        print(f"\rGen {self.iteration:3d} [{elapsed:5.1f}s]: "
              f"i0_1={i0_1:6.1f} | i0_2={i0_2:6.1f} | "
              f"α1={alpha1:.3f} | α2={alpha2:.3f} | "
              f"A={A_eff:.4f} | δ={delta*1e6:3.0f}μm | "
              f"D_MeOH={D_CH3OH:.1e} | D_HCOO={D_HCOO:.1e} | D_CO32={D_CO32:.1e} | D_OH={D_OH:.1e} | "
              f"f={obj_value:.2e}{improvement} | TW={self.transport_weight_factor:.1f}x", end="", flush=True)

def run_fitting(datasets, use_initial_guess=True, transport_weight_factor=3.0):
    """Führt das Fitting mit Diffusionskoeffizienten und Zeitgewichtung durch

    # Parameter-Grenzen (10 Parameter: 6 kinetische + 4 Diffusionskoeffizienten)
    bounds = [
        (65, 120),           # i0_1 [A/m²]
        (4, 20),             # i0_2 [A/m²]
        (0.02, 0.06),         # alpha1 [-]
        (0.06, 0.22),         # alpha2 [-]
        (0.07, 0.18),        # A_eff [m²]
        (1e-5, 1.1e-4),        # delta [m]
        (1e-11, 5e-9),       # D_CH3OH_ref [m²/s] bei 25°C
        (1e-11, 1e-9),       # D_HCOO_ref [m²/s] bei 25°C
        (1e-11, 1e-8),       # D_CO32_ref [m²/s] bei 25°C
        (1e-12, 1e-7),       # D_OH_ref [m²/s] bei 25°C
    ]
    """
    bounds = [
        (1, 15),           # i0_1 [A/m²]
        (0.1, 3),             # i0_2 [A/m²]
        (0.1, 0.5),         # alpha1 [-]
        (0.2, 0.6),         # alpha2 [-]
        (0.07, 0.18),        # A_eff [m²]
        (1e-5, 1.1e-4),        # delta [m]
        (1e-10, 5e-9),       # D_CH3OH_ref [m²/s] bei 25°C
        (1e-11, 5e-9),       # D_HCOO_ref [m²/s] bei 25°C
        (1e-10, 1e-8),       # D_CO32_ref [m²/s] bei 25°C
        (1e-11, 1e-8),       # D_OH_ref [m²/s] bei 25°C
    ]
    # Startwerte basierend auf Elektrolysezelle_V1.1.1 + Literatur-Diffusionskoeffizienten
    initial_guess = [
        2.5,        # i0_1 aus V1.1.1
        1.2,         # i0_2 aus V1.1.1
        0.2,     # alpha1 aus V1.1.1  
        0.242,      # alpha2 aus V1.1.1
        0.08,       # A_eff aus V1.1.1
        9.9e-05,  # delta aus V1.1.1 (157 μm)
        2.2e-10,      # D_CH3OH_ref - Literaturwert
        1.8e-10,    # D_HCOO_ref - reduziert für ionische Spezies
        7.9e-10,    # D_CO32_ref - reduziert für ionische Spezies
        2.6e-9,   # D_OH_ref 
    ]
    
    print(f"\n🔬 PARAMETER-FITTING (10 Parameter) - MIT ZEITGEWICHTUNG")
    print(f"📊 Datasets: {len(datasets)}")
    print(f"🎯 Fittbare Parameter: i0_1, i0_2, α1, α2, A_eff, δ, D_CH3OH, D_HCOO⁻, D_CO3²⁻, D_OH⁻")
    print(f"📈 Gefittete Spezies: CH₃OH, HCOO⁻, CO₃²⁻ (ohne OH⁻)")
    print(f"⚡ CuO-Poren-Effekt: AKTIVIERT (wie im Hauptmodell)")
    print(f"🧪 Diffusionskoeffizienten: ALLE FITBAR (Referenzwerte bei 25°C)")
    print(f"⏱️ ZEITGEWICHTUNG: Letztes Viertel {transport_weight_factor:.1f}x höher gewichtet (Transportlimitierung)")
    

    
    if use_initial_guess:
        print(f"\n🎯 Verwende Startwerte aus Elektrolysezelle_V1.1.1 + Literatur:")
        print(f"   i0_1={initial_guess[0]:.1f}, i0_2={initial_guess[1]:.1f}, "
              f"α1={initial_guess[2]:.3f}, α2={initial_guess[3]:.3f}, "
              f"A={initial_guess[4]:.4f}, δ={initial_guess[5]*1e6:.0f}μm")
        print(f"   D_CH3OH={initial_guess[6]*1e9:.1f}nm²/s, "
              f"D_HCOO={initial_guess[7]*1e9:.2f}nm²/s, "
              f"D_CO32={initial_guess[8]*1e9:.2f}nm²/s, "
              f"D_OH={initial_guess[9]*1e9:.1f}nm²/s")
        
        # Erstelle initiale Population mit Startwerten
        init_pop = np.zeros((15, 10))  # popsize x n_params
        init_pop[0, :] = initial_guess
        
        # Restliche Population: Variation um Startwerte (±20%)
        for i in range(1, 15):
            for j in range(10):
                lower = max(bounds[j][0], initial_guess[j] * 0.8)
                upper = min(bounds[j][1], initial_guess[j] * 1.2)
                init_pop[i, j] = np.random.uniform(lower, upper)
    else:
        print(f"\n🎲 Verwende zufällige Startwerte (Latin Hypercube)")
        init_pop = 'latinhypercube'
    
    # Live-Monitor
    monitor = LiveMonitorTimeWeighted(datasets, transport_weight_factor)
    
    print(f"\n📈 Live Fitting-Fortschritt (mit Zeitgewichtung):")
    
    # Differential Evolution mit direkter Funktion und args
    result = differential_evolution(
        objective_function_multi,
        bounds=bounds,
        args=(datasets, transport_weight_factor),
        maxiter=200,
        popsize=11,
        seed=42,
        polish=True,
        disp=False,
        updating='deferred',
        workers=-1,
        tol=1e-3,
        atol=1e-3,
        init=init_pop,
        callback=monitor.callback
    )
    
    print()  # Neue Zeile
    elapsed_time = time.time() - monitor.start_time
    print(f"\n✅ Fitting abgeschlossen in {elapsed_time:.1f}s")
    print(f"🎯 Beste Zielfunktion: {monitor.best_objective:.2e}")
    print(f"📊 Basierend auf: CH₃OH, HCOO⁻, CO₃²⁻ (OH⁻ ausgeschlossen wegen BPM)")
    print(f"⏱️ Zeitgewichtung: Letztes Viertel {transport_weight_factor:.1f}x höher gewichtet")
    
    return result

def create_time_weight_visualization(dataset, transport_weight_factor=3.0):
    """Visualisiert die Zeitgewichtung für ein Dataset"""
    n_points = len(dataset.t_conc_fit)
    time_weights = np.ones(n_points)
    
    # Letztes Viertel der Zeitpunkte identifizieren
    last_quarter_start = int(0.75 * n_points)
    
    # Graduelle Gewichtung im letzten Viertel
    for i in range(last_quarter_start, n_points):
        relative_pos = (i - last_quarter_start) / (n_points - last_quarter_start)
        time_weights[i] = 1.0 + relative_pos * (transport_weight_factor - 1.0)
    
    # Visualisierung
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Zeitgewichtung
    plt.subplot(2, 1, 1)
    plt.plot(dataset.t_conc_fit/60, time_weights, 'b-', linewidth=2)
    plt.axvline(x=dataset.t_conc_fit[last_quarter_start]/60, color='red', linestyle='--', 
                label=f'Letztes Viertel (t > {dataset.t_conc_fit[last_quarter_start]/60:.1f} min)')
    plt.xlabel('Zeit (min)')
    plt.ylabel('Zeitgewichtung')
    plt.title(f'Zeitgewichtung für {dataset.name}\n(Transportlimitierung höher gewichtet)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Subplot 2: Beispiel-Konzentration mit Gewichtung
    plt.subplot(2, 1, 2)
    plt.plot(dataset.t_conc_fit/60, dataset.c_MeOH_fit, 'bo-', alpha=0.6, label='CH₃OH', markersize=4)
    plt.plot(dataset.t_conc_fit/60, dataset.c_HCOO_fit, 'go-', alpha=0.6, label='HCOO⁻', markersize=4)
    plt.plot(dataset.t_conc_fit/60, dataset.c_CO32_fit, 'ro-', alpha=0.6, label='CO₃²⁻', markersize=4)
    
    # Markiere gewichtete Punkte
    weighted_indices = np.where(time_weights > 1.1)[0]
    plt.scatter(dataset.t_conc_fit[weighted_indices]/60, dataset.c_MeOH_fit[weighted_indices], 
                s=80, color='blue', alpha=0.8, marker='s', label='CH₃OH (höher gewichtet)')
    plt.scatter(dataset.t_conc_fit[weighted_indices]/60, dataset.c_HCOO_fit[weighted_indices], 
                s=80, color='green', alpha=0.8, marker='s', label='HCOO⁻ (höher gewichtet)')
    plt.scatter(dataset.t_conc_fit[weighted_indices]/60, dataset.c_CO32_fit[weighted_indices], 
                s=80, color='red', alpha=0.8, marker='s', label='CO₃²⁻ (höher gewichtet)')
    
    plt.xlabel('Zeit (min)')
    plt.ylabel('Konzentration (mol/L)')
    plt.title('Konzentrationsverläufe mit Gewichtungsmarkierung')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return plt.gcf()

def visualize_results(datasets, fitted_params, transport_weight_factor=3.0):
    """Visualisiert die Fitting-Ergebnisse mit Diffusionskoeffizienten und Zeitgewichtung"""
    model = PhysicsCorrectElectrolysisModel({
        'i0_1': fitted_params[0],
        'i0_2': fitted_params[1],
        'alpha1': fitted_params[2],
        'alpha2': fitted_params[3],
        'A_eff': fitted_params[4],
        'delta': fitted_params[5],
        'D_CH3OH_ref': fitted_params[6],
        'D_HCOO_ref': fitted_params[7],
        'D_CO32_ref': fitted_params[8],
        'D_OH_ref': fitted_params[9]
    })
    
    n_datasets = len(datasets)
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_datasets == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        
        # Simulation mit gefitteten Parametern
        t_sim = np.linspace(0, np.max(dataset.t_conc), 200)
        c_sim = model.simulate(dataset, t_sim)
        
        # Plot Experiment - Gefittete Spezies
        ax.scatter(dataset.t_conc/60, dataset.c_MeOH, alpha=0.6, s=30,
                  label='CH₃OH (exp)', color='blue')
        ax.scatter(dataset.t_conc/60, dataset.c_HCOO, alpha=0.6, s=30,
                  label='HCOO⁻ (exp)', color='green')
        ax.scatter(dataset.t_conc/60, dataset.c_CO32, alpha=0.6, s=30,
                  label='CO₃²⁻ (exp)', color='red')
        
        # OH⁻ mit anderem Marker (nicht gefittet)
        ax.scatter(dataset.t_conc/60, dataset.c_OH, alpha=0.3, s=20, marker='x',
                  label='OH⁻ (exp, nicht gefittet)', color='purple')
        
        # Plot Simulation - Gefittete Spezies
        ax.plot(t_sim/60, c_sim[0], '-', color='blue', linewidth=2,
                label='CH₃OH (sim)', alpha=0.8)
        ax.plot(t_sim/60, c_sim[1], '-', color='green', linewidth=2,
                label='HCOO⁻ (sim)', alpha=0.8)
        ax.plot(t_sim/60, c_sim[2], '-', color='red', linewidth=2,
                label='CO₃²⁻ (sim)', alpha=0.8)
        
        # OH⁻ gestrichelt (nicht gefittet)
        ax.plot(t_sim/60, c_sim[3], '--', color='purple', linewidth=1.5,
                label='OH⁻ (sim, nicht gefittet)', alpha=0.5)
        
        ax.set_xlabel('Zeit (min)')
        ax.set_ylabel('Konzentration (mol/L)')
        ax.set_title(f'{dataset.name}\nI = {dataset.i_total:.2f} A, V = {dataset.V*1000:.1f} mL')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
    
    # Verstecke ungenutzte Subplots
    for idx in range(n_datasets, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f'Fitting-Ergebnisse (inkl. fitbare D-Koeff. + Zeitgewichtung {transport_weight_factor:.1f}x)', y=1.02)
    
    return fig

def validate_results(datasets, fitted_params, transport_weight_factor=3.0):
    """Validiert die Fitting-Ergebnisse mit Diffusionskoeffizienten und Zeitgewichtung"""
    model = PhysicsCorrectElectrolysisModel({
        'i0_1': fitted_params[0],
        'i0_2': fitted_params[1],
        'alpha1': fitted_params[2],
        'alpha2': fitted_params[3],
        'A_eff': fitted_params[4],
        'delta': fitted_params[5],
        'D_CH3OH_ref': fitted_params[6],
        'D_HCOO_ref': fitted_params[7],
        'D_CO32_ref': fitted_params[8],
        'D_OH_ref': fitted_params[9]
    })
    
    print("\n=== VALIDIERUNG (mit fitbaren D-Koeff. + Zeitgewichtung) ===")
    print(f"{'Dataset':<30} {'Fehler':<12} {'CH3OH RMSE':<12} {'HCOO RMSE':<12} {'CO32 RMSE':<12} {'OH RMSE*':<12}")
    print("-" * 92)
    
    for dataset in datasets:
        c_sim = model.simulate(dataset)
        
        # RMSE berechnen
        rmse_ch3oh = np.sqrt(np.mean((c_sim[0] - dataset.c_MeOH_fit)**2))
        rmse_hcoo = np.sqrt(np.mean((c_sim[1] - dataset.c_HCOO_fit)**2))
        rmse_co32 = np.sqrt(np.mean((c_sim[2] - dataset.c_CO32_fit)**2))
        rmse_oh = np.sqrt(np.mean((c_sim[3] - dataset.c_OH_fit)**2))
        
        error = objective_function(fitted_params, dataset, model, transport_weight_factor)
        
        print(f"{dataset.name:<30} {error:<12.2e} {rmse_ch3oh:<12.4f} "
              f"{rmse_hcoo:<12.4f} {rmse_co32:<12.4f} {rmse_oh:<12.4f}")
    
    print("\n* OH⁻ wurde NICHT gefittet (BPM-Einfluss), aber D_OH⁻ wurde gefittet")
    print("  Gefittete Parameter: Kinetik + alle Diffusionskoeffizienten (CH₃OH, HCOO⁻, CO₃²⁻, OH⁻)")
    print(f"  Zeitgewichtung: Letztes Viertel {transport_weight_factor:.1f}x höher gewichtet")

# === HAUPTPROGRAMM ===
if __name__ == "__main__":
    # Pfad-Konfiguration - CSV-Dateien aus dem Skript-Ordner
    data_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"📁 Datenverzeichnis: {data_dir}")
    
    if not os.path.exists(data_dir):
        print("✗ Fitting-Ordner nicht gefunden!")
        exit()
    
    # CSV-Dateien auswählen
    print("\n=== CSV-DATEIEN AUSWÄHLEN ===")
    all_csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not all_csv_files:
        print("✗ Keine CSV-Dateien gefunden!")
        exit()
    
    print("Verfügbare CSV-Dateien:")
    for i, f in enumerate(all_csv_files):
        print(f"  {i+1}. {f}")
    
    print("\nDateien auswählen (Nummern durch Komma getrennt, oder 'alle'):")
    selection = input().strip()
    
    if selection.lower() == 'alle':
        csv_files = all_csv_files
    else:
        try:
            indices = [int(x.strip())-1 for x in selection.split(',')]
            csv_files = [all_csv_files[i] for i in indices if 0 <= i < len(all_csv_files)]
        except:
            print("✗ Ungültige Auswahl!")
            exit()
    
    # Datensätze laden
    print("\n=== LADE EXPERIMENTELLE DATEN ===")
    datasets = []
    
    for csv_file in csv_files:
        filepath = os.path.join(data_dir, csv_file)
        dataset = ElectrolysisDataset(filepath)
        if dataset.valid:
            datasets.append(dataset)
    
    if len(datasets) == 0:
        print("⚠ Keine gültigen Datensätze gefunden!")
        exit()
    
    print(f"\n✓ {len(datasets)} Datensätze erfolgreich geladen")
    
    # Fitting-Parameter abfragen
    print("\n💡 Zeitgewichtungsfaktor für Transportlimitierung (Standard: 3.0): ", end="")
    try:
        weight_factor = float(input().strip() or "3.0")
    except:
        weight_factor = 3.0
    
    print("💡 Startwerte verwenden? (j/n, Standard: j): ", end="")
    use_guess = input().strip().lower() != 'n'
    
    # Fitting durchführen
    result = run_fitting(datasets, use_initial_guess=use_guess, 
                        transport_weight_factor=weight_factor)
    
    if result.success:
        fitted_params = result.x
        
        print(f"\n=== GEFITTETE PARAMETER ===")
        print(f"KINETIK:")
        print(f"  i0_1   = {fitted_params[0]:.2f} A/m²")
        print(f"  i0_2   = {fitted_params[1]:.2f} A/m²")
        print(f"  alpha1 = {fitted_params[2]:.4f}")
        print(f"  alpha2 = {fitted_params[3]:.4f}")
        print(f"SYSTEM:")
        print(f"  A_eff  = {fitted_params[4]:.6f} m²")
        print(f"  delta  = {fitted_params[5]*1e6:.1f} μm")
        print(f"DIFFUSION (25°C Referenzwerte):")
        print(f"  D_CH3OH = {fitted_params[6]*1e9:.2f} nm²/s ({fitted_params[6]:.2e} m²/s)")
        print(f"  D_HCOO  = {fitted_params[7]*1e9:.2f} nm²/s ({fitted_params[7]:.2e} m²/s)")
        print(f"  D_CO32  = {fitted_params[8]*1e9:.2f} nm²/s ({fitted_params[8]:.2e} m²/s)")
        print(f"  D_OH    = {fitted_params[9]*1e9:.2f} nm²/s ({fitted_params[9]:.2e} m²/s)")
        print(f"ZEITGEWICHTUNG:")
        print(f"  Transport-Faktor = {weight_factor:.1f}x (letztes Viertel)")
        print(f"\nZielfunktion: {result.fun:.2e}")
        print(f"Iterationen: {result.nit}")
        print(f"Funktionsaufrufe: {result.nfev}")
        
        # Visualisierung
        print("\nErstelle Plots...")
        fig = visualize_results(datasets, fitted_params, weight_factor)
        
        # Validierung
        validate_results(datasets, fitted_params, weight_factor)
        
        # Parameter für Hauptmodell ausgeben
        print(f"\n=== PARAMETER FÜR HAUPTMODELL (Elektrolysezelle_V1.1.1.py) ===")
        print(f"# === KINETISCHE PARAMETER ===")
        print(f"i0_1 = {fitted_params[0]:.2f}       # A/m² (Austauschstromdichte R1)")
        print(f"i0_2 = {fitted_params[1]:.2f}       # A/m² (Austauschstromdichte R2)")
        print(f"alpha1 = {fitted_params[2]:.4f}    # Transferkoeffizient R1")
        print(f"alpha2 = {fitted_params[3]:.4f}    # Transferkoeffizient R2")
        print(f"\n# === SYSTEM-PARAMETER ===")
        print(f"A_eff = {fitted_params[4]:.6f}    # m² (Effektive Elektrodenfläche)")
        print(f"\n# === MASSENTRANSPORT PARAMETER ===")
        print(f"delta_diff = {fitted_params[5]:.2e}  # m Diffusionsschichtdicke")
        print(f"\n# === DIFFUSIONSKOEFFIZIENTEN (25°C Referenzwerte) ===")
        print(f"# Diese ersetzen die festen Werte in D_ref:")
        print(f"D_ref = {{")
        print(f"    'CH3OH': {fitted_params[6]:.2e},   # Methanol (gefittet)")
        print(f"    'HCOO': {fitted_params[7]:.2e},    # Formiat (gefittet)")
        print(f"    'CO32': {fitted_params[8]:.2e},    # Carbonat (gefittet)")
        print(f"    'OH': {fitted_params[9]:.2e},      # Hydroxid (gefittet)")
        print(f"    'H2O': 2.3e-9      # Wasser (fest)")
        print(f"}}")
        print(f"\n# === ZEITGEWICHTUNG INFO ===")
        print(f"# Fitting wurde mit {weight_factor:.1f}x Gewichtung für das letzte Viertel durchgeführt")
        print(f"# (Transportlimitierung höher gewichtet)")
        
        plt.show()
        
    else:
        print(f"\n⚠ Fitting fehlgeschlagen!")
        print(f"  Grund: {result.message}")