import numpy as np
import matplotlib.pyplot as plt
import json
import os
import random
from scipy.interpolate import interp1d
import sympy as sp

# === WCZYTYWANIE MODELU I PARAMETRÓW ===
print("🔍 Wczytywanie modelu i parametrów...")

with open("params.json") as f:
    param_values = json.load(f)

with open("model.json") as f:
    model_data = json.load(f)

species = model_data["species"]
initial_state = model_data["initial_state"]
model_matrix = model_data["model_matrix"]
rate_laws_str = model_data["rate_laws"]

# === PRZYGOTOWANIE SYMPY ===
species_syms = sp.symbols(species)
param_syms = sp.symbols(list(param_values.keys()) + ["p2_eff", "p3_eff", "d2_eff"])
sym_dict = {s.name: s for s in species_syms + param_syms}

rate_exprs = [sp.sympify(expr, locals=sym_dict) for expr in rate_laws_str]
rate_lambdas = [sp.lambdify(species_syms + param_syms, expr, modules='numpy') for expr in rate_exprs]

# === DEFINICJA FUNKCJI SZYBKOŚCI ===
def rate_function(state, scen_flags):
    p2_eff = param_values["p2"] * 0.02 if scen_flags["siRNA"] else param_values["p2"]
    p3_eff = 0.0 if scen_flags["pten_off"] else param_values["p3"]
    d2_eff = param_values["d2"] * 0.1 if not scen_flags["dna_damage"] else param_values["d2"]

    args = (
        state +
        [param_values[k] for k in param_values] +
        [p2_eff, p3_eff, d2_eff]
    )
    return [f(*args) for f in rate_lambdas]

# === PARAMETRY SYMULACJI ===
t_max =96 * 60  # w minutach
dt = 1
time_grid = np.arange(0, t_max + dt, dt)
n_runs = 3

labels = ["p53", "MDM2_cyt", "MDM2_nuc", "PTEN"]
colors = ["blue", "orange", "green", "red"]

# === SCENARIUSZE ===
scenarios = {
    "A Zdrowa komórka": {"siRNA": False, "pten_off": False, "dna_damage": False},
    "B Uszkodzenie DNA": {"siRNA": False, "pten_off": False, "dna_damage": True},
    "C Nowotwór": {"siRNA": False, "pten_off": True, "dna_damage": True},
    "D Terapia": {"siRNA": True, "pten_off": True, "dna_damage": True},
}

# === SSA OGÓLNY (Gillespie) ===
def gillespie_generic(model_matrix, initial_state, rate_fn, t_max, scen_flags):
    state = initial_state.copy()
    time = 0
    times = [0]
    results = [state.copy()]
    while time < t_max:
        a = rate_fn(state, scen_flags)
        a0 = sum(a)
        if a0 <= 0:
            break
        r1, r2 = random.random(), random.random()
        tau = (1.0 / a0) * np.log(1.0 / r1)
        cumulative_sum = np.cumsum(a)
        reaction_index = np.searchsorted(cumulative_sum, r2 * a0)
        for i in range(len(state)):
            state[i] += model_matrix[reaction_index][i]
        time += tau
        times.append(time)
        results.append(state.copy())
    return np.array(times), np.array(results)

# === NEXT REACTION METHOD ===
def next_reaction_method(model_matrix, initial_state, rate_fn, t_max, scen_flags):
    state = initial_state.copy()
    time = 0
    times = [0]
    results = [state.copy()]
    while time < t_max:
        a = rate_fn(state, scen_flags)
        if all(rate <= 0 for rate in a):
            break
        taus = [np.random.exponential(1 / rate) if rate > 0 else np.inf for rate in a]
        tau = min(taus)
        reaction_index = taus.index(tau)
        for i in range(len(state)):
            state[i] += model_matrix[reaction_index][i]
        time += tau
        times.append(time)
        results.append(state.copy())
    return np.array(times), np.array(results)

# === ŚREDNIA Z N TRAJEKTORII ===
def average_simulation(sim_func, *args):
    all_interp = np.zeros((len(time_grid), 4))
    for _ in range(n_runs):
        t, res = sim_func(*args)
        interp_funcs = [interp1d(t, res[:, i], kind='previous', bounds_error=False, fill_value='extrapolate') for i in range(4)]
        interp_vals = np.array([f(time_grid) for f in interp_funcs]).T
        all_interp += interp_vals
    return all_interp / n_runs

# === FOLDER WYJŚCIOWY ===
output_dir = "wyk"
os.makedirs(output_dir, exist_ok=True)

# === WYKRESY DLA KAŻDEGO SCENARIUSZA I METODY ===
for scen_label, scen_flags in scenarios.items():
    scen_code = scen_label.split()[0]
    print(f"\n▶ Symulacja scenariusza: {scen_label}")

    # Wariant 1: Ogólny SSA
    print("  ⏳ Wariant 1 (ogólny SSA)...")
    avg2 = average_simulation(gillespie_generic, model_matrix, initial_state, rate_function, t_max, scen_flags)
    plt.figure(figsize=(8, 6), dpi=100)
    for i in range(4):
        plt.plot(time_grid / 60, avg2[:, i], label=labels[i], color=colors[i])
    plt.title(f"Wariant 1: Ogólny SSA – {scen_label}")
    plt.xlabel("Czas [h]")
    plt.ylabel("Liczba cząsteczek")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    file_path2 = os.path.join(output_dir, f"wariant1_{scen_code}.png")
    plt.savefig(file_path2)
    plt.close()
    print(f"    ✔ Zapisano {file_path2}")

    # Wariant 2: Next Reaction Method
    print("  ⏳ Wariant 2 (Next Reaction)...")
    avg3 = average_simulation(next_reaction_method, model_matrix, initial_state, rate_function, t_max, scen_flags)
    plt.figure(figsize=(8, 6), dpi=100)
    for i in range(4):
        plt.plot(time_grid / 60, avg3[:, i], label=labels[i], color=colors[i])
    plt.title(f"Wariant 2: Next Reaction – {scen_label}")
    plt.xlabel("Czas [h]")
    plt.ylabel("Liczba cząsteczek")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    file_path3 = os.path.join(output_dir, f"wariant2_{scen_code}.png")
    plt.savefig(file_path3)
    plt.close()
    print(f"    ✔ Zapisano {file_path3}")

print("\n✅ Wszystkie wykresy zostały zapisane do folderu 'wyk'.")
