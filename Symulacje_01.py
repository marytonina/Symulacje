import matplotlib.pyplot as plt
import numpy as np
from sympy import sympify, symbols
import math
import os

#Scenariusze
def build_model_config(scenario: str) -> dict:
    # Flagi
    dna_damage = scenario in ["B", "C", "D"]
    pten_off = scenario in ["C", "D"]
    siRNA = scenario == "D"

    # Bazowe parametry
    p1 = 8.8
    p2 = 440
    p3 = 100
    d1 = 1.3753e-14
    d2 = 1.375e-4
    d3 = 3.0e-5
    k1 = 1.925e-4
    k2 = 100000
    k3 = 150000

    # Modyfikacje według scenariusza
    p2_eff = p2 * 0.02 if siRNA else p2          # siRNA → hamowanie produkcji MDM2
    p3_eff = 0.0 if pten_off else p3             # brak PTEN
    d2_eff = d2 * 0.1 if not dna_damage else d2  # brak DNA damage → wolniejszy rozpad MDM2

    config = {
        'options': {'hours': 48, 'dt': 10},
        'parameters': {
            'p1': p1, 'p2': p2_eff, 'p3': p3_eff,
            'd1': d1, 'd2': d2_eff, 'd3': d3,
            'k1': k1, 'k2': k2, 'k3': k3
        },
        'factors': {
            'p53': {
                'starting_value': 100,
                'function': "p1 - (d1 * p53 * MDMn**2)"
            },
            'MDMcyt': {
                'starting_value': 100,
                'function': "p2 * (p53**4 / (p53**4 + k2**4)) - (k1 * (k3**2 / (k3**2 + PTEN**2)) * MDMcyt) - (d2 * MDMcyt)"
            },
            'MDMn': {
                'starting_value': 100,
                'function': "(k1 * k3**2 * MDMcyt) / (k3**2 + PTEN**2) - (d2 * MDMn)"
            },
            'PTEN': {
                'starting_value': 0 if pten_off else 100,
                'function': "p3 * (p53**4 / (p53**4 + k2**4)) - (d3 * PTEN)"
            }
        }
    }

    return config


#ALGORYTM RK4
def simulate(config):
    factors = config['factors']
    params = config['parameters']
    h = config['options']['dt']
    hours = config['options']['hours']
    steps = int(3600 * hours / h)

    # Inicjalizacja zmiennych i funkcji
    state = {k: v['starting_value'] for k, v in factors.items()}
    factor_keys = sorted(factors.keys())

    functions = {
        k: eval(f"lambda {','.join(factor_keys)}: {sympify(factors[k]['function']).subs({symbols(p): v for p, v in params.items()})}")
        for k in factors
    }

    rk_buffer = [dict() for _ in range(5)]
    results = []

    for _ in range(steps):
        for j, dt_step in enumerate([0, h / 2, h / 2, h]):
            for key in functions:
                rk_buffer[4][key] = state[key] + dt_step * rk_buffer[j - 1].get(key, 0)
            for key in functions:
                rk_buffer[j][key] = functions[key](**rk_buffer[4])
        for key in state:
            state[key] += h / 6 * (rk_buffer[0][key] + 2 * rk_buffer[1][key] + 2 * rk_buffer[2][key] + rk_buffer[3][key])
        results.append(state.copy())

    return results

#Wykresy
def plot_results(results, scenario, hours):
    steps = len(results)
    for key in results[0]:
        plt.plot(range(steps), [r[key] for r in results], label=key)

    plt.xlabel("Czas (h)")
    plt.ylabel("Liczba cząsteczek")
    plt.title(f"Symulacja {hours}h – scenariusz {scenario}")
    plt.xticks(np.linspace(0, steps - 1, 10, dtype=int),
               labels=[str(int(hours * i / 10)) for i in range(10)])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    os.makedirs("wykresy", exist_ok=True)
    plt.savefig(f"wykresy/wykres_{scenario}.jpg")
    plt.clf()

#Głowny kod
if __name__ == "__main__":
    for scen in ["A", "B", "C", "D"]:
        print(f"--- Scenariusz {scen} ---")
        config = build_model_config(scen)
        results = simulate(config)
        plot_results(results, scen, config['options']['hours'])

    print("Wszystkie wykresy zapisane w folderze 'wykresy/'.")

