import numpy as np
import matplotlib.pyplot as plt

#PARAMETRY BIOLOGICZNE (stałe)
p1 = 8.8       # produkcja p53
p2 = 440       # produkcja MDM2 (cytoplazmatycznego)
p3 = 100       # produkcja PTEN

d1 = 1.375e-14 # rozpad p53 przez MDM2_nuc
d2 = 1.375e-4  # rozpad MDM2 (cytoplazma i jądro)
d3 = 3e-5      # rozpad PTEN

k1 = 1.925e-4  # szybkość transportu MDM2 do jądra
k2 = 1e5       # parametr aktywacji MDM2 przez p53
k3 = 1000      # parametr wpływu PTEN na transport MDM2

#FUNKCJA RK4
def rk4(model, y0, t, **kwargs):
    h = t[1] - t[0]  # krok całkowania
    y = np.zeros((len(t), len(y0)))  # macierz wyników
    y[0] = y0
    for i in range(1, len(t)):
        k1_vec = model(y[i-1], t[i-1], **kwargs)
        k2_vec = model(y[i-1] + 0.5*h*k1_vec, t[i-1] + 0.5*h, **kwargs)
        k3_vec = model(y[i-1] + 0.5*h*k2_vec, t[i-1] + 0.5*h, **kwargs)
        k4_vec = model(y[i-1] + h*k3_vec, t[i-1] + h, **kwargs)
        y[i] = y[i-1] + (h/6)*(k1_vec + 2*k2_vec + 2*k3_vec + k4_vec)
    return y

#MODEL BIOLOGICZNY – opisuje zmiany stężeń białek
def model_p53_adjusted_fixed(y, t, siRNA=False, pten_off=False, dna_damage=False):
    p53, MDM2_cyt, MDM2_nuc, PTEN = y

    # Modyfikacja parametrów w zależności od scenariusza
    p2_eff = p2 * 0.02 if siRNA else p2        # terapia siRNA → osłabienie produkcji MDM2
    p3_eff = 0.0 if pten_off else p3           # nowotwór/terapia → brak produkcji PTEN
    d2_eff = d2 * 0.1 if not dna_damage else d2  # brak uszkodzenia DNA → większy rozpad MDM2

    # Równania różniczkowe opisujące zmiany białek
    dp53 = p1 - d1 * p53 * MDM2_nuc**2
    dMDM2_cyt = p2_eff * (p53**4 / (p53**4 + k2**4)) \
                - k1 * (k3**2 / (k3**2 + PTEN**2)) * MDM2_cyt \
                - d2_eff * MDM2_cyt
    dMDM2_nuc = k1 * (k3**2 / (k3**2 + PTEN**2)) * MDM2_cyt - d2_eff * MDM2_nuc
    dPTEN = p3_eff * (p53**4 / (p53**4 + k2**4)) - d3 * PTEN

    return np.array([dp53, dMDM2_cyt, dMDM2_nuc, dPTEN])

#CZAS SYMULACJI (0–48h co 10 minut)
t = np.arange(0, 48, 10/60)

#WARTOŚCI POCZĄTKOWE – realistyczna zdrowa komórka
y0 = [10, 100, 50, 200]  # p53, MDM2_cyt, MDM2_nuc, PTEN

#SCENARIUSZE DO ANALIZY
scenarios = {
    "A Zdrowa komórka": {"siRNA": False, "pten_off": False, "dna_damage": False},
    "B Uszkodzenie DNA": {"siRNA": False, "pten_off": False, "dna_damage": True},
    "C Nowotwór": {"siRNA": False, "pten_off": True, "dna_damage": True},
    "D Terapia": {"siRNA": True, "pten_off": True, "dna_damage": True},
}

labels = ["p53", "MDM2_cyt", "MDM2_nuc", "PTEN"]
results = {}

#SYMULACJA DLA KAŻDEGO SCENARIUSZA
for name, opts in scenarios.items():
    results[name] = rk4(model_p53_adjusted_fixed, y0, t, **opts)

#RYSOWANIE WYKRESÓW
fig, axs = plt.subplots(2, 2, figsize=(14, 8))
axs = axs.flatten()

for ax, (name, y) in zip(axs, results.items()):
    for i in range(4):
        ax.plot(t, y[:, i], label=labels[i])
    ax.set_title(name)
    ax.set_xlabel("Czas [h]")
    ax.set_ylabel("Stężenie")
    ax.set_ylim(0, 750)  # stała skala Y do porównania
    ax.grid(True)
    ax.legend()

plt.suptitle("Model p53 / PTEN / MDM2 – 48h, poprawione efekty scenariuszy", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
