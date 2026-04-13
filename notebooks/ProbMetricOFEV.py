import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use("seaborn-v0_8-whitegrid")

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path("..").resolve() / "dp_arve"
DATA_DIR = BASE_DIR / "data/Data Fornisseurs"

obs_path = DATA_DIR / "ARVE/2170_Abfluss_10-Min-Mittel_1999-01-01_2024-12-31.csv"
ofev_path = DATA_DIR / "forecasts_OFEV_models.csv"

output_dir = BASE_DIR / "outputs/OFEV_probabilistic"
output_dir.mkdir(parents=True, exist_ok=True)

LEAD_TIMES = [6,12,24,36,48]

HIGH_FLOW = 80  # seuil crue

# ============================================================
# LOAD OBS
# ============================================================

obs = pd.read_csv(obs_path, sep=";", encoding="cp1252", skiprows=8)
obs.columns = obs.columns.str.strip()

obs["Zeitstempel"] = pd.to_datetime(obs["Zeitstempel"])
obs["Wert"] = pd.to_numeric(obs["Wert"], errors="coerce")

obs = obs[["Zeitstempel","Wert"]].rename(columns={"Wert":"Q_obs"})
obs = obs.set_index("Zeitstempel")

obs_h = obs.resample("1h").mean()

# ============================================================
# LOAD OFEV
# ============================================================

ofev = pd.read_csv(ofev_path)

ofev["forecast_date"] = pd.to_datetime(ofev["forecast_date"])
ofev["datetime"] = pd.to_datetime(ofev["datetime"])

ofev["discharge_m3s"] = (
    ofev["discharge_m3s"]
    .astype(str)
    .str.replace(",", ".")
    .astype(float)
)

# ============================================================
# CRPS
# ============================================================

def crps_ensemble(obs, ens):
    """
    ens shape: (n_members,)
    """
    ens = np.sort(ens)
    n = len(ens)

    term1 = np.mean(np.abs(ens - obs))
    term2 = 0

    for i in range(n):
        for j in range(n):
            term2 += np.abs(ens[i]-ens[j])

    term2 = term2 / (2*n*n)

    return term1 - term2

# ============================================================
# LOOP
# ============================================================

results = []

for LT in LEAD_TIMES:

    print("\nLEAD TIME",LT)

    out_lt = output_dir / f"LT_{LT}h"
    out_lt.mkdir(exist_ok=True)

    ofev_lt = ofev[np.round(ofev["lead_time_h"])==LT]

    grouped = ofev_lt.groupby("datetime")

    crps_list = []
    spread_list = []
    skill_list = []
    obs_list = []
    median_list = []

    for t, g in grouped:

        if t not in obs_h.index:
            continue

        ens = g["discharge_m3s"].values
        obs_val = obs_h.loc[t,"Q_obs"]

        if np.isnan(obs_val):
            continue

        # CRPS
        crps = crps_ensemble(obs_val, ens)

        crps_list.append(crps)

        # spread
        spread = np.std(ens)
        spread_list.append(spread)

        # skill
        median = np.median(ens)
        skill = np.abs(median-obs_val)

        skill_list.append(skill)

        obs_list.append(obs_val)
        median_list.append(median)

    crps = np.mean(crps_list)

    spread = np.mean(spread_list)
    skill = np.mean(skill_list)

    spread_skill = spread/skill

    # CRPSS (reference climatology)
    clim = np.mean(np.abs(obs_h["Q_obs"] - obs_h["Q_obs"].mean()))
    crpss = 1 - crps/clim

    # Brier score high flow
    obs_bin = np.array(obs_list) > HIGH_FLOW
    prob = []

    for t,g in grouped:

        if t not in obs_h.index:
            continue

        ens = g["discharge_m3s"].values
        prob.append(np.mean(ens > HIGH_FLOW))

    prob = np.array(prob)

    bs = np.mean((prob - obs_bin)**2)

    results.append({
        "lead":LT,
        "CRPS":crps,
        "CRPSS":crpss,
        "SpreadSkill":spread_skill,
        "Brier":bs
    })

    # ======================================================
    # RANK HISTOGRAM
    # ======================================================

    ranks = []

    for t,g in grouped:

        if t not in obs_h.index:
            continue

        ens = np.sort(g["discharge_m3s"].values)
        obs_val = obs_h.loc[t,"Q_obs"]

        rank = np.searchsorted(ens, obs_val)
        ranks.append(rank)

    plt.figure(figsize=(6,4))

    max_rank = max(ranks)

    plt.hist(ranks, bins=np.arange(0, max_rank+2), edgecolor='black')

    plt.xlim(0, max_rank)

    plt.xlabel("Rank")
    plt.ylabel("Frequency")

    plt.title(f"Rank Histogram - Lead Time {LT} h - OFEV Model")

    plt.margins(x=0)  # pas d'espace gauche/droite

    plt.savefig(out_lt/"rank_histogram.png")
    plt.close()

    # ======================================================
    # RELIABILITY
    # ======================================================

    bins = np.linspace(0,1,11)
    bin_centers = (bins[:-1]+bins[1:])/2

    rel_obs = []
    rel_prob = []

    for i in range(len(bins)-1):

        mask = (prob>=bins[i]) & (prob<bins[i+1])

        if np.sum(mask)==0:
            rel_obs.append(np.nan)
            rel_prob.append(bin_centers[i])
            continue

        rel_obs.append(np.mean(obs_bin[mask]))
        rel_prob.append(bin_centers[i])

    plt.figure(figsize=(5,5))

    plt.plot([0,1],[0,1],'k--', label="Perfect reliability")
    plt.plot(rel_prob, rel_obs,"o-")

    plt.xlabel("Forecast probability")
    plt.ylabel("Observed frequency")

    plt.title(f"Reliability Diagram - Lead Time {LT} h - OFEV Model")

    plt.xlim(0,1)
    plt.ylim(0,1)

    plt.margins(0)

    plt.savefig(out_lt/"reliability.png")
    plt.close()

    # ======================================================
    # SPREAD SKILL
    # ======================================================

    plt.figure(figsize=(5,5))

    plt.scatter(spread_list, skill_list, s=10)

    m = max(max(spread_list), max(skill_list))

    plt.plot([0,m],[0,m],'k--')

    plt.xlabel("Spread (ensemble std)")
    plt.ylabel("Skill (|median - obs|)")

    plt.title(f"Spread-Skill - Lead Time {LT} h - OFEV Model")

    plt.xlim(0, m)
    plt.ylim(0, m)

    plt.margins(0)

    plt.savefig(out_lt/"spread_skill.png")
    plt.close()

# ============================================================
# SAVE
# ============================================================

res = pd.DataFrame(results)
res.to_csv(output_dir/"OFEV_probabilistic_metrics.csv",index=False)

print(res)