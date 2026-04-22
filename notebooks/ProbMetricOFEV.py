import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns

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

LEAD_TIMES = [6, 12, 24, 36, 48]
HIGH_FLOW = 80

# ============================================================
# OBSERVATIONS
# ============================================================

obs = pd.read_csv(obs_path, sep=";", encoding="cp1252", skiprows=8)
obs.columns = obs.columns.str.strip()

obs["Zeitstempel"] = pd.to_datetime(obs["Zeitstempel"])
obs["Wert"] = pd.to_numeric(obs["Wert"], errors="coerce")

obs = obs[["Zeitstempel", "Wert"]].rename(columns={"Wert": "Q_obs"})
obs = obs.set_index("Zeitstempel")

obs_h = obs.resample("1h").mean()

# ============================================================
# FORECASTS
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
# METRICS FUNCTIONS
# ============================================================

def crps_ensemble(obs, ens):
    ens = np.sort(ens)
    n = len(ens)

    term1 = np.mean(np.abs(ens - obs))
    term2 = np.mean([np.abs(ens[i] - ens[j]) for i in range(n) for j in range(n)]) / 2

    return term1 - term2


def contingency_metrics(obs_bin, prob, thr=0.5):

    pred = prob > thr

    TP = np.sum((pred == 1) & (obs_bin == 1))
    FP = np.sum((pred == 1) & (obs_bin == 0))
    FN = np.sum((pred == 0) & (obs_bin == 1))

    POD = TP / (TP + FN + 1e-9)
    FAR = FP / (TP + FP + 1e-9)
    CSI = TP / (TP + FP + FN + 1e-9)

    return POD, FAR, CSI

# ============================================================
# STORAGE
# ============================================================

results_global = []
results_models = []

# ============================================================
# MAIN LOOP
# ============================================================

for LT in LEAD_TIMES:

    print("\nLEAD TIME", LT)

    out_lt = output_dir / f"LT_{LT}h"
    out_lt.mkdir(exist_ok=True)

    ofev_lt = ofev[np.round(ofev["lead_time_h"]) == LT]

    grouped_all = ofev_lt.groupby("datetime")

    # ========================================================
    # GLOBAL METRICS
    # ========================================================

    crps_list = []
    spread_list = []
    skill_list = []
    obs_list = []
    median_list = []
    prob_list = []

    for t, g in grouped_all:

        if t not in obs_h.index:
            continue

        ens = g["discharge_m3s"].values
        obs_val = obs_h.loc[t, "Q_obs"]

        if np.isnan(obs_val):
            continue

        crps_list.append(crps_ensemble(obs_val, ens))
        spread_list.append(np.std(ens))

        med = np.median(ens)
        median_list.append(med)
        obs_list.append(obs_val)

        skill_list.append(abs(med - obs_val))
        prob_list.append(np.mean(ens > HIGH_FLOW))

    obs_list = np.array(obs_list)
    prob_list = np.array(prob_list)

    crps = np.mean(crps_list)
    spread = np.mean(spread_list)
    skill = np.mean(skill_list)

    spread_skill = spread / (skill + 1e-9)

    clim = np.mean(np.abs(obs_h["Q_obs"] - obs_h["Q_obs"].mean()))
    crpss = 1 - crps / clim

    obs_bin = obs_list > HIGH_FLOW

    brier = np.mean((prob_list - obs_bin) ** 2)

    POD, FAR, CSI = contingency_metrics(obs_bin, prob_list)

    results_global.append({
        "lead": LT,
        "CRPS": crps,
        "CRPSS": crpss,
        "SpreadSkill": spread_skill,
        "Brier": brier,
        "POD": POD,
        "FAR": FAR,
        "CSI": CSI
    })

    # ========================================================
    # RANK HISTOGRAM (GLOBAL)
    # ========================================================

    ranks = []

    for t, g in grouped_all:

        if t not in obs_h.index:
            continue

        ens = np.sort(g["discharge_m3s"].values)
        obs_val = obs_h.loc[t, "Q_obs"]

        ranks.append(np.searchsorted(ens, obs_val))

    if len(ranks) > 0:

        plt.figure(figsize=(6,4))

        plt.hist(ranks, bins=np.arange(0, max(ranks)+2), edgecolor="black")

        plt.xlabel("Rank")
        plt.ylabel("Frequency")
        plt.title(f"Rank Histogram - {LT}h")

        plt.margins(x=0)

        plt.savefig(out_lt / "rank_histogram.png")
        plt.close()

    # ========================================================
    # RELIABILITY
    # ========================================================

    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    rel_obs, rel_prob = [], []

    for i in range(len(bins) - 1):

        mask = (prob_list >= bins[i]) & (prob_list < bins[i+1])

        if np.sum(mask) == 0:
            rel_obs.append(np.nan)
            rel_prob.append(bin_centers[i])
            continue

        rel_obs.append(np.mean(obs_bin[mask]))
        rel_prob.append(bin_centers[i])

    plt.figure(figsize=(5,5))

    plt.plot([0,1],[0,1],'k--')
    plt.plot(rel_prob, rel_obs, "o-")

    plt.xlabel("Forecast probability")
    plt.ylabel("Observed frequency")
    plt.title(f"Reliability - {LT}h")

    plt.xlim(0,1)
    plt.ylim(0,1)

    plt.margins(0)

    plt.savefig(out_lt / "reliability.png")
    plt.close()

    # ========================================================
    # SPREAD-SKILL
    # ========================================================

    if len(spread_list) > 0:

        plt.figure(figsize=(5,5))

        plt.scatter(spread_list, skill_list, s=10)

        m = max(max(spread_list), max(skill_list))

        plt.plot([0,m],[0,m],'k--')

        plt.xlabel("Spread")
        plt.ylabel("Skill")
        plt.title(f"Spread-Skill - {LT}h")

        plt.xlim(0,m)
        plt.ylim(0,m)

        plt.margins(0)

        plt.savefig(out_lt / "spread_skill.png")
        plt.close()

    # ========================================================
    # MODEL LOOP (COSMO etc.)
    # ========================================================

    for model, df_model in ofev_lt.groupby("model"):

        grouped = df_model.groupby("datetime")

        crps_m, spread_m, skill_m = [], [], []
        obs_m, median_m = [], []

        for t, g in grouped:

            if t not in obs_h.index:
                continue

            ens = g["discharge_m3s"].values
            obs_val = obs_h.loc[t, "Q_obs"]

            if np.isnan(obs_val):
                continue

            crps_m.append(crps_ensemble(obs_val, ens))
            spread_m.append(np.std(ens))

            med = np.median(ens)
            skill_m.append(abs(med - obs_val))

            obs_m.append(obs_val)
            median_m.append(med)

        if len(crps_m) == 0:
            continue

        results_models.append({
            "lead": LT,
            "model": model,
            "CRPS": np.mean(crps_m),
            "Spread": np.mean(spread_m),
            "Skill": np.mean(skill_m),
            "SpreadSkill": np.mean(spread_m) / (np.mean(skill_m) + 1e-9),
            "StabilityIndex": np.mean(spread_m) / (np.mean(crps_m) + 1e-9)
        })

# ============================================================
# SAVE RESULTS
# ============================================================

res_global = pd.DataFrame(results_global)
res_models = pd.DataFrame(results_models)

res_global.to_csv(output_dir / "OFEV_global_metrics.csv", index=False)
res_models.to_csv(output_dir / "OFEV_by_model_metrics.csv", index=False)

print("\nGLOBAL RESULTS")
print(res_global)

print("\nMODEL RESULTS")
print(res_models)

# CLUSTERING
df = res_models.copy()
features = ["CRPS", "Spread", "Skill", "SpreadSkill", "StabilityIndex"]
X = df[features].dropna()
models = df.loc[X.index, "model"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["model"] = models.values

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_pca["cluster"] = kmeans.fit_predict(X_scaled)
df["cluster"] = df_pca["cluster"].values

plt.figure(figsize=(7,5))
sns.scatterplot(
    data=df_pca,
    x="PC1",
    y="PC2",
    hue="cluster",
    style="model",
    s=100
)
for i, row in df_pca.iterrows():
    plt.text(row["PC1"]+0.05, row["PC2"]+0.05, row["model"], fontsize=8)

plt.title("Clustering des modèles météo (PCA + KMeans)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "model_clustering.png")
plt.close()

summary = df.groupby("cluster")[features].mean()
print(summary)