import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score

plt.style.use('seaborn-v0_8-whitegrid')

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path('..').resolve() / 'dp_arve'
DATA_DIR = BASE_DIR / 'data/Data Fornisseurs'

obs_path = DATA_DIR / 'ARVE/2170_Abfluss_10-Min-Mittel_1999-01-01_2024-12-31.csv'
ml_path = DATA_DIR / 'Hydrique_model/Archive prévisions Hydrique ML Arve-Bout du Monde.json'
hyd_path = DATA_DIR / 'Hydrique_model/Archive prévisions Hydrique hydrique-curve Arve-Bout du Monde.json'
cnr_path = DATA_DIR / 'SIG-CNR_model/Previsions_CNR_20_25.csv'
ofev_path = BASE_DIR / 'outputs/OFEV_probabilistic/station2170_q_quantiles_by_model.csv'

output_dir = BASE_DIR / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

out_matrice = output_dir / "Confusion Matrix"
out_matrice.mkdir(exist_ok=True)

LEAD_TIMES = [6,12,24,36,48]
FLOOD_THR = 400

# ============================================================
# METRICS
# ============================================================

def nse(obs, sim):
    return 1 - np.sum((obs-sim)**2) / np.sum((obs-np.mean(obs))**2)

def kge(obs, sim):
    r = np.corrcoef(obs, sim)[0,1]
    alpha = np.std(sim)/np.std(obs)
    beta = np.mean(sim)/np.mean(obs)
    return 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)

def peak_error(obs, sim):
    return (np.max(sim)-np.max(obs))/np.max(obs)*100

def peak_timing(obs, sim, index):
    t_obs = index[np.argmax(obs)]
    t_sim = index[np.argmax(sim)]
    return (t_sim - t_obs).total_seconds()/3600

def volume_error(obs, sim):
    return (np.sum(sim)-np.sum(obs))/np.sum(obs)*100

def rmse(obs, sim):
    return np.sqrt(np.mean((obs - sim)**2))

def relative_volume_error(obs, sim):
    return (np.sum(sim) - np.sum(obs)) / np.sum(obs) * 100

def mape_high_flows(obs, sim, thr):
    mask = obs > thr
    if np.sum(mask) == 0:
        return np.nan
    return 100 * np.mean(np.abs((obs[mask] - sim[mask]) / obs[mask]))

# ============================================================
# EVENT DETECTION
# ============================================================
"""
def get_flood_events(series, thr):
    mask = series > thr
    groups = (mask != mask.shift()).cumsum()
    events = []

    for _, g in series[mask].groupby(groups):
        events.append((g.index.min(), g.index.max()))

    return events
"""
def get_flood_events(series, thr, gap_hours=12):

    mask = series > thr

    # détecte les transitions (entrée/sortie de crue)
    groups = (mask != mask.shift()).cumsum()

    raw_events = []
    for _, g in series[mask].groupby(groups):
        raw_events.append((g.index.min(), g.index.max()))

    # ============================================================
    # FUSION DES EVENEMENTS SI GAP < gap_hours
    # ============================================================

    if len(raw_events) == 0:
        return []

    merged = [raw_events[0]]

    for start, end in raw_events[1:]:

        prev_start, prev_end = merged[-1]

        gap = (start - prev_end).total_seconds() / 3600

        if gap <= gap_hours:
            # fusion
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    return merged

def event_confusion_matrix(df, thr):
    obs_events = get_flood_events(df["Q_obs"], thr)
    sim_events = get_flood_events(df.iloc[:,1], thr)

    TP, FP, FN = 0, 0, 0

    # TP / FN (événements observés)
    for start, end in obs_events:
        if df.loc[start:end].iloc[:,1].max() > thr:
            TP += 1
        else:
            FN += 1

    # FP (événements simulés non observés)
    for start, end in sim_events:
        if df.loc[start:end]["Q_obs"].max() <= thr:
            FP += 1

    return TP, FP, FN

def plot_confusion_matrix(TP, FP, FN, model, label, output_dir):

    cm = np.array([[0, FP],
                   [FN, TP]])

    total = TP + FP + FN
    cm_norm = cm / total if total > 0 else cm

    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    labels = [["TN (NA)", "FP"],
              ["FN", "TP"]]

    for i in range(2):
        for j in range(2):
            txt = "NA" if (i==0 and j==0) else f"{cm[i,j]}\n({cm_norm[i,j]:.2f})"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="black",
                bbox=dict(
                    facecolor="white",
                    edgecolor="white",
                    boxstyle="round,pad=0.3",
                    alpha=0.9
                )
            )
    
    ax.grid(False)
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_xticklabels(["No Flood","Flood"])
    ax.set_yticklabels(["No Flood","Flood"])

    ax.set_ylabel("Observed")
    ax.set_xlabel("Predicted")

    ax.set_title(f"{model} - {label} h")

    plt.colorbar(im, ax=ax, shrink=0.65)
    plt.tight_layout()

    plt.savefig(output_dir / f"CM_{model}_{label}.png", dpi=300)
    plt.close()

def event_scores(TP, FP, FN):
    POD = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    #POD probabilty of detection -> 1 si crues sont détecté, 0 si aucune
    FAR = FP / (TP + FP) if (TP + FP) > 0 else np.nan
    #FAR false alarm ratio -> 0 si aucune fausse alerte, 1 si toutes les alertes sont fausses
    CSI = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else np.nan
    #CSI critical success index -> 1 si parfait et 0 si nul
    return POD, FAR, CSI

def peak_timing_events(df, thr):

    events = get_flood_events(df["Q_obs"], FLOOD_THR, gap_hours=12)
    errors = []

    for start, end in events:

        sub = df.loc[start:end]

        if len(sub) < 3:
            continue

        obs = sub["Q_obs"].values
        sim = sub.iloc[:,1].values
        idx = sub.index

        if np.all(np.isnan(sim)):
            continue

        t_obs = idx[np.argmax(obs)]
        t_sim = idx[np.argmax(sim)]

        err = (t_sim - t_obs).total_seconds()/3600
        errors.append(err)

    return np.mean(errors) if len(errors) > 0 else np.nan

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
# LOAD ML
# ============================================================

with open(ml_path) as f:
    data = json.load(f)

records = []
for ftime, ts in data["ForecastFirstDate2Timeseries"].items():
    for vtime, q in ts.items():
        records.append([ftime, vtime, q])

ml = pd.DataFrame(records, columns=["forecast_time","valid_time","Q_ml"])
ml["forecast_time"] = pd.to_datetime(ml["forecast_time"])
ml["valid_time"] = pd.to_datetime(ml["valid_time"])
ml["lead_time_h"] = (ml["valid_time"]-ml["forecast_time"]).dt.total_seconds()/3600
ml = ml.set_index("valid_time")

# ============================================================
# LOAD HYD
# ============================================================

with open(hyd_path) as f:
    data = json.load(f)

records = []
for ftime, ts in data["ForecastFirstDate2Timeseries"].items():
    for vtime, q in ts.items():
        records.append([ftime, vtime, q])

hyd = pd.DataFrame(records, columns=["forecast_time","valid_time","Q_hyd"])
hyd["forecast_time"] = pd.to_datetime(hyd["forecast_time"])
hyd["valid_time"] = pd.to_datetime(hyd["valid_time"])
hyd["lead_time_h"] = (hyd["valid_time"]-hyd["forecast_time"]).dt.total_seconds()/3600
hyd = hyd.set_index("valid_time")

# ============================================================
# LOAD CNR
# ============================================================

cnr = pd.read_csv(cnr_path)

cnr["DateTime_PREV"] = pd.to_datetime(cnr["DateTime_PREV"])
cnr["Date_Prevision"] = pd.to_datetime(cnr["Date_Prevision"])

cnr = cnr.rename(columns={"Q":"Q_cnr"})

cnr["lead_time_h"] = (
    cnr["Date_Prevision"] - cnr["DateTime_PREV"]
).dt.total_seconds()/3600

cnr["valid_time"] = cnr["Date_Prevision"]
cnr = cnr.set_index("valid_time")

# ============================================================
# LOAD OFEV
# ============================================================

ofev = pd.read_csv(ofev_path)
ofev.columns = ofev.columns.str.strip()

# bonnes colonnes maintenant
ofev["issue_time"] = pd.to_datetime(ofev["issue_time"])
ofev["valid_time"] = pd.to_datetime(ofev["valid_time"])

# lead time déjà présent normalement → sinon sécurité
if "lead_time_h" not in ofev.columns:
    ofev["lead_time_h"] = (
        ofev["valid_time"] - ofev["issue_time"]
    ).dt.total_seconds()/3600


# MÉDIANE DES MÉDIANES 

ofev = (
    ofev
    .groupby(["valid_time", "lead_time_h"])["Q_p50"]
    .median()
    .reset_index()
)

# format final comme les autres modèles
ofev = ofev.rename(columns={"Q_p50": "Q_ofev"})
ofev = ofev.set_index("valid_time")

# ============================================================
# LOOP LEAD TIMES
# ============================================================

all_metrics = []
event_totals = {
    "Hydrique_ML": np.array([0,0,0]),
    "Hydrique_physique": np.array([0,0,0]),
    "SIG_CNR": np.array([0,0,0]),
    "OFEV": np.array([0,0,0])
}

for LT in LEAD_TIMES:

    print(f"\n========== LT {LT}h ==========")

    out_lt = output_dir / f"leadtime_{LT}h"
    out_lt.mkdir(exist_ok=True)

    ml_lt = ml[np.round(ml["lead_time_h"])==LT][["Q_ml"]]
    hyd_lt = hyd[np.round(hyd["lead_time_h"])==LT][["Q_hyd"]]
    cnr_lt = cnr[np.round(cnr["lead_time_h"])==LT][["Q_cnr"]]
    ofev_lt = ofev[np.round(ofev["lead_time_h"])==LT][["Q_ofev"]]

    

    # ============================================================
    # PERIODE COMMUNE
    # ============================================================

    series = [s for s in [ml_lt, hyd_lt, cnr_lt, ofev_lt] if len(s)>0]

    start_common = max(s.index.min() for s in series)
    end_common   = min(s.index.max() for s in series)

    obs_plot  = obs_h.loc[start_common:end_common]
    ml_plot   = ml_lt.loc[start_common:end_common]
    hyd_plot  = hyd_lt.loc[start_common:end_common]
    cnr_plot  = cnr_lt.loc[start_common:end_common]
    ofev_plot = ofev_lt.loc[start_common:end_common]

    # ============================================================
    # TIMESERIES COMPLETE
    # ============================================================

    plt.figure(figsize=(17,6))

    plt.plot(obs_plot.index, obs_plot["Q_obs"], color="black", label="Observed")
    plt.plot(ml_plot.index, ml_plot["Q_ml"], label="Hydrique ML")
    plt.plot(hyd_plot.index, hyd_plot["Q_hyd"], label="Hydrique physique")
    plt.plot(cnr_plot.index, cnr_plot["Q_cnr"], label="SIG-CNR")
    plt.plot(ofev_plot.index, ofev_plot["Q_ofev"], label="OFEV")

    plt.xlim(start_common, end_common)
    plt.ylim(0,1000)
    plt.legend()
    plt.ylabel("Discharge [m3/s]")
    plt.title(f"Timeseries, lead time {LT}h")
    plt.tight_layout()
    plt.savefig(out_lt / f"timeseries_LT{LT}.png", dpi=200)
    plt.close()

    
    # ============================================================
    # FLOOD EVENTS TIMESERIES
    # ============================================================

    events = get_flood_events(obs_plot["Q_obs"], FLOOD_THR)

    for i,(start,end) in enumerate(events):

        # fenêtre ±2 jours
        start_win = start - pd.Timedelta(days=2)
        end_win   = end + pd.Timedelta(days=2)

        fig, ax = plt.subplots(figsize=(16,6))

        # plots
        ax.plot(obs_plot.loc[start_win:end_win].index,
                obs_plot.loc[start_win:end_win,"Q_obs"],
                color="black", label="Observed")

        if len(ml_plot)>0:
            ax.plot(ml_plot.loc[start_win:end_win].index,
                    ml_plot.loc[start_win:end_win,"Q_ml"],
                    label="Hydrique ML")

        if len(hyd_plot)>0:
            ax.plot(hyd_plot.loc[start_win:end_win].index,
                    hyd_plot.loc[start_win:end_win,"Q_hyd"],
                    label="Hydrique physique")

        if len(cnr_plot)>0:
            ax.plot(cnr_plot.loc[start_win:end_win].index,
                    cnr_plot.loc[start_win:end_win,"Q_cnr"],
                    label="SIG-CNR")

        if len(ofev_plot)>0:
            ax.plot(ofev_plot.loc[start_win:end_win].index,
                    ofev_plot.loc[start_win:end_win,"Q_ofev"],
                    label="OFEV")

        # limites strictes
        ax.set_xlim(start_win +pd.Timedelta(hours=8), end_win - pd.Timedelta(hours=8))
        
        # titre avec debut/fin du plot
        title = (
            f"Flood {start_win.strftime('%d/%m')} - "
            f"{end_win.strftime('%d/%m %Y')}"
        )

        ax.set_title(title)

        # format axe temps
        ax.xaxis.set_major_formatter(
            plt.matplotlib.dates.DateFormatter('%d/%m\n%H:%M')
        )

        ax.legend()
        
        plt.ylabel("Discharge [m3/s]")
        # nom fichier avec dates
        fname = (
            f"flood_"
            f"{start_win.strftime('%Y%m%d')}_"
            f"{end_win.strftime('%Y%m%d')}_"
            f"LT{LT}.png"
        )
        plt.tight_layout()
        plt.savefig(out_lt / fname, dpi=200)
        plt.close()

    # ============================================================
    # METRICS + SCATTER
    # ============================================================

    for name,model in [
        ("Hydrique_ML",ml_plot),
        ("Hydrique_physique",hyd_plot),
        ("SIG_CNR",cnr_plot),
        ("OFEV",ofev_plot)
    ]:

        df = pd.merge(obs_plot, model, left_index=True,
                      right_index=True, how="inner")
        
        df_high = df[df["Q_obs"] > FLOOD_THR]
        
        TP, FP, FN = event_confusion_matrix(df, FLOOD_THR)
        POD, FAR, CSI = event_scores(TP, FP, FN)
        event_totals[name] += np.array([TP, FP, FN])

        plot_confusion_matrix(TP, FP, FN, name, f"LT{LT}", out_matrice)

        # ============================================================
        # EVENT-BASED METRICS 
        # ============================================================

        events = get_flood_events(df["Q_obs"], FLOOD_THR)

        req_list, tp_list, rer_list = [], [], []

        for start, end in events:

            # fenêtre élargie (CRUCIAL)
            start_win = start - pd.Timedelta(days=2)
            end_win   = end + pd.Timedelta(days=2)

            df_event = df.loc[start_win:end_win]

            if len(df_event) < 5:
                continue

            obs_e = df_event["Q_obs"].values
            sim_e = df_event.iloc[:,1].values
            idx_e = df_event.index

            # ignorer si modèle ne simule pas de crue, calcul métrique sur "fausse" crue
            #if np.max(sim_e) < FLOOD_THR:
            #    continue

            req_list.append(peak_error(obs_e, sim_e))
            tp_list.append(peak_timing(obs_e, sim_e, idx_e))
            rer_list.append(relative_volume_error(obs_e, sim_e))

        REQ = np.median(req_list) if len(req_list)>0 else np.nan
        TP  = np.median(tp_list) if len(tp_list)>0 else np.nan
        RER = np.median(rer_list) if len(rer_list)>0 else np.nan

        if len(df)<10:
            continue

        obs_v = df["Q_obs"].values
        sim_v = df.iloc[:,1].values

        if len(df_high) > 5:
            obs_hf = df_high["Q_obs"].values
            sim_hf = df_high.iloc[:,1].values
        else:
            obs_hf, sim_hf = None, None

        all_metrics.append({
            "lead_time": LT,
            "model": name,

            # global metrics
            "NSE": nse(obs_v, sim_v),
            "KGE": kge(obs_v, sim_v),
            "RMSE": rmse(obs_v, sim_v),

            # flood-focused metrics
            "REQ_%": REQ,
            "TP_h": TP,
            "RER_%": RER,

            # high-flow metric
            "MAPE_high_%": mape_high_flows(obs_v, sim_v, FLOOD_THR),

            #truc confusion
            "POD": POD,
            "FAR": FAR,
            "CSI": CSI,

            # HIGH FLOW METRICS (NEW)
            "NSE_high": nse(obs_hf, sim_hf) if obs_hf is not None else np.nan,
            "KGE_high": kge(obs_hf, sim_hf) if obs_hf is not None else np.nan,
            "RMSE_high": rmse(obs_hf, sim_hf) if obs_hf is not None else np.nan,

            "REQ_high_%": peak_error(obs_hf, sim_hf) if obs_hf is not None else np.nan,
            "TP_high_h": peak_timing_events(df, FLOOD_THR),
            "RER_high_%": relative_volume_error(obs_hf, sim_hf) if obs_hf is not None else np.nan
                    })

        pobs_v = df["Q_obs"].values
        sim_v = df.iloc[:,1].values

        # enlever NaN (important pour R²)
        mask = ~np.isnan(obs_v) & ~np.isnan(sim_v)
        obs_v_clean = obs_v[mask]
        sim_v_clean = sim_v[mask]

        r2 = r2_score(obs_v_clean, sim_v_clean)

        plt.figure(figsize=(5,5))

        plt.scatter(obs_v_clean, sim_v_clean, s=5, alpha=0.3)

        m = max(obs_v_clean.max(), sim_v_clean.max())
        plt.plot([0, m], [0, m], "k--")

        plt.xlim(0, 1000)
        plt.ylim(0, 1000)
        plt.xlabel("Observed")
        plt.ylabel(name)

        plt.title(f"{name} model, lead time {LT}h")

        # R² affiché sur le graphe
        plt.text(
            0.05, 0.95,
            f"$R^2$ = {r2:.3f}",
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
        )

        plt.tight_layout()
        plt.savefig(out_lt / f"scatter_{name}_LT{LT}.png", dpi=200)
        plt.close()

for model, vals in event_totals.items():
    TP, FP, FN = vals
    plot_confusion_matrix(TP, FP, FN, model, "ALL_LT", out_matrice)
# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame(all_metrics)
metrics.to_csv(output_dir / "metrics_all_models.csv", index=False)

print(metrics)
print("finished")