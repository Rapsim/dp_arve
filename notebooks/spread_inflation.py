import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def inflate_q_ensemble(df: pd.DataFrame, factor: float, q_prefix: str = "Q_e") -> pd.DataFrame:
    qcols = [c for c in df.columns if c.startswith(q_prefix)]
    if not qcols:
        raise ValueError(f"no columns starting with {q_prefix} found")

    # compute mean across ensemble (skip NaNs)
    means = df[qcols].mean(axis=1)

    # deviations and inflation
    dev = df[qcols].sub(means, axis=0)
    inflated = means.to_numpy()[:, None] + factor * dev.values

    # put back inflated values (preserve original index)
    df_infl = df.copy()
    df_infl[qcols] = inflated

    # recompute Q quantiles and min/max
    q_min = np.nanmin(inflated, axis=1)
    q_p25 = np.nanpercentile(inflated, 25, axis=1)
    q_p50 = np.nanpercentile(inflated, 50, axis=1)
    q_p75 = np.nanpercentile(inflated, 75, axis=1)
    q_max = np.nanmax(inflated, axis=1)

    # update summary columns if present or create them
    df_infl["Q_min"] = q_min
    df_infl["Q_p25"] = q_p25
    df_infl["Q_p50"] = q_p50
    df_infl["Q_p75"] = q_p75
    df_infl["Q_max"] = q_max

    return df_infl


def main():
    p = argparse.ArgumentParser(description="Inflate ensemble Q spread")
    p.add_argument("--input", "-i", default="outputs/OFEV_probabilistic/station2170_ensemble_wide.csv")
    p.add_argument("--output", "-o", default=None)
    p.add_argument("--factor", "-f", type=float, default=3000, help="multiplicative inflation factor on deviations from ensemble mean (>=0)")
    p.add_argument("--q-prefix", default="Q_e", help="prefix for ensemble Q members")
    args = p.parse_args()

    inp = Path(args.input)
    if args.output:
        outp = Path(args.output)
    else:
        outp = inp.with_name(inp.stem + f"_inflated_f{args.factor:.2f}.csv")

    if not inp.exists():
        raise FileNotFoundError(inp)

    df = pd.read_csv(inp, parse_dates=["issue_time", "valid_time"], low_memory=False)

    df_out = inflate_q_ensemble(df, args.factor, q_prefix=args.q_prefix)

    df_out.to_csv(outp, index=False)
    print("WROTE", outp)


if __name__ == "__main__":
    main()
