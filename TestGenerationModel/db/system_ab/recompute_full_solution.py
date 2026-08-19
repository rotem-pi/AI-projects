"""Recompute the head-to-head with GuardedBand = calibrated band + guardrail
(fallback series-weeks are handled by the existing test, as at cutover)."""

import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
j = pd.read_csv(HERE / "comparison_common_points.csv", parse_dates=["pit"])

sh = j["pit"] - pd.Timedelta(nanoseconds=1)
j["as_of"] = (sh - pd.to_timedelta(sh.dt.weekday, unit="D")).dt.normalize()
rows = j.groupby(["metric_id", "as_of"])["gb"].agg(["size", "sum"]).reset_index()
rows["fallback"] = (rows["sum"] >= 3) & (rows["sum"] > 0.03 * rows["size"])
j = j.merge(rows[["metric_id", "as_of", "fallback"]], on=["metric_id", "as_of"])
j["guarded_band"] = j["gb"].where(~j["fallback"], j["inc"])


def stats(col, sub=j):
    a, nm = sub[sub["is_anomaly"]], sub[~sub["is_anomaly"]]
    tp, fp = int(a[col].sum()), int(nm[col].sum())
    acc = 100 * (tp + int((~nm[col]).sum())) / len(sub)
    return (f"tp={tp}/{len(a)} recall={100 * tp / len(a):.1f}% fp={fp} "
            f"fpr={100 * fp / len(nm):.2f}% acc={acc:.2f}% "
            f"prec={100 * tp / max(tp + fp, 1):.1f}%")


print("points:", len(j), "| fallback points:", int(j["fallback"].sum()),
      f"({100 * j['fallback'].mean():.1f}%) in", int(rows["fallback"].sum()),
      "series-weeks of", len(rows))
print("GuardedBand (full solution):", stats("guarded_band"))
print("band alone (no guardrail):  ", stats("gb"))
print("existing pipeline:          ", stats("inc"))

agree = (j["guarded_band"] == j["inc"])
print(f"\nagreement GuardedBand vs existing: {100 * agree.mean():.2f}%")
for name, mask in [("both flag", j["guarded_band"] & j["inc"]),
                   ("only GuardedBand", j["guarded_band"] & ~j["inc"]),
                   ("only existing", ~j["guarded_band"] & j["inc"]),
                   ("both pass", ~j["guarded_band"] & ~j["inc"])]:
    d = j[mask]
    print(f"  {name}: {len(d)} pts, true anomalies: {int(d['is_anomaly'].sum())}")

print("\nper metric_group_type:")
for grp, g in j.groupby("metric_group_type"):
    ga, gn = g[g["is_anomaly"]], g[~g["is_anomaly"]]
    print(f"  {grp:13s} pts={len(g):6d} anom={len(ga):3d} | "
          f"GBand det={int(ga['guarded_band'].sum()):3d} fpr={100 * gn['guarded_band'].mean():5.2f}% | "
          f"exist det={int(ga['inc'].sum()):3d} fpr={100 * gn['inc'].mean():5.2f}%")

g = j[j["inc_test"] == "none"]
ga, gn = g[g["is_anomaly"]], g[~g["is_anomaly"]]
print(f"\nno-test blind spot: pts={len(g)} anom={len(ga)} | "
      f"GuardedBand det={int(ga['guarded_band'].sum())} fpr={100 * gn['guarded_band'].mean():.2f}%")

g = j[j["gb_test"] == "Const"]
ga, gn = g[g["is_anomaly"]], g[~g["is_anomaly"]]
print(f"Const-served under full solution: pts={len(g)} anom={len(ga)} | "
      f"det={int(ga['guarded_band'].sum())} fpr={100 * gn['guarded_band'].mean():.2f}%")

j.drop(columns=["as_of"]).to_csv(HERE / "comparison_common_points.csv", index=False)
