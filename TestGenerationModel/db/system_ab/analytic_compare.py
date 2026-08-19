"""Add the analytic model (trainer/model@model=analytic) to the GuardedBand
vs existing head-to-head, on the same common comparison points."""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
j = pd.read_csv(HERE / "comparison_common_points.csv", parse_dates=["pit"])
ana = pd.read_csv(
    HERE / "analytic_replay_qualified_points.csv", parse_dates=["pit"]
)[["dataset", "metric_id", "pit", "flagged", "test_type"]].rename(
    columns={"flagged": "ana", "test_type": "ana_test"}
)

before = len(j)
j = j.merge(ana, on=["dataset", "metric_id", "pit"], how="inner")
print(f"common points: {before} -> merged with analytic: {len(j)}")


def stats(col, sub=j):
    a, nm = sub[sub["is_anomaly"]], sub[~sub["is_anomaly"]]
    tp, fp = int(a[col].sum()), int(nm[col].sum())
    acc = 100 * (tp + int((~nm[col]).sum())) / len(sub)
    return (f"tp={tp}/{len(a)} recall={100 * tp / len(a):.1f}% fp={fp} "
            f"fpr={100 * fp / len(nm):.2f}% acc={acc:.2f}% "
            f"prec={100 * tp / max(tp + fp, 1):.1f}%")


print("GuardedBand (full solution):", stats("guarded_band"))
print("band alone (no guardrail):  ", stats("gb"))
print("existing (prophet):         ", stats("inc"))
print("analytic:                   ", stats("ana"))

agree = (j["guarded_band"] == j["ana"])
print(f"\nagreement GuardedBand vs analytic: {100 * agree.mean():.2f}%")
for name, mask in [("both flag", j["guarded_band"] & j["ana"]),
                   ("only GuardedBand", j["guarded_band"] & ~j["ana"]),
                   ("only analytic", ~j["guarded_band"] & j["ana"]),
                   ("both pass", ~j["guarded_band"] & ~j["ana"])]:
    d = j[mask]
    print(f"  {name}: {len(d)} pts, true anomalies: {int(d['is_anomaly'].sum())}")

anoms = j[j["is_anomaly"]]
venn = anoms.groupby(["guarded_band", "ana", "inc"]).size()
print("\nanomaly capture (guarded_band, analytic, existing):")
print(venn.to_string())

print("\nper metric_group_type:")
for grp, g in j.groupby("metric_group_type"):
    ga, gn = g[g["is_anomaly"]], g[~g["is_anomaly"]]
    print(f"  {grp:13s} pts={len(g):6d} anom={len(ga):3d} | "
          f"GBand det={int(ga['guarded_band'].sum()):3d} fpr={100 * gn['guarded_band'].mean():5.2f}% | "
          f"analytic det={int(ga['ana'].sum()):3d} fpr={100 * gn['ana'].mean():5.2f}% | "
          f"exist det={int(ga['inc'].sum()):3d} fpr={100 * gn['inc'].mean():5.2f}%")

g = j[j["ana_test"] == "none"]
ga, gn = g[g["is_anomaly"]], g[~g["is_anomaly"]]
print(f"\nanalytic no-test blind spot: pts={len(g)} anom={len(ga)} | "
      f"GuardedBand det={int(ga['guarded_band'].sum())} fpr={100 * gn['guarded_band'].mean():.2f}%")

print("\nanalytic test-type mix on common points:")
print(j["ana_test"].value_counts().to_string())

j.to_csv(HERE / "comparison_common_points.csv", index=False)
