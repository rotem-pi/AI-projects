"""Aggregate the resilient_matrices.py checkpoint into B/C matrices.
Safe to run at any time, including mid-progress, to check status."""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

df = pd.read_csv(HERE / "resilient_checkpoint.csv")
df["asset_value"] = df["asset_value"].fillna("")
skipped = df["skip"].sum() if "skip" in df else 0
good = df[df["skip"] == False] if "skip" in df else df  # noqa: E712

print(f"checkpointed rows: {len(df)}  (skipped: {skipped}, usable: {len(good)})")
if len(good) == 0:
    raise SystemExit("no usable rows yet")

print(f"\n=== B. METRICS WITH >=1 ANOMALY LAST MONTH ===")
mm = pd.crosstab(good["old_any"], good["new_any"])
print(mm.to_string())
pct = mm / mm.values.sum() * 100
print(pct.round(1).to_string())

print(f"\n=== C. RUN LEVEL ({good['n_runs'].sum():,} runs) ===")
total = good["n_runs"].sum()
c00, c01, c10, c11 = (good["both_00"].sum(), good["old0_new1"].sum(),
                     good["old1_new0"].sum(), good["both_11"].sum())
print(f"{'':22} {'new: not flagged':>18} {'new: flagged':>14}")
print(f"{'old: not flagged':22} {c00:>12,} ({c00/total:.2%}) "
      f"{c01:>8,} ({c01/total:.2%})")
print(f"{'old: flagged':22} {c10:>12,} ({c10/total:.2%}) "
      f"{c11:>8,} ({c11/total:.2%})")

print(f"\nfallback rate: {good['fell_back'].mean():.1%}")
print(f"old flag rate: {good['old_flagged'].sum() / total:.3%}")
print(f"new flag rate (post-guardrail): {good['new_flagged'].sum() / total:.3%}")
