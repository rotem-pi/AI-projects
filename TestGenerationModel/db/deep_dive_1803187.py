"""Deep dive: metric 1803187 (meta_numOutputRows, daily) - why are the
guarded-band alerts so close to the band?"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "backend"))

from stage3_guarded_band import guarded_band_bounds  # noqa: E402
from visual_side_by_side import DB, EVAL_DAYS  # noqa: E402

METRIC_ID = 1803187
FLOOR, K, ALPHA = 0.10, 8.0, 0.002

conn = psycopg2.connect(DB)
conn.set_session(readonly=True, autocommit=True)
cur = conn.cursor()
cur.execute("SET statement_timeout = 120000")
cur.execute("""
SELECT app_pit, metric_value FROM (
  SELECT m.app_pit, m.metric_value,
         ROW_NUMBER() OVER (PARTITION BY m.app_pit ORDER BY m.end_time DESC) rn
  FROM metrics m
  WHERE m.metric_id = %(mid)s AND m.metric_value IS NOT NULL
    AND m.app_pit >= now() - interval '88 days'
) x WHERE rn = 1 ORDER BY app_pit""", {"mid": METRIC_ID})
rows = cur.fetchall()
conn.close()

ts = pd.DatetimeIndex([r[0] for r in rows])
y = np.array([float(r[1]) for r in rows])
print(f"points: {len(y)}, span {ts.min():%Y-%m-%d} .. {ts.max():%Y-%m-%d}")
print(f"value stats: min {y.min():.0f}, p25 {np.percentile(y,25):.0f}, "
      f"median {np.median(y):.0f}, p75 {np.percentile(y,75):.0f}, "
      f"max {y.max():.0f}")

lo, hi = guarded_band_bounds(ts, y, FLOOR, K, ALPHA)
eval_mask = np.asarray(ts >= ts.max() - pd.Timedelta(days=EVAL_DAYS))
ok = eval_mask & ~np.isnan(lo)
breach = ((y < lo) | (y > hi)) & ok

pred = (lo + hi) / 2
half = (hi - lo) / 2
print(f"\nflagged points ({int(breach.sum())}):")
print(f"{'date':>12} {'value':>10} {'predicted':>10} {'band':>23} "
      f"{'abs diff':>9} {'rel diff':>8}")
for i in np.where(breach)[0]:
    rel = abs(y[i] - pred[i]) / max(abs(pred[i]), 1e-9)
    print(f"{ts[i]:%Y-%m-%d} {y[i]:>10.0f} {pred[i]:>10.0f} "
          f"[{lo[i]:>9.0f}, {hi[i]:>9.0f}] {y[i]-pred[i]:>9.0f} {rel:>7.0%}")

print(f"\nband half-width on eval days: median {np.median(half[ok]):.0f} "
      f"(= tolerance {np.median(half[ok]/np.maximum(np.abs(pred[ok]),1e-9)):.0%} "
      f"of prediction)")
print(f"non-flagged eval values: median abs deviation "
      f"{np.median(np.abs(y[ok & ~breach] - pred[ok & ~breach])):.0f}")

fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))
axes[0].plot(ts, y, ".-", ms=4, lw=0.5)
axes[0].fill_between(ts, lo, hi, color="gray", alpha=0.3)
axes[0].plot(ts[breach], y[breach], "rx", ms=9, mew=2)
axes[0].set_title("full scale (as in the report)")
body = y[y < np.percentile(y, 98)]
axes[1].plot(ts, y, ".-", ms=5, lw=0.6)
axes[1].fill_between(ts, lo, hi, color="gray", alpha=0.3)
axes[1].plot(ts[breach], y[breach], "rx", ms=10, mew=2.5)
axes[1].axvline(ts.max() - pd.Timedelta(days=EVAL_DAYS), color="orange",
                ls="--", lw=1)
axes[1].set_ylim(body.min() - 0.2 * body.ptp(), body.max() + 0.2 * body.ptp())
axes[1].set_title("zoomed to the data body")
plt.tight_layout()
plt.savefig(HERE / "deep_dive_1803187.png", dpi=110)
print(f"\nsaved {HERE / 'deep_dive_1803187.png'}")
