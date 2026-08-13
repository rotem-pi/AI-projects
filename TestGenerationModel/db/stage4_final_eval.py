"""Final evaluation of the guarded band (v2) + rate guardrail:
coverage and accuracy on the representative 2,000-series sample, refreshed
side-by-side examples, and a regenerated HTML report.

Guardrail rule: a series whose realized flag rate in the evaluation month
exceeds FALLBACK_RATE (with at least FALLBACK_MIN_FLAGS flags) falls back to
its existing test (or to no test) and is queued for refit. Applied here as
measurement, exactly as it would apply in production.
"""

import base64
import html as html_mod
import io
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

from research_negative_bounds import clean_quantile  # noqa: E402
from stage2_paired_fix import rebuild_sample  # noqa: E402
from visual_side_by_side import (  # noqa: E402
    DB, EVAL_DAYS, LINK_SQL, TESTS_SQL, VALUES_SQL, incumbent_bounds,
)

FLOOR, OUTLIER_K = 0.10, 8.0
ALPHA_SIDE = 0.001          # per-side miss budget (total 0.2%)
ALPHA = 2 * ALPHA_SIDE
TRAILING_WINDOW = 3
CAL_DAYS_BAND = 56
MIN_CAL_BAND = 10
# Const carve-out: constancy must hold over the metric's ENTIRE causal
# history so far, not just the recent calibration window - a short lull
# (paused pipeline, quiet stretch) in an otherwise-variable metric must not
# qualify. MIN_CONST_LOOKBACK is deliberately well above MIN_CAL_BAND so a
# few flat weeks can't trigger it; only a long, unbroken flat history can.
MIN_CONST_LOOKBACK = 40


def band_with_flag(ts, y):
    """Final band: per weekly refresh, if the metric's entire causal history
    so far is exactly constant (and long enough to trust), emit a Const test
    (value must equal X); otherwise the asymmetric guarded band. Lower bound
    clamped at 0 for de-facto non-negative series.
    Returns (lower, upper, const_mask) where const_mask is a per-point bool
    array marking exactly which points were governed by Const - NOT a single
    series-wide flag, since a series can switch from Const to the regular
    band (once it stops being constant) partway through its history. Any
    caller that needs a single "was this series on Const during the eval
    window" answer must intersect const_mask with its own eval_mask, not
    treat the return as series-wide."""
    predicted = np.full(len(y), np.nan)
    for t in range(1, len(y)):
        predicted[t] = np.median(y[max(0, t - TRAILING_WINDOW):t])
    scale = np.maximum(np.abs(predicted), 1e-9)
    rel = (y - predicted) / scale
    lo = np.full(len(y), np.nan)
    hi = np.full(len(y), np.nan)
    const_mask = np.zeros(len(y), dtype=bool)
    week_start, end = ts.min(), ts.max()
    while week_start <= end:
        week_end = week_start + pd.Timedelta(days=7)
        seg = np.asarray((ts >= week_start) & (ts < week_end))
        cal = np.asarray((ts >= week_start - pd.Timedelta(days=CAL_DAYS_BAND))
                         & (ts < week_start))
        lifetime = np.asarray(ts < week_start)  # ALL causal history so far
        if not seg.sum():
            week_start = week_end
            continue
        lifetime_vals = y[lifetime]
        is_const_lifetime = (
            lifetime.sum() >= MIN_CONST_LOOKBACK
            and lifetime_vals.max() == lifetime_vals.min())
        if is_const_lifetime:
            const_val = float(lifetime_vals[0])
            atol = 1e-9 * max(1.0, abs(const_val))
            lo[seg] = const_val - atol
            hi[seg] = const_val + atol
            const_mask[seg] = True
        elif cal.sum() >= MIN_CAL_BAND:
            seg_ok = seg & ~np.isnan(predicted)
            cal_ok = cal & ~np.isnan(predicted)
            if seg_ok.sum() and cal_ok.sum() >= MIN_CAL_BAND:
                down = clean_quantile(np.maximum(-rel[cal_ok], 0), ALPHA_SIDE)
                up = clean_quantile(np.maximum(rel[cal_ok], 0), ALPHA_SIDE)
                if not (np.isnan(down) or np.isnan(up)):
                    down, up = max(down, FLOOR), max(up, FLOOR)
                    lo[seg_ok] = predicted[seg_ok] - down * scale[seg_ok]
                    hi[seg_ok] = predicted[seg_ok] + up * scale[seg_ok]
        week_start = week_end
    if (y >= 0).all():
        lo = np.maximum(lo, 0.0)
    return lo, hi, const_mask


def final_band(ts, y):
    lo, hi, _ = band_with_flag(ts, y)
    return lo, hi
FALLBACK_RATE = 0.03
FALLBACK_MIN_FLAGS = 3
CHUNK = 40
POP_TOTAL = 547_171
INCUMBENT_ELIGIBLE = 71_537
REL_ELIGIBLE = 341_907


def series_metrics(y, lo, hi, eval_mask):
    ok = eval_mask & ~np.isnan(lo)
    if ok.sum() < 5:
        return None
    flags = ((y < lo) | (y > hi))[ok]
    det = {}
    for d in [0.5, 1.0]:
        inj = y[ok] * (1 + d)
        det[d] = float(np.mean((inj < lo[ok]) | (inj > hi[ok])))
    episodes, run = 0, 0
    for f in list(flags) + [False]:
        if f:
            run += 1
        elif run:
            episodes += 1
            run = 0
    return {"n_runs": int(ok.sum()), "n_flagged": int(flags.sum()),
            "episodes": episodes, "det50": det[0.5], "det100": det[1.0]}


def main() -> None:
    pop = pd.read_csv(HERE / "stage2_population.csv")
    pop["asset_value"] = pop["asset_value"].fillna("")
    sample = rebuild_sample(pop)
    print(f"sample: {len(sample)}")

    conn = psycopg2.connect(DB)
    conn.set_session(readonly=True, autocommit=True)
    cur = conn.cursor()
    cur.execute("SET statement_timeout = 300000")

    tested = sample[sample["has_auto_test"]]
    cur.execute(TESTS_SQL, {"mids": [int(m) for m in tested["metric_id"]],
                            "avs": [str(a) for a in tested["asset_value"]]})
    tests = pd.DataFrame(cur.fetchall(), columns=[
        "metric_id", "asset_value", "test_type", "var1", "var2", "var3",
        "metric_type"]).drop_duplicates(["metric_id", "asset_value"])

    rows, showcase_pool = [], []
    recs = sample[["metric_id", "asset_value", "has_auto_test", "metric_type",
                   "tier"]].values
    for i in range(0, len(recs), CHUNK):
        chunk = recs[i:i + CHUNK]
        cur.execute(VALUES_SQL, {"mids": [int(r[0]) for r in chunk],
                                 "avs": [str(r[1]) for r in chunk]})
        vals = pd.DataFrame(cur.fetchall(), columns=[
            "metric_id", "asset_value", "app_pit", "metric_value"])
        vals["asset_value"] = vals["asset_value"].fillna("")
        for r in chunk:
            s = vals[(vals["metric_id"] == r[0])
                     & (vals["asset_value"] == r[1])].sort_values("app_pit")
            if len(s) < 20:
                continue
            ts = pd.DatetimeIndex(s["app_pit"])
            y = s["metric_value"].astype(float).values
            eval_mask = np.asarray(ts >= ts.max() - pd.Timedelta(days=EVAL_DAYS))
            lo, hi, const_mask = band_with_flag(ts, y)
            m = series_metrics(y, lo, hi, eval_mask)
            if m is None:
                continue
            # "carved out" means Const governed the EVAL WINDOW specifically,
            # not merely somewhere in the series' earlier history - a series
            # that outgrew Const months ago and is now on the regular band
            # must not be counted here.
            eval_const_mask = const_mask[eval_mask]
            m.update({"metric_id": r[0], "asset_value": r[1],
                      "has_auto_test": bool(r[2]),
                      "const_carveout": bool(eval_const_mask.mean() > 0.5)})
            rows.append(m)
            entry = {"t_row": None, "ts": ts, "y": y, "v2": (lo, hi), "m": m,
                     "eval_mask": eval_mask, "metric_type": r[3],
                     "tier": r[4], "inc": None, "im": None}
            if r[2]:
                t_match = tests[(tests["metric_id"] == r[0])
                                & (tests["asset_value"] == r[1])]
                if len(t_match):
                    entry["t_row"] = t_match.iloc[0]
            showcase_pool.append(entry)

    df = pd.DataFrame(rows)
    df["rate"] = df["n_flagged"] / df["n_runs"]
    df["fallback"] = (df["rate"] > FALLBACK_RATE) & (
        df["n_flagged"] >= FALLBACK_MIN_FLAGS)
    df.to_csv(HERE / "stage4_results.csv", index=False)
    active = df[~df["fallback"]]

    fallback_share = df["fallback"].mean()
    net_coverage = REL_ELIGIBLE / POP_TOTAL * (1 - fallback_share)
    print(f"\n=== COVERAGE ===")
    print(f"incumbent eligible: {INCUMBENT_ELIGIBLE / POP_TOTAL:.1%}")
    print(f"guarded band eligible: {REL_ELIGIBLE / POP_TOTAL:.1%}")
    print(f"guardrail fallback share (sample): {fallback_share:.1%}")
    print(f"net active coverage: {net_coverage:.1%}")

    cc = df[df["const_carveout"]]
    print(f"\nConst carve-out: {df['const_carveout'].mean():.1%} of series "
          f"({len(cc)}); their flag rate "
          f"{cc['n_flagged'].sum() / max(cc['n_runs'].sum(), 1):.3%}, "
          f"fallback {cc['fallback'].mean():.1%}")

    print(f"\n=== ACCURACY (representative sample, n={len(df)}) ===")
    for name, g in [("pre-guardrail", df), ("post-guardrail (active)", active)]:
        print(f"\n{name} (n={len(g)}):")
        print(f"  runs flagged: {g['n_flagged'].sum() / g['n_runs'].sum():.3%}")
        print(f"  episodes/metric/month: {g['episodes'].mean():.2f}")
        print(f"  det+50%: {g['det50'].mean():.1%}  det+100%: {g['det100'].mean():.1%}")
        print(f"  buckets: >=3%: {(g['rate'] >= 0.03).mean():.1%}  "
              f"1-3%: {((g['rate'] >= 0.01) & (g['rate'] < 0.03)).mean():.1%}  "
              f"<1%: {(g['rate'] < 0.01).mean():.1%}  "
              f"zero: {(g['rate'] == 0).mean():.1%}")

    # incumbent bounds for every tested series in the pool
    for c in showcase_pool:
        c["m"]["rate"] = c["m"]["n_flagged"] / c["m"]["n_runs"]
        if c["t_row"] is None:
            continue
        try:
            ilo, ihi = incumbent_bounds(c["t_row"], list(c["ts"]), c["y"])
        except Exception:
            continue
        im = series_metrics(c["y"], ilo, ihi, c["eval_mask"])
        if im is None:
            continue
        c["inc"] = (ilo, ihi)
        c["im"] = im
        c["im"]["rate"] = im["n_flagged"] / im["n_runs"]

    from app.tests_gen.metrics.metrics_registry import MetricsRegistry
    registry = set(MetricsRegistry.get_instance().registry.keys())

    def caught_big_drop(c) -> bool:
        lo, hi = c["v2"]
        ok = c["eval_mask"] & ~np.isnan(lo)
        breach_low = ok & (c["y"] < lo)
        if not breach_low.any():
            return False
        pred_mid = (lo + hi) / 2
        drops = c["y"][breach_low] < 0.5 * np.abs(pred_mid[breach_low])
        if not drops.any():
            return False
        if c["inc"] is None:
            return True
        ilo, ihi = c["inc"]
        idx = np.where(breach_low)[0]
        return any(not np.isnan(ilo[i]) and ilo[i] <= c["y"][i] <= ihi[i]
                   for i in idx)

    def band_width_ratio(c) -> float:
        if c["inc"] is None:
            return 0.0
        lo, hi = c["v2"]
        ilo, ihi = c["inc"]
        ok = c["eval_mask"] & ~np.isnan(lo) & ~np.isnan(ilo)
        if ok.sum() < 5:
            return 0.0
        v2w = np.median(hi[ok] - lo[ok])
        incw = np.median(ihi[ok] - ilo[ok])
        return float(incw / v2w) if v2w > 0 else 0.0

    def has_survived_spikes(c) -> bool:
        pre = ~c["eval_mask"]
        if pre.sum() < 20:
            return False
        body = np.quantile(np.abs(c["y"][pre]), 0.95)
        return (np.abs(c["y"][pre]) > 5 * max(body, 1e-9)).sum() >= 1 and \
            c["m"]["rate"] < 0.01

    tested = [c for c in showcase_pool if c["im"] is not None]
    untested = [c for c in showcase_pool if c["t_row"] is None]

    def pick(pool, cond, sort_key=None):
        cands = [c for c in pool if cond(c)]
        if not cands:
            return None
        return max(cands, key=sort_key) if sort_key else cands[0]

    cases = [
        ("Case 1: similar alert rate",
         pick(tested, lambda c: c["im"]["rate"] > 0 and c["m"]["rate"] > 0
              and abs(c["im"]["rate"] - c["m"]["rate"]) < 0.005),
         "Equal noise on the same runs; compare which points each flags."),
        ("Case 2: existing test noisier",
         pick(tested, lambda c: c["im"]["rate"] >= 0.03
              and c["m"]["rate"] < 0.01,
              lambda c: c["im"]["rate"] - c["m"]["rate"]),
         "The existing tolerance is miscalibrated for this series; the "
         "guarded band sizes it from observed errors and stays quiet."),
        ("Case 3: guarded band noisier (pre-guardrail)",
         pick(tested, lambda c: c["m"]["rate"] >= 0.03
              and c["im"]["rate"] < 0.01,
              lambda c: c["m"]["rate"] - c["im"]["rate"]),
         "The guarded band fires where the existing test is silent - either "
         "real structural change or a series the guardrail will retire."),
        ("Case 4: guardrail in action (chaotic series)",
         pick(showcase_pool, lambda c: c["m"]["rate"] > 0.10
              and c["m"]["n_runs"] > 50,
              lambda c: c["m"]["n_flagged"]),
         "No band fits this series. In production the rate guardrail "
         "detects the budget breach within days, falls back, and queues a "
         "refit - this is the mechanism that keeps fleet noise at parity."),
        ("Case 5: coverage win - metric type the generator cannot test",
         pick(untested, lambda c: str(c["metric_type"]).lower() not in registry
              and not str(c["metric_type"]).startswith("custom.")
              and 0 < c["m"]["rate"] < 0.02,
              lambda c: c["m"]["det50"]),
         "This metric type is outside the registry allow-list, so the "
         "existing generator can never test it. The guarded band needs no "
         "type-specific configuration and covers it immediately."),
        ("Case 6: genuine collapse caught, existing band slept",
         pick(tested, caught_big_drop, lambda c: c["m"]["det100"]),
         "The value collapsed below half its baseline. The guarded band "
         "flags it; the existing band is too wide to notice."),
        ("Case 7: the quiet majority, and why width still matters",
         pick(tested, lambda c: c["im"]["n_flagged"] == 0
              and c["m"]["n_flagged"] == 0, band_width_ratio),
         "Both methods are silent, which is the typical case. But the "
         "existing band is many times wider, which is invisible until "
         "something breaks: sensitivity to injected anomalies differs "
         "accordingly (see table)."),
        ("Case 8: contamination guard - spikes did not widen the band",
         pick(showcase_pool, has_survived_spikes,
              lambda c: float(np.max(np.abs(c["y"])))),
         "Large spikes sit in this series' calibration window. The "
         "rare-spike exclusion keeps them out of the tolerance, so the band "
         "stays tight instead of inheriting the spikes as 'normal'."),
        ("Case 9: high-frequency pipeline",
         pick(tested, lambda c: c["tier"] == "hf"
              and c["m"]["rate"] < 0.02,
              lambda c: c["m"]["n_runs"]),
         "Hundreds of runs per day: bounds must be cheap and the band "
         "recalibrates into dense data quickly."),
    ]

    sections = []
    for label, c, note in cases:
        if c is None:
            continue
        t_row = c["t_row"]
        panels = [(f"GUARDED BAND (asymmetric, floor {FLOOR:.0%}, rare-spike "
                   f"exclusion, {ALPHA_SIDE:.1%}/side, non-negative clamp, "
                   f"Const carve-out)", c["v2"], c["m"])]
        if t_row is not None and c["inc"] is not None:
            panels.insert(0, (
                f"EXISTING: {t_row['test_type']}(var1={t_row['var1']:.4g}, "
                f"var2={t_row['var2'] or 0:.4g})", c["inc"], c["im"]))
        fig, axes = plt.subplots(1, len(panels),
                                 figsize=(8 * len(panels), 4.6), squeeze=False)
        for ax, (name, (lo, hi), met) in zip(axes[0], panels):
            ts, y = c["ts"], c["y"]
            ax.plot(ts, y, ".", ms=3, color="tab:blue")
            ax.fill_between(ts, lo, hi, color="gray", alpha=0.3)
            breach = ((y < lo) | (y > hi)) & ~np.isnan(lo) & c["eval_mask"]
            ax.plot(ts[breach], y[breach], "rx", ms=8, mew=2)
            ax.axvline(ts.max() - pd.Timedelta(days=EVAL_DAYS), color="orange",
                       ls="--", lw=1)
            ax.set_title(f"{name}\nalerts last 30d: {int(breach.sum())}",
                         fontsize=10)
        buf = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        img64 = base64.b64encode(buf.getvalue()).decode()

        mid = int(c["m"]["metric_id"])
        cur.execute(LINK_SQL, {"mid": mid, "av": str(c["m"]["asset_value"])})
        row = cur.fetchone()
        link = ""
        if row and row[0] is not None:
            url = f"https://app.definity.run/tasks/{row[0]}?tid={row[1]}"
            link = (f'<a class="btn" href="{url}" target="_blank">'
                    f'Open latest task in definity</a>')
        vm = c["m"]
        if c["im"] is not None:
            im = c["im"]
            stats_rows = f"""
      <tr><th></th><th>Existing test</th><th>Guarded band</th></tr>
      <tr><td>Alerts, last 30 days</td>
          <td>{im['n_flagged']} ({im['rate']:.1%})</td>
          <td>{vm['n_flagged']} ({vm['rate']:.1%})</td></tr>
      <tr><td>Detection of injected +50%</td>
          <td>{im['det50']:.0%}</td><td>{vm['det50']:.0%}</td></tr>"""
        else:
            stats_rows = f"""
      <tr><th></th><th>Existing test</th><th>Guarded band</th></tr>
      <tr><td>Alerts, last 30 days</td>
          <td>no test exists</td>
          <td>{vm['n_flagged']} ({vm['rate']:.1%})</td></tr>
      <tr><td>Detection of injected +50%</td>
          <td>-</td><td>{vm['det50']:.0%}</td></tr>"""
        sections.append(f"""
  <section>
    <h2>{html_mod.escape(label)}</h2>
    <p class="meta">metric <code>{mid}</code>
       ({html_mod.escape(str(c['metric_type']))}, {c['tier']} cadence) {link}</p>
    <table>{stats_rows}
    </table>
    <img src="data:image/png;base64,{img64}"/>
    <p class="note">{html_mod.escape(note)}</p>
  </section>""")
    conn.close()

    summary = f"""
  <table>
    <tr><th></th><th>Existing method</th><th>Guarded band + guardrail</th></tr>
    <tr><td>Coverage (eligible share of 547K active series)</td>
        <td>13.1%</td><td>62.5% eligible, {net_coverage:.1%} net active
        (after {fallback_share:.1%} guardrail fallback)</td></tr>
    <tr><td>Runs flagged, last 30 days (representative sample)</td>
        <td>0.95% (paired) / 1.20% (fleet)</td>
        <td>{df['n_flagged'].sum() / df['n_runs'].sum():.2%} pre-guardrail;
        {active['n_flagged'].sum() / active['n_runs'].sum():.2%} with the
        guardrail active ({int(df['fallback'].sum())} sample series fall
        back)</td></tr>
    <tr><td>Detection of +50% / +100% injected anomalies</td>
        <td>30.5% / 38.3% (paired)</td>
        <td>{active['det50'].mean():.1%} / {active['det100'].mean():.1%}
        (active series)</td></tr>
    <tr><td>Blind to a collapse-to-zero</td>
        <td>widespread (bands 2-5x wider than needed)</td>
        <td>4.8% of points (was 37.2% before asymmetric calibration)</td></tr>
  </table>"""

    html_doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Guarded band vs existing tests - final evaluation</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, sans-serif; margin: 2rem auto;
         max-width: 1200px; color: #1f2937; }}
  h1 {{ font-size: 1.5rem; }} h2 {{ font-size: 1.15rem; margin-top: 2.2rem; }}
  .meta {{ color: #6b7280; }} code {{ background: #f3f4f6; padding: 1px 4px; }}
  table {{ border-collapse: collapse; margin: .8rem 0; }}
  th, td {{ border: 1px solid #d1d5db; padding: 6px 12px; text-align: left;
            font-size: .9rem; }}
  th {{ background: #eff4ff; }}
  img {{ max-width: 100%; border: 1px solid #e5e7eb; }}
  .note {{ background: #eff4ff; border-left: 4px solid #1a56db;
           padding: .7rem 1rem; font-size: .92rem; }}
  .btn {{ background: #1a56db; color: #fff; padding: 4px 10px;
          border-radius: 6px; text-decoration: none; font-size: .85rem;
          margin-left: .8rem; }}
</style></head><body>
<h1>Guarded band vs existing tests: final evaluation</h1>
<p>Guarded band (final variant): median-of-3 baseline; down-side and up-side
   tolerances calibrated separately from observed errors ({ALPHA_SIDE:.1%}
   miss budget per side) with rare-spike exclusion and a 10% tolerance
   floor; lower bound clamped at 0 for non-negative metrics; constant
   calibration windows emit an exact-equality Const test instead of a band;
   weekly recalibration; rate guardrail (fallback above {FALLBACK_RATE:.0%}
   realized alert rate).</p>
{summary}
{''.join(sections)}
</body></html>"""
    (HERE / "guarded_band_report.html").write_text(html_doc)
    print(f"\nsaved {HERE / 'guarded_band_report.html'}")


if __name__ == "__main__":
    main()
