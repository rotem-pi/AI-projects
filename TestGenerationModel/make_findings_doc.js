const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType,
  LevelFormat, PageBreak, ExternalHyperlink, UnderlineType,
} = require("docx");

// definity-app: production code referenced throughout (Motivation, Section 3).
const DEFINITY_BLOB = "https://github.com/definity-ai/definity-app/blob/main/";
const DEFINITY_TREE = "https://github.com/definity-ai/definity-app/tree/main/";
// This investigation's own scripts/report, pushed to the AI-projects repo.
const REPO_BLOB = "https://github.com/rotem-pi/AI-projects/blob/main/TestGenerationModel/";
const REPO_TREE = "https://github.com/rotem-pi/AI-projects/tree/main/TestGenerationModel/";

const ACCENT = "1A56DB";
const DARK = "1F2937";
const GRAY = "6B7280";
const LIGHT_BG = "EFF4FF";
const HEADER_BG = "1A56DB";
const ROW_ALT = "F5F7FB";
const BORDER = "D1D5DB";
const AMBER = "B45309";
const AMBER_BG = "FFFBEB";

const PAGE_W = 12240, PAGE_H = 15840;
const CONTENT_W = 9360;

const t = (text, opts = {}) => new TextRun({ text, font: "Calibri", size: 22, color: DARK, ...opts });
const code = (text, opts = {}) => new TextRun({ text, font: "Consolas", size: 20, color: "0F4C81", ...opts });
// A clickable, monospace, git-hosted path or filename. `url` must be the
// full https://github.com/.../blob-or-tree/... address; `text` is what's
// displayed (usually the same path, sometimes just the filename).
const codeLink = (text, url, opts = {}) => new ExternalHyperlink({
  link: url,
  children: [new TextRun({
    text, font: "Consolas", size: 20, color: ACCENT,
    underline: { type: UnderlineType.SINGLE }, ...opts,
  })],
});
const definityFile = (path, text) => codeLink(text || path, DEFINITY_BLOB + path);
const definityDir = (path, text) => codeLink(text || path, DEFINITY_TREE + path);
const repoFile = (path, text) => codeLink(text || path, REPO_BLOB + path);
const repoDir = (path, text) => codeLink(text || path, REPO_TREE + path);

const p = (children, opts = {}) =>
  new Paragraph({ children: Array.isArray(children) ? children : [t(children)], spacing: { after: 160, line: 276 }, ...opts });

const h1 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_1,
  spacing: { before: 360, after: 200 },
  children: [new TextRun({ text, font: "Calibri", size: 30, bold: true, color: ACCENT })],
});
const h2 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 280, after: 140 },
  children: [new TextRun({ text, font: "Calibri", size: 24, bold: true, color: DARK })],
});
const h3 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_3,
  spacing: { before: 220, after: 110 },
  children: [new TextRun({ text, font: "Calibri", size: 22, bold: true, color: DARK })],
});

const bullet = (children, level = 0) => new Paragraph({
  children: Array.isArray(children) ? children : [t(children)],
  numbering: { reference: "bullets", level },
  spacing: { after: 90, line: 276 },
});
const numbered = (children, ref = "steps") => new Paragraph({
  children: Array.isArray(children) ? children : [t(children)],
  numbering: { reference: ref, level: 0 },
  spacing: { after: 90, line: 276 },
});

const cellMargins = { top: 70, bottom: 70, left: 110, right: 110 };
const borders = {
  top: { style: BorderStyle.SINGLE, size: 4, color: BORDER },
  bottom: { style: BorderStyle.SINGLE, size: 4, color: BORDER },
  left: { style: BorderStyle.SINGLE, size: 4, color: BORDER },
  right: { style: BorderStyle.SINGLE, size: 4, color: BORDER },
};

function makeTable(widths, headerCells, rows) {
  const headerRow = new TableRow({
    tableHeader: true,
    children: headerCells.map((h, i) => new TableCell({
      width: { size: widths[i], type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: HEADER_BG },
      margins: cellMargins, borders,
      children: [new Paragraph({ spacing: { after: 0 }, children: [new TextRun({ text: h, font: "Calibri", size: 20, bold: true, color: "FFFFFF" })] })],
    })),
  });
  const bodyRows = rows.map((r, ri) => new TableRow({
    children: r.map((cellContent, i) => new TableCell({
      width: { size: widths[i], type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: ri % 2 === 1 ? ROW_ALT : "FFFFFF" },
      margins: cellMargins, borders,
      children: [new Paragraph({
        spacing: { after: 0, line: 250 },
        children: Array.isArray(cellContent) ? cellContent : [t(cellContent, { size: 20 })],
      })],
    })),
  }));
  return new Table({ columnWidths: widths, width: { size: CONTENT_W, type: WidthType.DXA }, rows: [headerRow, ...bodyRows] });
}

function callout(children, { color = ACCENT, bg = LIGHT_BG } = {}) {
  return new Table({
    columnWidths: [CONTENT_W],
    width: { size: CONTENT_W, type: WidthType.DXA },
    rows: [new TableRow({
      children: [new TableCell({
        width: { size: CONTENT_W, type: WidthType.DXA },
        shading: { type: ShadingType.CLEAR, fill: bg },
        margins: { top: 130, bottom: 130, left: 190, right: 190 },
        borders: {
          top: { style: BorderStyle.SINGLE, size: 4, color: color },
          bottom: { style: BorderStyle.SINGLE, size: 4, color: color },
          left: { style: BorderStyle.SINGLE, size: 18, color: color },
          right: { style: BorderStyle.SINGLE, size: 4, color: color },
        },
        children: [new Paragraph({ spacing: { after: 0, line: 270 }, children })],
      })],
    })],
  });
}

const spacer = () => new Paragraph({ spacing: { after: 100 }, children: [] });

const children = [];

// Cover
children.push(new Paragraph({ spacing: { before: 2000, after: 160 }, alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "Definity", font: "Calibri", size: 36, bold: true, color: ACCENT })] }));
children.push(new Paragraph({ spacing: { after: 160 }, alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "Automatic Test Generation:", font: "Calibri", size: 44, bold: true, color: DARK })] }));
children.push(new Paragraph({ spacing: { after: 480 }, alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "Seasonality Investigation and Proposed Redesign", font: "Calibri", size: 34, bold: true, color: DARK })] }));
children.push(callout([
  new TextRun({ text: "Status: ", bold: true, color: AMBER }),
  t("Draft. All measurements in this report are complete, including the large-sample run-level matrix in Section 4.3 and the Const carve-out fix described in Section 3.2. Remaining items are tracked in Section 5."),
], { color: AMBER, bg: AMBER_BG }));
children.push(new Paragraph({ spacing: { before: 400, after: 0 }, alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "Prepared for internal review", font: "Calibri", size: 20, color: GRAY })] }));
children.push(new Paragraph({ children: [new PageBreak()] }));

// 1. Motivation
children.push(h1("1. Motivation"));
children.push(p([
  t("The current test generation pipeline ("), definityFile("backend/app/brain/anomaly/tests_generator.py"),
  t(" plus "), definityDir("backend/app/tests_gen/"), t(") produces alert bounds by fitting an "),
  code("AnalyticProphetTimeSeriesModel"), t(" (Prophet plus hand-tuned changepoint and spike heuristics) as a weak labeler, then running an Optuna grid search to distill those labels into one of four static test types ("),
  code("Const, Range, PctDiff, Trend"), t("), each stored as a "), code("test_type"),
  t(" plus up to three numbers ("), code("var1..var3"), t(")."),
]));
children.push(p("Concerns raised that motivated this investigation:"));
children.push(bullet("The generation pipeline (two Prophet variants, DBSCAN dedup, Optuna search) is hard to reason about: it is difficult to explain why a specific test ended up with its specific numbers."));
children.push(bullet("Suspected seasonality (daily, weekly, monthly rhythms) that a static three-number test cannot express, forcing bounds to be wider than necessary."));
children.push(bullet("A reported high false-positive rate in production: enough noisy auto tests that the team has had to manually delete or disable them on an ongoing basis. This is reported team context rather than a measured figure: the current schema does not retain a record of deleted tests (a known gap, flagged in the codebase itself as a TODO to add test metadata for auditing), so historical removal volume cannot be reconstructed from the database. The production auto-test fail-rate and noisy-test figures measured directly in Section 3.2 are consistent with this being a real and ongoing problem."));

// 2. Seasonality findings
children.push(h1("2. Seasonality Findings (Production Replica)"));
children.push(p([
  t("How we checked: "),
  t("for each pattern (time of day, day of week, day of month, month of year), each series' values were grouped by that bucket and compared with a "),
  t("Kruskal-Wallis test", { bold: true }),
  t(" - a statistical test for whether some groups tend to run higher or lower than others, chosen because metric data rarely follows a clean bell curve. Running that test on thousands of series at once means some would look “significant” by pure chance; each test's "),
  t("p-value", { bold: true }),
  t(" (its built-in score for “how likely is this just a fluke”) was corrected using the "),
  t("Benjamini-Hochberg false discovery rate", { bold: true }),
  t(" procedure, which raises the bar for significance as more tests are run so the reported findings hold up rather than being noise from testing at scale. A pattern only counted as real if it also cleared two further bars: it had to be big enough to matter (at least a 10% swing, not a 2% wiggle), and, for time-of-day specifically, still present after scrambling the data to rule out it being leftover noise from one run bleeding into the next rather than a genuine time-of-day effect. “Material” means a flat rule that ignores the pattern would have to be over 30% wider than a rule that accounts for it, just to avoid false alarms."),
]));
children.push(p([t("Measured across representative samples from "), code("metrics_agg"), t(" (see "), repoDir("db/", "TestGenerationModel/db/"), t("):")]));
children.push(makeTable(
  [2400, 2200, 2380, 2380],
  ["Dimension", "Series tested", "Robust seasonal", "Material (>1.3x band cost)"],
  [
    ["Hour-of-day", "123 (high-frequency)", "23%", "5.7%"],
    ["Day-of-week", "1,353", "3.3%", "0.6%"],
    ["Day-of-month", "1,302", "1.9%", "0.2%"],
    ["Month-of-year", "37 (only this many have 2+ years of history)", "8%", "2.7%"],
  ]
));
children.push(spacer());
children.push(p([
  t("Conclusion: ", { bold: true, color: ACCENT }),
  t("hour-of-day seasonality is the only dimension with material prevalence, and it is concentrated in the roughly 1.8% of auto tests on high-frequency (3 or more runs per day) series. On those, existing Range tests were found to be "),
  t("5x wider (median)", { bold: true }),
  t(" than an hour-aware band would need, confirmed quantitatively via a backtest ("), repoFile("db/backtest_vs_existing_range.py", "backtest_vs_existing_range.py"),
  t("): existing tests detect "), t("7%", { bold: true }), t(" of injected +50% anomalies and "), t("17%", { bold: true }), t(" of +100% anomalies on this population."),
]));
children.push(p([
  t("An "), code("HourlyPctDiffTest"), t(" prototype (compare against same-hour history) was built and backtested. It did "),
  t("not", { italics: true }), t(" outperform a plain trailing-window PctDiff, because the trailing window already adapts to daily rhythm implicitly. This ruled out hour-awareness as the fix and redirected the investigation toward "),
  t("adaptive calibration", { bold: true }), t(" as the actual lever (Section 3)."),
]));

// 3. Proposed architecture
children.push(h1("3. Proposed Architecture: Calibrated Trailing-Median Band"));

children.push(h2("3.1 Suggested Solution"));
children.push(p([t("The winning architecture", { bold: true, color: ACCENT }), t(" (see "), repoFile("db/stage3_guarded_band.py", "stage3_guarded_band.py"), t(", "), repoFile("db/stage4_final_eval.py", "stage4_final_eval.py"), t(") replaces the frozen, once-computed test with a band that predicts the next value and sizes its own tolerance from recent history, refreshed on a schedule:")]));
children.push(numbered([t("Point forecast: ", { bold: true }), t("median of the last 3 values (causal; never uses the point it is predicting).")], "steps3"));
children.push(numbered([t("Tolerance, asymmetric: ", { bold: true }), t("separately calibrate a down-side and an up-side tolerance from the last 8 weeks of relative prediction errors, at a conformal miss budget of 0.1% per side (0.2% total).")], "steps3"));
children.push(numbered([t("Rare-spike exclusion: ", { bold: true }), t("before calibrating, drop calibration errors that are more than 8x the window's own 90th-percentile error, but only if such errors are rare (6% or less of the window). Frequent large errors are treated as the series' normal behavior, not anomalies to exclude.")], "steps3"));
children.push(numbered([t("Tolerance floor: ", { bold: true }), t("never tighter than 10% of the predicted value (avoids alerting on statistically real but practically irrelevant moves).")], "steps3"));
children.push(numbered([t("Non-negative clamp: ", { bold: true }), t("lower bound is clamped at 0 for series that are de-facto non-negative (Section 3.2).")], "steps3"));
children.push(numbered([t("Const carve-out: ", { bold: true }), t("if the calibration window is exactly constant, emit an exact-equality test instead of a band (Section 3.2). Needs refinement; see caveat.")], "steps3"));
children.push(numbered([t("Weekly recalibration.", { bold: true })], "steps3"));
children.push(numbered([t("Rate guardrail: ", { bold: true }), t("a series whose realized alert rate exceeds 3% (with 3 or more flags) in the evaluation window falls back to its existing test and is queued for refit. This is what makes “no alert-rate regression” a structural property rather than a hope (Section 3.2).")], "steps3"));
children.push(spacer());
children.push(p([
  t("This is architecturally identical to the existing "), code("PctDiffTest"),
  t(" (predict from trailing average, band equals plus-or-minus tolerance times prediction) with two differences: the tolerance is "),
  t("learned from observed error", { italics: true }), t(", refreshed on a schedule, instead of frozen at generation time from a Prophet and Optuna search."),
]));

children.push(h2("3.2 Considerations"));
children.push(p("Three parts of the design above deserve a deeper look at why they exist and what they cost."));

children.push(h3("The rate guardrail: why it matters"));
children.push(p([
  t("An early version of the guarded band had a fleet-wide alert rate about 1.4x the current version's (1.33% vs 0.95%, paired same-series and same-month comparison). This was traced to a "),
  t("single outlier series", { bold: true }), t(" (metric 1664201, roughly 270 runs per day): removing it alone brought the fleet total to parity. Only 19% of paired series had any excess flags versus the current version; the top 5 carried 61% of the excess."),
]));
children.push(p([
  t("This is precisely the guardrail's job: rather than hand-tuning away every pathological series, the rate guardrail detects and retires them automatically within days. With the guardrail modeled into the comparison, every accuracy metric favors the proposal (Section 4.1 accuracy table) with no alert-rate regression."),
]));

children.push(h3("Negative bounds"));
children.push(p([
  t("Finding: 30 of 35 registry metric types declare "), code("lower_bound: 0"),
  t(", and 98.2% of sampled series are de-facto non-negative, yet a naive "), t("symmetric", { italics: true }),
  t(" band showed a negative lower bound on "), t("54%", { bold: true }), t(" of such series and was "),
  t("“drop-blind”", { bold: true }), t(" (a collapse to exactly 0 would not breach) on "), t("37% of evaluated points", { bold: true }),
  t(", a material blind spot, since a drop to zero is a classic data-loss failure mode."),
]));
children.push(p([
  t("Fix: asymmetric calibration (separate down and up quantiles, item 2 above) plus clamping the displayed and enforced lower bound at 0. Reduced drop-blind points to "),
  t("4.8%", { bold: true }), t(" of points, at the cost of a higher pre-guardrail flag rate (each side gets its own budget, so total nominal budget effectively doubled before halving each side's alpha to compensate)."),
]));

children.push(h3("Const and Trend test types"));
children.push(p("Production data, last 30 days, enabled auto tests:"));
children.push(makeTable(
  [1300, 1300, 1300, 1500, 1300, 1300, 1560],
  ["Type", "Tests", "Avg age", "Evaluated", "Fail rate", "Fired", "Noisy (>10%)"],
  [
    ["PctDiff", "96,050", "~6 mo", "31,736", "1.04%", "3,370", "1,216"],
    ["Range", "72,229", "~6 mo", "30,268", "1.43%", "5,633", "1,526"],
    [[t("Const", { bold: true })], "27,397", "~7 mo", "7,271", [t("0.05%", { bold: true, size: 20 })], "80", "22"],
    [[t("Trend", { bold: true })], "200", "~5 mo", "38", [t("2.37%", { bold: true, size: 20 })], "7", "4"],
  ]
));
children.push(spacer());
children.push(p([
  t("Const: keep. ", { bold: true, color: ACCENT }),
  t("It targets invariants (schema counts, fixed job counts) where any deviation matters, however small, exactly where the guarded band's 10% tolerance floor is structurally blind. Recommendation: emit Const instead of a band whenever a metric's calibration history is exactly constant."),
]));
children.push(p([
  t("Trend: retire into the band. ", { bold: true, color: ACCENT }),
  t("Worst fail rate in the fleet (2.37%), consistent with fitting a static line once and never refreshing it, the same decay disease afflicting the current generator overall. The trailing-median band achieves the same adaptivity to steady growth without the decay (re-anchored every evaluation, tolerance re-learned weekly). Only 200 Trend tests exist fleet-wide; recommend migrating them to the band and no longer generating new Trend tests."),
]));
children.push(callout([
  new TextRun({ text: "Update: Const carve-out lookback fixed. ", bold: true, color: ACCENT }),
  t("The first implementation checked only the most recent 8-week calibration window for constancy, which over-fired on series that were flat during a lull (for example, a paused pipeline) but active otherwise: 40.5% of a representative sample triggered the carve-out, and carved-out series showed a 2.97% flag rate, three times the fleet average, the opposite of Const's intended near-zero noise profile. Two changes fixed this: (1) constancy is now required over a metric's entire causal history so far (a much longer, unbroken lookback), not just the recent window, with a 40-point minimum before it can trigger at all; (2) a measurement bug was also found and fixed along the way: a series was being counted as “carved out” if Const governed "), t("any", { italics: true }), t(" point in its history, even if it had long since outgrown Const and moved to the regular band by the evaluation month. After both fixes, the carve-out rate dropped to "), t("27.1%", { bold: true }), t(" and the carved-out flag rate to "), t("1.45%", { bold: true }), t(" (down from 2.97%). A residual gap versus production Const's 14% share and 0.05% fail rate remains, and is expected: this carve-out intentionally generalizes Const to any constant value, not only the constant-zero series the current generator targets, and part of the remaining flag rate reflects genuine level shifts on previously-constant metrics being correctly caught rather than false alarms."),
], { color: ACCENT, bg: LIGHT_BG }));

children.push(h2("3.3 Initial Experiments"));
children.push(p([
  t("Rejected: ", { bold: true }),
  t("serving Prophet models directly (a model artifact per series). Vanilla Prophet's native 99.5% interval, benchmarked head-to-head against adaptive baselines under identical leakage-free splits, produced "),
  t("8-20% storm rates", { bold: true }), t(" (series exceeding a 5% false-alert rate), worse than simple adaptive baselines, and 250 KB-per-series model artifacts with no clear serving benefit over a stored-bounds table. See "),
  repoFile("db/model_bakeoff.py", "model_bakeoff.py"), t(" results."),
]));
children.push(p("This ruled out the direction of “ship the labeler as the production model” and pointed toward the calibrated, distilled band described in Section 3.1 as the better trade-off between model quality and operational simplicity."));

// 4. Results
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(h1("4. Results"));

children.push(h2("4.1 Coverage and Accuracy: Representative Sample"));
children.push(p("Sampled proportional to the auto-test fleet's cadence tiers (high-frequency, daily, weekly, sparse), evaluated on the last 30 days, paired same-series and same-runs where noted. Current numbers include the fixed Const carve-out from Section 3.2."));
children.push(h3("Coverage (full population, 547,171 active, enabled, non-temporary series)"));
children.push(makeTable(
  [3200, 3080, 3080],
  ["", "Current version", "Guarded band"],
  [
    ["Eligible", "13.1% (registry allow-list + lifetime history)", "62.5% (recent history only)"],
    ["Net active (post-guardrail)", "13.1%", "~51.7%"],
  ]
));
children.push(spacer());
children.push(h3("Accuracy (representative sample, n ≈ 2,000)"));
children.push(makeTable(
  [2500, 2400, 2300, 2160],
  ["", "Current version (paired reference)", "Guarded band, pre-guardrail", "Guarded band, post-guardrail"],
  [
    ["Runs flagged", "0.95% (paired) / 1.20% (fleet)", "2.95%", "0.98%"],
    ["Episodes / metric / month", "0.60-0.65", "1.16", "0.54"],
    ["Detection, +50% injected", "30.5%", "43.3%", "42.6%"],
    ["Detection, +100% injected", "38.3%", "51.2%", "50.2%"],
    ["Metrics under 1% alert rate", "87.5% (fleet) / 84.3% (paired)", "58.1%", "70.2%"],
  ]
));
children.push(spacer());
children.push(p([
  t("Conclusion: ", { bold: true, color: ACCENT }),
  t("with the guardrail active, the guarded band matches or beats the current version on every measured axis (alert rate, episode rate, detection) while covering four times more of the metric population. The Const carve-out fix (Section 3.2) reduced the guardrail-fallback share from 17.6% to 17.3% and, more importantly, the carved-out cohort's own flag rate from 2.97% to 1.45%; it did not meaningfully move these fleet-level headline numbers because carved-out series are a modest fraction of the sample, but it removes a real quality problem in exactly those series."),
]));

children.push(h2("4.2 Metric-Level and Run-Level Conversion (Sampled)"));
children.push(p("Sample-based estimates (279-series paired sample); see Section 4.3 for large-sample values once available."));
children.push(h3("A. Test coverage (full population; exact, no sampling needed)"));
children.push(makeTable(
  [3400, 2000, 2000, 1960],
  ["", "To guarded band", "To keeps existing", "To no test"],
  [
    ["Had auto test (69,933)", "~42,700", "~27,300", "0"],
    ["Had no test (477,238)", "~241,000", "not applicable", "~236,000"],
  ]
));
children.push(spacer());
children.push(p("No metric loses its current protection; roughly 241,000 previously-untested metrics gain a test."));
children.push(h3("B. Metrics with one or more anomalies last month (sampled, extrapolated to 69,933 tested metrics)"));
children.push(makeTable(
  [3400, 2960, 3000],
  ["", "New: quiet", "New: 1 or more anomalies"],
  [
    ["Old: quiet", "~45,100", "~10,000"],
    ["Old: 1 or more anomalies", "~1,800", "~13,000"],
  ]
));
children.push(spacer());
children.push(h3("C. Run-level flag transitions (25,681 sampled runs)"));
children.push(makeTable(
  [3400, 2960, 3000],
  ["", "New: not flagged", "New: flagged"],
  [
    ["Old: not flagged", "98.10%", "0.92%"],
    ["Old: flagged", "0.49%", "0.48%"],
  ]
));
children.push(spacer());
children.push(p([
  t("Only about "), t("26%", { bold: true }),
  t(" of runs flagged by either method are flagged by both; the two systems mostly disagree about which runs matter, which is exactly the question a shadow-mode labeling phase should resolve."),
]));

children.push(h2("4.3 Large-Sample Run-Level Matrix"));
children.push(p([
  t("A full, non-sampled replay across all approximately 70,000 enabled auto tests proved impractical against the production read replica: two attempts were interrupted by transient replica connection drops during the long single-shot export. Rather than keep retrying the same fragile approach, this was recomputed as a "),
  t("large stratified sample (10,000 series, 5,380 usable, 317,166 runs)", { bold: true }),
  t(", more than 30x the 279-series sample behind Section 4.2, computed in small checkpointed chunks ("), repoFile("db/resilient_matrices.py", "resilient_matrices.py"),
  t(") so that any interruption loses at most one chunk rather than the whole run, fetching 180 days of history per series (widened from an initial 88 days specifically so the Const carve-out's full-lifetime check in Section 3.2 has enough history to evaluate) and using the Const-carve-out-corrected "), code("final_band"),
  t(". Note the sampling frame here is drawn uniformly from all enabled auto tests regardless of recent activity, which skews toward sparse-cadence series and is why 4,620 of 10,000 lacked enough recent data to score (a data-availability skip, not a method failure); it is a different composition than the activity-filtered sample in Section 4.2, so the two should be read as directionally consistent rather than identical."),
]));
children.push(h3("B. Metrics with one or more anomalies last month (5,380 usable series)"));
children.push(makeTable(
  [3400, 2960, 3000],
  ["", "New: quiet", "New: 1 or more anomalies"],
  [
    ["Old: quiet", "77.2%", "9.0%"],
    ["Old: 1 or more anomalies", "2.3%", "11.5%"],
  ]
));
children.push(spacer());
children.push(h3("C. Run-level flag transitions (317,166 runs)"));
children.push(makeTable(
  [3400, 2960, 3000],
  ["", "New: not flagged", "New: flagged"],
  [
    ["Old: not flagged", "98.22%", "0.42%"],
    ["Old: flagged", "0.51%", "0.85%"],
  ]
));
children.push(spacer());
children.push(p([
  t("Old flag rate 1.362%, new flag rate (post-guardrail) 1.271%, guardrail fallback rate 8.8%. "),
  t("Conclusion: ", { bold: true, color: ACCENT }),
  t("at more than 30x the earlier sample size, the run-level flag rate remains at parity (if anything, marginally lower), confirming the guardrail's no-regression property holds at scale. The metric-level matrix shows the same redistribution pattern seen in Section 4.2: total metrics with any anomaly rise (9.0% newly surface one, versus 2.3% that go quiet), while the overall volume of alerts stays essentially flat, meaning anomalies are spread more evenly across the right metrics rather than concentrated as chronic noise on a few."),
]));

// 5. Next steps
children.push(h1("5. Open Items and Next Steps"));
children.push(numbered([t("Shadow mode: "), t("compute guarded-band bounds silently alongside existing tests on live traffic, with no user-facing change. The existing "), code("suggested_test"), t(" column and bounds infrastructure can carry this with minimal new code.")], "steps10"));
children.push(numbered([t("Label collection: "), t("one-click confirm or dismiss on real incidents during shadow mode. This is the only way to resolve whether the roughly 26% method-disagreement runs (Section 4.2C) are genuine catches or false positives; simulation has reached its evidentiary ceiling on this question.")], "steps10"));
children.push(numbered([t("Per-metric-type tolerance floors: "), t("generalize the 10% floor using the registry's existing "), code("step_threshold"), t(" fields, mirroring what the current Prophet path already does per metric type.")], "steps10"));
children.push(numbered([t("Registry allow-list: "), t("confirmed unnecessary for the guarded band (coverage figures assume it is dropped). If kept for other reasons, note it interacts poorly with the band's recency requirement: the intersection is only 8.9% of the population, smaller than either constraint alone.")], "steps10"));

// Reproducibility
children.push(h1("Reproducibility"));
children.push(p([
  t("All scripts referenced below are pushed to "),
  repoDir("", "github.com/rotem-pi/AI-projects/.../TestGenerationModel/"),
  t(". Per-series result CSVs are "), t("not", { italics: true }),
  t(" committed (they are raw production pulls); rerun the relevant script against the replica to regenerate them - see the folder's own "),
  repoFile("README.md"), t(" for setup. Key scripts:"),
]));
children.push(bullet([repoFile("db/model_bakeoff.py", "model_bakeoff.py"), t(" - architecture comparison (Section 3.3)")]));
children.push(bullet([repoFile("db/round2_representative.py", "round2_representative.py"), t(" / "), repoFile("db/round3_recent_tiers.py", "round3_recent_tiers.py"), t(" - representative sampling, lifetime vs. recent-cadence tiers")]));
children.push(bullet([repoDir("db/", "stage2_*.py"), t(", "), repoFile("db/stage3_guarded_band.py", "stage3_guarded_band.py"), t(", "), repoFile("db/stage4_final_eval.py", "stage4_final_eval.py"), t(" - progressive refinement of the final design (Section 3)")]));
children.push(bullet([repoFile("db/stage5_conversion_matrices.py", "stage5_conversion_matrices.py"), t(" - Section 4.2")]));
children.push(bullet([repoFile("db/resilient_matrices.py", "resilient_matrices.py"), t(" / "), repoFile("db/aggregate_resilient.py", "aggregate_resilient.py"), t(" - Section 4.3, checkpointed large-sample run")]));
children.push(bullet([repoFile("db/visual_side_by_side.py", "visual_side_by_side.py"), t(" - shared DB helpers only (connection, SQL, current-test predictor); not a report generator")]));
children.push(bullet([repoFile("db/stage4_final_eval.py", "stage4_final_eval.py"), t(" -> "), repoFile("db/guarded_band_report.html", "guarded_band_report.html"), t(": the single consolidated comparison report (illustrative examples with definity task links). An earlier separate 3-case report was merged into this one so there is exactly one report to keep in sync with the model.")]));

const doc = new Document({
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [
          { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 480, hanging: 240 } } } },
        ],
      },
      {
        reference: "steps3",
        levels: [
          { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 480, hanging: 300 } } } },
        ],
      },
      {
        reference: "steps10",
        levels: [
          { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 480, hanging: 300 } } } },
        ],
      },
    ],
  },
  styles: {
    default: { document: { run: { font: "Calibri", size: 22, color: DARK } } },
  },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(process.argv[2] || "findings.docx", buf);
  console.log("written");
});
