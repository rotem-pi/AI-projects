# AutoRecommendation

Standalone test harness for definity-app's **auto-recommendations agent**. This harness is based
on the code in definity-app's `auto-recommendations-agent` branch
(`backend/app/brain/insights/agent`) — same nodes, prompts, knowledge base, tools and safety
rules — with Athena (or a REST-API dump) standing in for Postgres as the data source.

## Layout

```
main.py                # ① run the agent on Athena data (batch / single task)
run_from_dump.py       # ① run the agent on a REST-API dump (no Athena needed)
agent/                 # agent core — mostly symlinks into definity-app (see below)
evaluation/            # ② tests: consistency gate + notebook, PG-vs-Athena diff
tools/                 # ③ data acquisition: REST dump script + JSON->CSV helper
data/
  benchmark_tasks.json # pinned task set for the consistency gate
  dumps/dump_<id>/     # REST-API dumps (input to run_from_dump.py)
  results/             # saved agent runs (gitignored, local only)
```

## ① Running the agent

- **[main.py](main.py)** — CLI entry point against Athena.

  ```bash
  python main.py --sample 10        # 10 random tasks with real history
  python main.py --task-id 2481696  # one specific task
  python main.py --sample 10 --json
  ```

- **[run_from_dump.py](run_from_dump.py)** — run the same inference graph on a directory made
  by `tools/dump-rest-api.sh`. No Athena/AWS needed; known gaps (no historical runs → no trend
  analysis) are printed at startup.

  ```bash
  python run_from_dump.py data/dumps/dump_5453
  ```

Both save results to `data/results/` (override with `RESULTS_DIR` in `.env`): one JSON per task
with the input enrichment row, the full plan, `assembled_output`, and a per-node graph trace;
batch runs also get a `batch_summary.csv`.

## ② Evaluation

- **[evaluation/consistency_gate.py](evaluation/consistency_gate.py)** — CI-style pass/fail
  gate: runs the agent N cycles over the pinned benchmark
  ([data/benchmark_tasks.json](data/benchmark_tasks.json)) and fails (exit 1) if average
  config-key consistency drops below the threshold or too many runs degrade. Run it before
  shipping any prompt/KB change.

  ```bash
  python evaluation/consistency_gate.py                      # 5 cycles, 80% threshold
  python evaluation/consistency_gate.py --create-benchmark 5 # (re)pin the benchmark set
  ```

- **[evaluation/consistency_test.ipynb](evaluation/consistency_test.ipynb)** — interactive
  exploration of a gate run (per-task / per-key breakdowns). Reuses the gate's scoring; never
  re-implements the metric.

- **[evaluation/compare_sql_stream.py](evaluation/compare_sql_stream.py)** — diffs the
  *production* fetch_context → entry_gate → sql_stream nodes (run unmodified against Postgres
  via `PG_DATABASE_URL`) against the insights recorded in Athena, per insight type.

  ```bash
  python evaluation/compare_sql_stream.py --task-id 1473514
  ```

- **[evaluation/pg_nodes_runner.py](evaluation/pg_nodes_runner.py)** — subprocess helper for
  the above: executes the literal production nodes inside the definity-app backend venv and
  dumps their outputs as JSON. Not usually run by hand.

## ③ Data acquisition

- **[tools/dump-rest-api.sh](tools/dump-rest-api.sh)** — dump everything the definity REST API
  knows about one task into `data/dumps/dump_<task_id>/` (task detail, params, metrics, TFs,
  lineage, events, time-series, plus per-TF detail/events/physical-plan/lineage/stages, all as
  CSV). Needs `DEFINITY_API_TOKEN` in `.env`. Endpoints are retried 3× — the prod API 500s
  intermittently.

  ```bash
  ./tools/dump-rest-api.sh 5453
  python run_from_dump.py data/dumps/dump_5453
  ```

- **[tools/json2csv.py](tools/json2csv.py)** — stdin JSON → CSV converter used by the dump
  script.

## Important: symlinked agent code

The agent core under [agent/](agent/) (`constants.py`, `enums.py`, `models.py`, `prompts.py`,
`state.py`, `tools.py`, `knowledge_base.py`, `kb/`, and most of `agent/nodes/`) consists of
**symlinks into a local definity-app checkout** on the `auto-recommendations-agent` branch.
The definity-app code itself is intentionally **not** vendored into this repo, so those links
appear broken on GitHub. To run the harness you need definity-app checked out at
`~/GitCode/definity-app` on that branch (or re-point the symlinks).

Files that are real (local to this harness):

- [agent/inference_graph.py](agent/inference_graph.py) — the branch's graph with two data nodes
  substituted for Athena.
- [agent/nodes/fetch_context.py](agent/nodes/fetch_context.py),
  [agent/nodes/sql_stream.py](agent/nodes/sql_stream.py) — the substituted Athena-backed nodes
  (both support the in-memory overrides run_from_dump.py injects).
- [agent/cost_utils.py](agent/cost_utils.py) — tenant-pricing cost computation.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in ANTHROPIC_API_KEY, AWS profile, DEFINITY_API_TOKEN, etc.
```

No API keys or credentials are committed — see [.env.example](.env.example) for the required
variables.

## Results

Saved agent runs are kept in `data/results/` (gitignored, local only). Each batch directory
contains a `batch_summary.csv` (one row per task: utilization metrics, gate decision,
recommendation counts) plus one JSON per task with the full saved graph state (`input`, `plan`,
`run_metrics`, `cost_profile`, `assembled_output`, `trace`).
