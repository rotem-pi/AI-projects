# AutoRecommendation

Standalone test harness for definity-app's **auto-recommendations agent**. This test is based on
the code in definity-app's `auto-recommendations-agent` branch
(`backend/app/brain/insights/agent`) — same nodes, prompts, knowledge base, tools and safety
rules — with Athena standing in for Postgres as the data source.

## Important: symlinked agent code

The agent core under [agent/](agent/) (`constants.py`, `enums.py`, `models.py`, `prompts.py`,
`state.py`, `tools.py`, `knowledge_base.py`, `kb/`, and most of `agent/nodes/`) consists of
**symlinks into a local definity-app checkout** on the `auto-recommendations-agent` branch.
The definity-app code itself is intentionally **not** vendored into this repo, so those links
appear broken on GitHub. To run the harness you need definity-app checked out at
`~/GitCode/definity-app` on that branch (or re-point the symlinks).

Files that are real (local to this harness):

- [main.py](main.py) — CLI entry point (`--sample N`, `--task-id`, `--athena` batch mode).
- [agent/inference_graph.py](agent/inference_graph.py) — the branch's graph with two data nodes
  substituted for Athena.
- [agent/nodes/fetch_context.py](agent/nodes/fetch_context.py),
  [agent/nodes/sql_stream.py](agent/nodes/sql_stream.py) — the substituted Athena-backed nodes.
- [agent/cost_utils.py](agent/cost_utils.py) — tenant-pricing cost computation.
- [pg_nodes_runner.py](pg_nodes_runner.py), [compare_sql_stream.py](compare_sql_stream.py) —
  run/compare the original Postgres-backed nodes against a read replica.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in ANTHROPIC_API_KEY, AWS profile, etc.
```

No API keys or credentials are committed — see [.env.example](.env.example) for the required
variables.

## Results

Saved agent runs are kept in [data/results/](data/results/). Each batch directory contains a
`batch_summary.csv` (one row per task: utilization metrics, gate decision, recommendation
counts) plus one JSON per task with the full saved graph state (`input`, `plan`, `run_metrics`,
`cost_profile`, `assembled_output`, `trace`).
