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
tools/                 # ③ data acquisition: REST/Postgres/Athena dump scripts + JSON->CSV helper
service/               # ④ self-serve service pieces: run_job (name -> result) + deterministic TL;DR
data/
  benchmark_tasks.json # pinned task set for the consistency gate
  dumps/dump_<id>/     # REST-API dumps (input to run_from_dump.py)
  results/             # saved agent runs (gitignored, local only)
```

## ① Running the agent

- **[main.py](main.py)** — CLI entry point against Athena.

  ```bash
  ./run.sh main.py --sample 10        # 10 random tasks with real history
  ./run.sh main.py --task-id 2481696  # one specific task
  ./run.sh main.py --sample 10 --json
  ```

- **[run_from_dump.py](run_from_dump.py)** — run the same inference graph on a directory made
  by `tools/dump-rest-api.sh`. No Athena/AWS needed; known gaps (no historical runs → no trend
  analysis) are printed at startup.

  ```bash
  ./run.sh run_from_dump.py data/dumps/dump_5453
  ```

`./run.sh <script.py> [args...]` is the entry point for anything under `agent/` — it syncs the
dedicated worktree (see "Important: symlinked agent code" below) and runs `uv run --project
<worktree>/backend python <script.py> [args...]`. Prefix with `--no-sync` to skip the fetch when
offline (`./run.sh --no-sync main.py --sample 10`).

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
  ./run.sh evaluation/compare_sql_stream.py --task-id 1473514
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

- **[run_tuning_preview.py](run_tuning_preview.py)** — build the *production* tuning-preview
  plan (the exact `GET /api/envs/{env}/tuning/preview` payload) for one `(app_id, task_name)`
  against `PG_DATABASE_URL` and save it as `~/Downloads/<app_name>.json` (the app's display
name, e.g. "Job Cluster - \<job\> - \<cluster\>"), with `app_id` / `app_name` / `task_name` /
`tenant_id` embedded at the top of the JSON. Read-only. This is
  the downstream consumer of agent insights pushed into the `insights` table: run it after
  definity-app's `add-agent-insights` skill to see how the agent's recommendations land in the
  actual plan (knobs / blocked_knobs / staircases).

  ```bash
  ./run.sh run_tuning_preview.py --app-id 21483 --task-name compute
  ```

## ④ Self-serve service pieces (`service/`)

Building blocks for a task-name-in, recommendations-out web service (one subprocess per job —
the harness injects data through module globals, so jobs must never share a process):

- **[service/run_job.py](service/run_job.py)** — task name → Athena dump → agent run →
  `result.json` + `tldr.md`, all under one `--workdir` (delete the directory to remove the dump,
  the local run-sandbox folders and the outputs). Task names are not unique (`compute` runs under
  hundreds of apps): without `--app-id` an ambiguous name exits 3 and prints the candidates
  (app/env names included) as JSON for a UI to offer as choices; an unknown name exits 2. The
  analyzed run is the latest **COMPLETED** run of that (task_name, app_id) in the Athena export.

  ```bash
  ./run.sh service/run_job.py --task-name compute --app-id 21447 --workdir /tmp/job1 --print-tldr
  ./run.sh service/run_job.py --task-id 3136527 --workdir /tmp/job2
  ```

- **[service/tldr.py](service/tldr.py)** — deterministic Markdown TL;DR of a saved result JSON
  (no LLM call): *What to change* (change / why / how / risk / $), *Blocked* (intended change /
  why blocked), *Not actioned* (insight / why), *Data & provenance* (analyzed run, cost per
  run, **Athena export lag note**, model). Gate-blocked runs render a single *Not evaluated*
  section with the gate reasons.

  ```bash
  python service/tldr.py data/results/task_18609_20260906T081331Z.json
  ```

- **[service/app.py](service/app.py)** — **Recommendation Agent**, the web service itself (FastAPI, one
  process): serves [service/static/index.html](service/static/index.html), answers the Athena
  lookups (task-name type-ahead, per-app candidates, export freshness), runs each analysis as a
  `run_job.py` subprocess (`REC_AGENT_MAX_CONCURRENT`, default 2), streams progress from its
  stderr, serves `result.json` / `tldr.md`, and sweeps job directories `REC_AGENT_TTL_HOURS`
  (default 4) after they finish. AWS sign-in is the SSO device-code flow: the page shows the URL +
  code, the CLI caches the token, and the session is shared by every user of the site (intended
  for deployment behind the team's own SSO). Jobs live under `data/jobs/<id>/` (gitignored).

  ```bash
  ./run.sh --no-sync -m uvicorn service.app:app --host 127.0.0.1 --port 8765
  # then open http://localhost:8765  (API docs at /api/docs)
  ```

- **[deploy/](deploy/)** — running the service on **dev-eks** (`recommendation-agent` namespace,
  https://recommendation-agent.dev.definity.run, internal ALB → VPN only, same conventions as
  definity-app's `tools/manual_tuning`). [deploy/build.sh](deploy/build.sh) stages the pinned
  definity-app backend from the harness's own worktree plus this repo into a build context (no
  private-repo clone inside the build), builds `linux/amd64` and pushes to ECR
  `412550564892.dkr.ecr.eu-north-1.amazonaws.com/recommendation-agent:h<harness>-b<backend>`.
  [deploy/deploy.sh](deploy/deploy.sh) applies [deploy/k8s.yaml](deploy/k8s.yaml) (Namespace,
  ConfigMap with the non-secret `.env` values, Deployment with emptyDirs for jobs and the SSO
  token cache, NodePort Service, ALB Ingress) and rolls the new image out. No Secret exists: the
  pod is signed into AWS from the site via the SSO device-code flow. An IRSA role would remove
  that step — creating one needs IAM rights PowerUserAccess lacks; see the note in `deploy/k8s.yaml`.

  ```bash
  ./deploy/deploy.sh            # build + push + roll out (needs docker, dev-admin SSO, VPN for the URL)
  ./deploy/build.sh --no-push   # local image only: recommendation-agent:local
  ```

Supporting Athena helpers in [main.py](main.py): `_fetch_athena_task_candidates(task_name)`
(per-app latest COMPLETED run, joined to the `apps`/`envs` mirrors for names),
`_search_athena_task_names(fragment)` (type-ahead), `_fetch_athena_export_freshness()` (newest
exported run / insights snapshot). All enrichment queries now LEFT JOIN the `tasks` mirror for
`status` (the export's `task_enrichments` has none, and the entry gate requires COMPLETED), and
the sandbox dump includes the run's `tasks` row (`kb/analysis/store.py` reads it unconditionally).

Service-mode env flags (see [.env.example](.env.example)): `AWS_SSO_AUTO_LOGIN=0` (raise instead
of shelling out to `aws sso login`), `AWS_SSO_USE_DEVICE_CODE=1` (headless login prints a URL +
code), `LOCAL_SANDBOX_DIR` (parent for run-sandbox temp folders).

## Important: symlinked agent code

The agent core under [agent/](agent/) (`constants.py`, `enums.py`, `models.py`, `prompts.py`,
`state.py`, `tools.py`, `knowledge_base.py`, `kb/`, and most of `agent/nodes/`) consists of
**symlinks into definity-app's `auto-recommendations-agent` branch**. The definity-app code
itself is intentionally **not** vendored into this repo, so those links appear broken on GitHub.

Rather than pointing at your own definity-app checkout directly — which lives at whatever path
and branch you happen to have it on for other work, and would be unsafe for this harness to
check out or reset — [bootstrap_worktree.py](bootstrap_worktree.py) maintains its **own**
disposable `git worktree` of that branch (at `.worktrees/definity-app-auto-recs/`, gitignored),
fetched from `DEFINITY_APP_REPO` in `.env` (defaults to `~/GitCode/definity-app`) but never
modifying it. [run.sh](run.sh) runs this sync automatically before every command, so:

- the agent code you run is always `auto-recommendations-agent` at its latest fetched commit,
  regardless of what branch is checked out in `DEFINITY_APP_REPO` or your current directory;
- your own `DEFINITY_APP_REPO` clone is only ever read from (`git fetch`), never checked out,
  reset, or otherwise mutated — safe to run even while you have uncommitted work there.

Run `python bootstrap_worktree.py` by hand any time (e.g. to re-sync without also running a
script), or add `--no-sync` after `./run.sh` to skip the fetch when offline. First run creates
the worktree (one `git fetch` + `git worktree add --detach`); later runs just fetch + checkout
the latest commit into it.

Files that are real (local to this harness):

- [agent/inference_graph.py](agent/inference_graph.py) — the branch's graph with two data nodes
  substituted for Athena.
- [agent/nodes/fetch_context.py](agent/nodes/fetch_context.py),
  [agent/nodes/sql_stream.py](agent/nodes/sql_stream.py) — the substituted Athena-backed nodes
  (both support the in-memory overrides run_from_dump.py injects).
- [agent/local_sandbox.py](agent/local_sandbox.py) — local-filesystem stand-in for the branch's
  S3-backed run sandbox (`app.dal.sandbox_dal`), so `run_deterministic_review` /
  `estimate_change_saving` / `get_advanced_config_catalog` read real per-run data from a local
  temp folder instead of hitting S3. Single-PIT only (this run's rows, no multi-run history).
- [agent/cost_utils.py](agent/cost_utils.py) — tenant-pricing cost computation.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in AWS profile (Claude runs via Bedrock), DEFINITY_API_TOKEN, etc.
```

Also set `DEFINITY_APP_REPO` in `.env` to your local definity-app clone (any branch — see
"Important: symlinked agent code" above; defaults to `~/GitCode/definity-app` if unset).

No API keys or credentials are committed — see [.env.example](.env.example) for the required
variables.

Anything that imports `agent/` (`main.py`, `run_from_dump.py`, `run_from_pg_dump.py`,
`evaluation/compare_sql_stream.py`) should be run via `./run.sh <script.py> [args...]`, not
called with a bare `python`, so the worktree sync happens first — see "Important: symlinked
agent code" above.

## Results

Saved agent runs are kept in `data/results/` (gitignored, local only). Each batch directory
contains a `batch_summary.csv` (one row per task: utilization metrics, gate decision,
recommendation counts) plus one JSON per task with the full saved graph state (`input`, `plan`,
`run_metrics`, `cost_profile`, `assembled_output`, `trace`).
