# definiData

Ask the `dev_app_analytics` database questions in plain English and get a
clear answer back, with an optional view of the generated SQL. Styled to
match definity.ai (Ubuntu font, teal gradient), with a light/dark toggle.
Runs as a real macOS app (own icon in Applications/Launchpad) or in your
browser. Built on the same NL-to-SQL RAG pipeline as the `ask.py` CLI
prototype (`BedrockAthenaSQLExecutor` Lambda + Bedrock Nova Lite + Athena),
region `eu-north-1`, AWS account `412550564892`.

## Prerequisites

- Python 3.9+.
- ~500MB free disk space. Almost all of it is Streamlit's own dependencies
  (pandas, pyarrow, numpy) pulled in automatically - this app doesn't use a
  dataframe itself.
- A few minutes for first-time setup (a from-scratch, cache-free install
  measured at ~30s on a fast connection; expect longer on a slower one).
- An AWS SSO `dev-admin` profile (account `412550564892`, region `eu-north-1`)
  with permission to:
  - `lambda:InvokeFunction` on `arn:aws:lambda:eu-north-1:412550564892:function:BedrockAthenaSQLExecutor`
  - `bedrock:InvokeModel` on `arn:aws:bedrock:eu-north-1:412550564892:application-inference-profile/718rksnz6bq9`
    (and on the underlying `amazon.nova-lite-v1:0` foundation model)

  `install.sh` auto-configures this profile in `~/.aws/config` if it's
  missing. If you get `AccessDenied` errors, ask your AWS admin to grant the
  above on your role.
- Sign in with `aws sso login --profile dev-admin` before running definiData,
  and again whenever your session expires. `install.sh` offers to do this for
  you, and the app itself will prompt you to sign in (with a one-click
  button) if it detects you aren't.

## Setup - native macOS app (recommended)

```bash
git clone https://github.com/rotem-pi/AI-projects.git
cd AI-projects/tools/definidata
./install.sh
```

This creates a virtual environment, installs dependencies, and installs
**definiData.app** into `/Applications`. After that, open it from
Applications or Launchpad like any other app - no terminal, no re-running
scripts, no browser tab.

**Caveat:** the installed app points at the absolute path of wherever you
cloned the repo. If you move or delete that folder later, the app will stop
working until you `cd` back in and re-run `./install.sh`.

## Setup - browser tab (non-macOS, or if you'd rather not install an app)

```bash
git clone https://github.com/rotem-pi/AI-projects.git
cd AI-projects/tools/definidata
./init.sh
```

Same dependency setup, but launches in your browser each time (usually
http://localhost:8501) instead of installing an app icon.

## Usage

Type a question (e.g. "top 5 tenants by activity") and press Enter or click
**Ask**. Tick **Show SQL** to see the generated query and raw rows too.
Toggle **Dark mode** top-right for a dark theme.

If you're not signed in (or your session expired), the app tells you clearly
that's why nothing ran - not a bug in your question - and gives you a
**Sign in with aws sso login** button to fix it without leaving the app.

## Auto-update

definiData checks `origin/main` periodically (and on launch). If a newer
version is available, a banner appears with an **Update & Restart** button
(or **Update now** in browser mode) that pulls the latest code, reinstalls,
and restarts the app for you.

## Notes

- Each user queries with their own AWS credentials - there's no shared
  backend server or hosted deployment.
- The schema documentation used to generate accurate SQL joins lives inside
  the `BedrockAthenaSQLExecutor` Lambda itself, not in this repo - reach out
  if you need it updated for new tables/columns.
