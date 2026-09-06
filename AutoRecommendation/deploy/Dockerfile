# Recommendation Agent — the self-serve site over the AutoRecommendation harness.
#
# Built by deploy/build.sh, which stages the build context as:
#   backend/   definity-app's backend at the auto-recommendations-agent commit
#              the harness is pinned to (.worktrees/definity-app-auto-recs/backend,
#              minus tests/dev/scripts/.venv) — the private repo never has to be
#              cloned inside the build
#   harness/   this repo minus data/, .worktrees/, .env and other local state
#
# Runtime layout mirrors a developer checkout so nothing in the harness needs a
# code path for "in a container": /app/backend is what .worktrees/.backend_path
# points at, and agent/'s symlinks are re-pointed there at build time with the
# same bootstrap_worktree.relink_agent_symlinks() ./run.sh uses.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1 \
    VIRTUAL_ENV=/app/backend/.venv \
    PATH="/app/backend/.venv/bin:$PATH" \
    HOME=/home/app

# AWS CLI v2 is what performs `aws sso login --use-device-code` for the
# shared dev-admin session (service/app.py); boto3 then reads its token cache.
RUN set -ex \
    && apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends curl unzip ca-certificates procps \
    && curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-$(uname -m).zip" -o /tmp/awscli.zip \
    && unzip -q /tmp/awscli.zip -d /tmp && /tmp/aws/install && rm -rf /tmp/aws /tmp/awscli.zip \
    && apt-get purge -y unzip && apt-get autoremove -y && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend
# Dependencies first so a harness-only change reuses this (large) layer.
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev
COPY backend/ ./

WORKDIR /app/harness
COPY harness/ ./
# Point agent/'s symlinks at /app/backend and record the resolved backend path,
# exactly what ./run.sh's bootstrap does on a laptop (without git).
RUN python -c "import bootstrap_worktree as b; from pathlib import Path; \
    b.relink_agent_symlinks(Path('/app/backend')); \
    b.BACKEND_PATH_FILE.parent.mkdir(exist_ok=True); \
    b.BACKEND_PATH_FILE.write_text('/app/backend\n')"

# Non-secret AWS profile definition; the SSO token cache lands in
# $HOME/.aws/sso/cache at runtime (an emptyDir in k8s).
RUN groupadd -r app && useradd -r -g app -d /home/app -m app \
    && mkdir -p /home/app/.aws /data/jobs \
    && chown -R app:app /app /home/app /data \
    # World-readable home: the profile below holds no secrets, and this lets
    # the image also run under an arbitrary uid (e.g. --user for local tests).
    && chmod 755 /home/app /home/app/.aws && chmod -R a+rX /app
COPY --chown=app:app harness/deploy/aws-config /home/app/.aws/config

USER app
ENV AWS_PROFILE=dev-admin \
    AWS_REGION=eu-north-1 \
    AWS_SSO_AUTO_LOGIN=0 \
    REC_AGENT_JOBS_DIR=/data/jobs \
    # definity-app's Settings requires a database_url even though nothing here
    # opens Postgres (the harness reads Athena); is_saas/is_dev_mode mirror the
    # developer .env the harness normally points ENV_FILE_NAME at.
    DATABASE_URL=postgresql://unused:unused@localhost:5432/unused \
    DATABASE_POOL_MIN=0 \
    IS_SAAS=true \
    IS_DEV_MODE=true

# Build-time smoke test: the harness must import against /app/backend with the
# runtime env above (definity-app's Settings validates at import).
RUN python -c "import sys; sys.path.insert(0, '/app/backend'); import agent.local_sandbox, service.app" \
    && echo "harness imports OK"

EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD curl -fsS http://127.0.0.1:8765/health || exit 1
CMD ["python", "-m", "uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8765"]
