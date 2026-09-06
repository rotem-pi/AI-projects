#!/usr/bin/env bash
# Build the Recommendation Agent image and push it to ECR (dev account).
#
#   ./deploy/build.sh              # stage context, build linux/amd64, push
#   ./deploy/build.sh --no-push    # local build only (tag recommendation-agent:local)
#
# The build context is staged under deploy/.context/ so the private
# definity-app checkout is copied from the harness's own pinned worktree
# (.worktrees/definity-app-auto-recs, synced by ./run.sh) instead of being
# cloned in CI — this is a laptop-driven deploy like tools/manual_tuning in
# definity-app. Needs: docker (buildx), an `aws sso login --profile dev-admin`
# session for the ECR push.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${AWS_PROFILE:=dev-admin}"
REGION=eu-north-1
ACCOUNT=412550564892
REPO=recommendation-agent
REGISTRY="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
PUSH=1; [[ "${1:-}" == "--no-push" ]] && PUSH=0

python3 bootstrap_worktree.py --no-sync >/dev/null
BACKEND=$(cat .worktrees/.backend_path)
BACKEND_SHA=$(git -C "$BACKEND" rev-parse --short HEAD)
HARNESS_SHA=$(git rev-parse --short HEAD)$(git diff --quiet || echo "-dirty")
TAG="h${HARNESS_SHA}-b${BACKEND_SHA}"

CTX=deploy/.context
rm -rf "$CTX"; mkdir -p "$CTX/backend" "$CTX/harness/deploy"
# Backend: what the harness imports at runtime (app/, pyproject, lock, VERSION,
# logging conf, alembic is not needed). Mirrors backend/.dockerignore's intent.
rsync -a --delete \
  --exclude '.venv' --exclude '__pycache__' --exclude 'tests' --exclude 'dev' \
  --exclude 'stress_test' --exclude 'scripts' --exclude '*.egg-info' --exclude '.ruff_cache' \
  --exclude 'mjml' --exclude 'dist' --exclude '*.log' \
  "$BACKEND/" "$CTX/backend/"
# Harness: code only — no data/, results, dumps, .env, worktrees, venvs.
rsync -a --delete \
  --exclude '.git' --exclude '.worktrees' --exclude 'data' --exclude '.env' --exclude 'venv' \
  --exclude '.venv' --exclude '__pycache__' --exclude '.ruff_cache' --exclude '*.ipynb' \
  --exclude 'deploy/.context' --exclude '.claude' --exclude 'agent/kb' \
  ./ "$CTX/harness/"
# agent/ symlinks are rewritten in the image; ship them as-is (rsync -a keeps them).
cp deploy/aws-config "$CTX/harness/deploy/aws-config"
cp deploy/Dockerfile "$CTX/Dockerfile"
printf 'harness=%s\nbackend=%s\ntag=%s\n' "$HARNESS_SHA" "$BACKEND_SHA" "$TAG" > "$CTX/harness/BUILD_INFO"

if (( PUSH )); then
  aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" --profile "$AWS_PROFILE" >/dev/null 2>&1 \
    || aws ecr create-repository --repository-name "$REPO" --region "$REGION" --profile "$AWS_PROFILE" \
         --image-scanning-configuration scanOnPush=true >/dev/null
  aws ecr get-login-password --region "$REGION" --profile "$AWS_PROFILE" \
    | docker login --username AWS --password-stdin "$REGISTRY"
  IMAGE="$REGISTRY/$REPO:$TAG"
  docker buildx build --platform linux/amd64 -t "$IMAGE" -t "$REGISTRY/$REPO:latest" --push "$CTX"
  echo "$IMAGE" > deploy/.last-image
  echo "pushed $IMAGE"
else
  docker buildx build --platform linux/amd64 -t "$REPO:local" --load "$CTX"
  echo "built $REPO:local"
fi
