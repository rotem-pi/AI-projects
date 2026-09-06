#!/usr/bin/env bash
# Deploy the Recommendation Agent to dev-eks.
#
#   ./deploy/deploy.sh             # build + push (deploy/build.sh), then roll out
#   ./deploy/deploy.sh --no-build  # roll out the image recorded in deploy/.last-image
#
# Same shape as definity-app's tools/manual_tuning/deploy/deploy.sh: applied from
# a laptop with the dev-admin SSO session (dev-eks maps the PowerUserAccess SSO
# role). No Secret is needed — the pod signs into AWS via the device-code flow
# from the site itself, and nothing else here holds credentials.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ctx="${KUBE_CONTEXT:-dev-eks}"
k=(kubectl --context "$ctx")
ns=recommendation-agent

[[ "${1:-}" == "--no-build" ]] || ./deploy/build.sh
IMAGE=$(cat deploy/.last-image)

"${k[@]}" apply -f deploy/k8s.yaml
"${k[@]}" -n "$ns" set image deployment/recommendation-agent app="$IMAGE"
"${k[@]}" -n "$ns" rollout status deployment/recommendation-agent --timeout=300s

echo "https://recommendation-agent.dev.definity.run (internal ALB -- needs VPN)"
echo "image: $IMAGE"
