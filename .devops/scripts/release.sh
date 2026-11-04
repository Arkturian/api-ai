#!/usr/bin/env bash
# Release api-ai: dev -> main (fast-forward only), main triggers the
# GitHub Actions deploy on the self-hosted arkserver runner.
#
# Gates (#2642): clean tree, local dev == origin/dev, origin/main is an
# ancestor of dev (no merge, no force), full pytest suite green with
# pipefail. A push to main deploys ALL enabled hosts (arkturian, aiserver
# via "Deploy AI API"; arkserver via "Deploy to Arkserver"). To redeploy a
# single host from the already released main use: ./devops deploy <host>.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"

usage() {
  cat <<'USAGE'
Usage: release.sh [--dry-run]

Fast-forwards origin/main to dev after the gates pass. --dry-run runs all
gates and prints what would be pushed, without pushing.
USAGE
}

dry_run=false
for arg in "$@"; do
  case "$arg" in
    --dry-run) dry_run=true ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; usage; exit 1 ;;
  esac
done

cd "$REPO_ROOT"

[[ -z "$(git status --porcelain)" ]] || { echo "Error: working tree not clean." >&2; exit 1; }
[[ "$(git rev-parse --abbrev-ref HEAD)" == "dev" ]] || { echo "Error: release only from branch dev." >&2; exit 1; }

git fetch -q origin dev main
[[ "$(git rev-parse HEAD)" == "$(git rev-parse origin/dev)" ]] || {
  echo "Error: local dev ($(git rev-parse --short HEAD)) != origin/dev ($(git rev-parse --short origin/dev)). Push or pull first." >&2; exit 1; }
git merge-base --is-ancestor origin/main HEAD || {
  echo "Error: origin/main is not an ancestor of dev (main has commits dev lacks). Merge origin/main into dev first." >&2; exit 1; }

if [[ "$(git rev-parse origin/main)" == "$(git rev-parse HEAD)" ]]; then
  echo "Nothing to release: origin/main already at $(git rev-parse --short HEAD)."; exit 0
fi

printf '\n==> Compile check\n'
python3 -m compileall -q -q .

printf '\n==> Test suite\n'
log="$(mktemp)"
if ! python3 -m pytest -q -p no:cacheprovider tests 2>&1 | tee "$log" | tail -5; then
  echo "Error: test suite failed, release aborted (log: $log)." >&2; exit 1
fi
rm -f "$log"

printf '\n==> Commits to release\n'
git log --oneline origin/main..HEAD

if [[ "$dry_run" == true ]]; then
  echo "Dry run: would push $(git rev-parse --short HEAD) to origin/main."; exit 0
fi

git push origin HEAD:main
printf '\n✅ origin/main = %s. Deploy runs: gh run list -L 4\n' "$(git rev-parse --short HEAD)"
echo "Tenant instances follow the tag tenant-release (Cloud); move it only for tenant-relevant releases."
