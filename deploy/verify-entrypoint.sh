#!/usr/bin/env bash
# Runs inside the one-shot `verify` compose service. Every stage must pass;
# the script exits non-zero (which `docker compose run` reports) on failure.
set -u

cd /verify/backend
echo "=== [1/3] backend test suite (pytest) ==="
/opt/venv/bin/python -m pytest -v || exit 1

echo
echo "=== [2/3] web production build (tsc + vite) ==="
cd /verify/web
npm run build || exit 1

echo
echo "=== [3/3] live API smoke test against ${API_BASE_URL} ==="
/opt/venv/bin/python /verify/smoke.py "${API_BASE_URL}" || exit 1

echo
echo "VERIFY OK: tests, build and API smoke checks all passed."
exit 0
