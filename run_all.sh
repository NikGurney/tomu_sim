#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
echo "== residue self-check ==" ; "$PY" -m checks.verify_residue
echo "== structure self-check ==" ; "$PY" -m checks.verify_structure
echo "== run 1 =="             ; "$PY" -m runs.run1_ablation
echo "== run 2 =="             ; "$PY" -m runs.run2_reconciliation
echo "== run 3 =="             ; "$PY" -m runs.run3_faithfulness
echo "== tests =="             ; "$PY" -m checks.run_tests
echo "Done. See ./outputs/"
