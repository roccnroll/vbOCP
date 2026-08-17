#!/usr/bin/env bash
# Test di integrazione: risolve il FOM per Test_1 con i parametri di default
# (mu1=12, mu2=2.5, mu_u=0.99 - gli stessi del notebook originale, per confronto diretto).
#
# Uso: ./scripts/run_fom_test.sh
# Va lanciato dalla root della repo (vbOCP/), con l'ambiente pypolydim attivo.

set -euo pipefail

cd "$(dirname "$0")/.."

python -m src.full_order.run_single_solve \
    --config configs/test1.yaml \
    --mu1 12 \
    --mu2 2.5 \
    --mu_u 0.99
