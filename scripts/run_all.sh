#!/usr/bin/env bash
# Train the oracle, both tokenizers and all nine models, then evaluate and build the report.
# Two single-threaded lanes run in parallel (each lane trains its own tokenizer first).
#   RUNS=runs STEPS=2000 bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
RUNS=${RUNS:-runs}
PY=${PY:-.venv/bin/python}
EXTRA=${STEPS:+--steps $STEPS}
export OMP_NUM_THREADS=1
mkdir -p "$RUNS/logs"

train() {  # name [extra args...]
  local name=$1; shift
  if [[ -f "$RUNS/$name/model.pt" ]]; then echo "skip $name (exists)"; return; fi
  $PY -u -m tinyprism.train "$name" --out "$RUNS/$name" --threads 1 "$@" > "$RUNS/logs/$name.log" 2>&1
  tail -n 1 "$RUNS/logs/$name.log"
}

lane_a() {
  train oracle
  train vq
  for m in dit mmdit unet; do train "$m" $EXTRA; done
  for m in ar maskgit; do train "$m" --tokenizer "$RUNS/vq/model.pt" $EXTRA; done
}
lane_b() {
  train msvq
  for m in jit single-stream mar; do train "$m" $EXTRA; done
  train var --tokenizer "$RUNS/msvq/model.pt" $EXTRA
}

lane_a & A=$!
lane_b & B=$!
wait $A; wait $B

for m in unet dit jit mmdit single-stream ar maskgit var mar; do
  [[ -f "$RUNS/$m/metrics.json" ]] || $PY -u -m tinyprism.evaluate --ckpt "$RUNS/$m/model.pt" --oracle "$RUNS/oracle/model.pt" --n 128 --threads 1 > "$RUNS/logs/eval-$m.log" 2>&1
  echo "evaluated $m"
done
$PY -u scripts/make_report.py --runs "$RUNS" --out results
