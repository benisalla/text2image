#!/usr/bin/env bash
# Train the oracle, both tokenizers and all nine models, then evaluate and build the report.
#   JOBS=4 STEPS=2000 bash scripts/run_all.sh
# Phase 1 runs everything without dependencies, JOBS at a time (one thread each);
# phase 2 trains the token models on top of the finished tokenizers.
# Finished runs (runs/<name>/model.pt) are skipped, so the script can be resumed.
set -euo pipefail
cd "$(dirname "$0")/.."
export RUNS=${RUNS:-runs} PY=${PY:-.venv/bin/python} STEPS=${STEPS:-} OMP_NUM_THREADS=1
JOBS=${JOBS:-4}
mkdir -p "$RUNS/logs"

train() {  # name [--tokenizer path]
  local name=$1; shift
  # If the same run is already in progress (e.g. an earlier invocation), wait for it.
  while pgrep -f "text2image.train $name --out $RUNS/$name " > /dev/null; do sleep 30; done
  if [[ -f "$RUNS/$name/model.pt" ]]; then echo "skip $name (exists)"; return; fi
  local extra=()
  if [[ -n "$STEPS" && ! "$name" =~ ^(oracle|vq|msvq)$ ]]; then extra=(--steps "$STEPS"); fi
  if $PY -u -m text2image.train "$name" --out "$RUNS/$name" --threads 1 "$@" "${extra[@]}" > "$RUNS/logs/$name.log" 2>&1; then
    tail -n 1 "$RUNS/logs/$name.log"
  else
    echo "FAILED $name (see $RUNS/logs/$name.log)"
  fi
}
export -f train

printf '%s\n' oracle vq msvq unet unet-eps dit mmdit mar meanflow sana transfusion jit single-stream | xargs -P "$JOBS" -I{} bash -c 'train {}'
printf '%s\n' "ar vq" "maskgit vq" "var msvq" | xargs -P "$JOBS" -I{} bash -c 'set -- {}; train $1 --tokenizer $RUNS/$2/model.pt'

for m in unet unet-eps dit jit mmdit single-stream sana meanflow transfusion ar maskgit var mar; do
  [[ -f "$RUNS/$m/model.pt" ]] || continue
  [[ -f "$RUNS/$m/metrics.json" ]] || $PY -u -m text2image.evaluate --ckpt "$RUNS/$m/model.pt" \
    --oracle "$RUNS/oracle/model.pt" --n 128 --threads 1 > "$RUNS/logs/eval-$m.log" 2>&1
  echo "evaluated $m: $(grep '"all"' "$RUNS/$m/metrics.json")"
done
$PY -u scripts/make_report.py --runs "$RUNS" --out results
