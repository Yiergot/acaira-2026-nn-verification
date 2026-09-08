#!/usr/bin/env bash
# End-to-end check of the hands-on artefacts. Needs `vehicle` and `Marabou` on PATH
# (after `bash environment/colab_bootstrap.sh` + PATH export, or inside the Docker image).
#
#   bash src/smoke_test.sh                      # run everything, print a table, exit 1 on unexpected results
#   MAX_QUERY_SECONDS=30 bash src/smoke_test.sh
set -uo pipefail
cd "$(dirname "$0")/.."
MAX_QUERY_SECONDS="${MAX_QUERY_SECONDS:-30}"
LOGDIR="${LOGDIR:-/tmp/nnv-smoke}"; mkdir -p "$LOGDIR"
SPEC=specs/triage.vcl
DS=(-d patients:data/triage-eval-patients.idx -d labels:data/triage-eval-labels.idx)
EPS1=(-p epsRR:1 -p epsSpo2:1 -p epsSbp:3 -p epsPulse:2 -p epsTemp:0.1)   # measurement-level noise
EPS4=(-p epsRR:4 -p epsSpo2:4 -p epsSbp:12 -p epsPulse:8 -p epsTemp:0.4)  # 4x that
fail=0
printf '%-6s %-26s %-40s %-8s %s\n' MODEL PROPERTY RESULT SECONDS NOTE
run() {  # run MODEL PROPERTY EXPECTED [extra args...]
  local model=$1 prop=$2 expect=$3; shift 3
  local t0 t1 out res secs note=""
  t0=$(date +%s.%N)
  out=$(vehicle verify -s "$SPEC" -n triage:models/triage-$model.onnx -y "$prop" --solver Marabou -a "--timeout=$MAX_QUERY_SECONDS" "$@" 2>&1 | tr '\r' '\n')
  t1=$(date +%s.%N)
  if echo "$out" | grep -q "verified:"; then
    res=$(echo "$out" | grep -E "verified:|falsified:|timed-out:|errored:" | awk '{printf "%s%s ", $1, $2}')
  elif echo "$out" | grep -q "proved no counterexample"; then res="verified"
  elif echo "$out" | grep -q "found a counterexample"; then res="falsified"
  elif echo "$out" | grep -qi "timed out"; then res="timeout"
  else res="ERROR"; fi
  secs=$(python3 -c "print(f'{$t1-$t0:.1f}')")
  case "$expect" in
    verified|falsified) [ "$res" = "$expect" ] || { note="UNEXPECTED (wanted $expect)"; fail=1; } ;;
    all-verified) echo "$res" | grep -q "falsified:0/" || { note="UNEXPECTED (wanted none falsified)"; fail=1; } ;;
    mixed) { echo "$res" | grep -q "falsified:0/" || echo "$res" | grep -q "verified:0/"; } && { note="UNEXPECTED (wanted a mix)"; fail=1; } ;;
    any) ;;
  esac
  [ "$res" = "ERROR" ] && { note="ERROR: $(echo "$out" | grep -v '^\s*$' | tail -1)"; fail=1; }
  printf '%-6s %-26s %-40s %-8s %s\n' "$model" "$prop" "$res" "$secs" "$note"
  echo "$out" > "$LOGDIR/$model-$prop-$#.log"
}
for m in v1 v2 v3; do
  case $m in v1) E=falsified;; v2) E=any;; v3) E=verified;; esac
  run $m hypoxiaNeverLow $E
  run $m hypoxiaOnOxygenNeverLow any
  run $m shockNeverLow $E
  run $m notAlertNeverLow $E
  run $m normalVitalsAlwaysLow any
  run $m noiseRobust any "${DS[@]}" "${EPS1[@]}"
  run $m noiseRobust any "${DS[@]}" "${EPS4[@]}"
done
echo; echo "counterexample for v1 / hypoxiaNeverLow:"; grep -A1 "counterexample" "$LOGDIR/v1-hypoxiaNeverLow-0.log" | tail -2
echo
if [ $fail -ne 0 ]; then echo "SMOKE TEST: unexpected results above"; exit 1; else echo "SMOKE TEST: all expectations met"; fi
