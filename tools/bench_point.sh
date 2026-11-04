#!/bin/bash
# One benchmark point: start an SGLang server with the production launch script, run the
# InferenceX/aiperf workloads against it, stop the server.
#
# Usage: bench_point.sh <tag> <workloads> <kv_dtype> <attn_backend> [extra launch args...]
#   workloads: comma list of livekit | syn:<isl>:<osl>:<conc>   (e.g. livekit,syn:8192:256:32)
#   env passthrough: anything exported (SGLANG_PLUGINS, GREEN_ATTN_DISABLE, ...) reaches the server.
#   LAUNCH (default runs/fa4fp8/launch.sh), PORT (30000), SYN_DURATION (300), OUT_ROOT.
# Output: $OUT_ROOT/<tag>/{server.log, <workload>/aiperf_artifacts/..., server_info.json, DONE}
set -u
TAG=$1; WLS=$2; shift 2
LAUNCH=${LAUNCH:-/network-volume/rnd-sglang/runs/fa4fp8/launch.sh}
PORT=${PORT:-30000}
OUT_ROOT=${OUT_ROOT:-/network-volume/rnd-sglang/runs/green/bench}
SYN_DURATION=${SYN_DURATION:-300}
INFX=/network-volume/rnd-sglang/InferenceX-wt-livekit/inferencex-e2e
OUT=$OUT_ROOT/$TAG
mkdir -p "$OUT"
echo "$*" > "$OUT/launch_args.txt"
env | grep -E '^(SGLANG_|GREEN_)' > "$OUT/env.txt"
URL=http://127.0.0.1:$PORT

stop_server() {
  [ -n "${SPID:-}" ] && kill -TERM -- -"$SPID" 2>/dev/null
  for _ in $(seq 60); do kill -0 "$SPID" 2>/dev/null || break; sleep 2; done
  kill -KILL -- -"$SPID" 2>/dev/null
  sleep 5
}
trap stop_server EXIT

PORT=$PORT setsid bash "$LAUNCH" "$@" > "$OUT/server.log" 2>&1 &
SPID=$!
for i in $(seq 360); do
  curl -sf "$URL/health" > /dev/null && break
  kill -0 "$SPID" 2>/dev/null || { echo "server died" > "$OUT/FAILED"; exit 1; }
  sleep 5
done
curl -sf "$URL/health" > /dev/null || { echo "server not ready" > "$OUT/FAILED"; exit 1; }
curl -sf "$URL/get_server_info" > "$OUT/server_info.json"

COMMON="PATH=/usr/local/bin:$PATH PYTHONPATH=$INFX AIPERF_RUNTIME_DIR=/network-volume/rnd-sglang/.aiperf-runtime-livekit KV_OFFLOADING=none MODEL=Qwen/Qwen3.8-27B-FP8 PORT=$PORT SERVING_REMOTE_URL=$URL SERVING_TOKENIZER=Qwen/Qwen3.8-27B-FP8 SERVING_SEARCH=0 AIPERF_SERVER_METRICS_URLS=$URL/metrics"
for wl in ${WLS//,/ }; do
  case $wl in
    parity)
      /usr/bin/python "$(dirname "$0")/parity.py" run "$URL" "$OUT/parity.json" > "$OUT/parity.log" 2>&1
      echo "parity EXIT=$?" >> "$OUT/status.txt"; continue;;
    gsm8k)
      ( cd / && PYTHONPATH=/network-volume/rnd-sglang/sglang/python /usr/bin/python -m sglang.test.few_shot_gsm8k \
          --num-questions 1319 --port "$PORT" --parallel 64 ) > "$OUT/gsm8k.log" 2>&1
      echo "gsm8k EXIT=$?" >> "$OUT/status.txt"; continue;;
    livekit)
      D=$OUT/livekit
      SC="$(cat $INFX/benchmarks/single_node/serving/scenarios/livekit.yaml)"
      KIND=file; DS=livekit-111868-v1.1-warm; LABEL=custom-agentic-livekit111868-v1.1-warm; CONC=32;;
    syn:*)
      IFS=: read -r _ ISL OSL CONC <<< "$wl"
      D=$OUT/syn-$ISL-$OSL-c$CONC
      SC="$(/network-volume/rnd-sglang/.aiperf-runtime-livekit/venv/bin/python - "$INFX/benchmarks/single_node/serving/scenarios/synthetic.yaml" "$ISL" "$OSL" "$SYN_DURATION" <<'EOF'
import sys, yaml
doc = yaml.safe_load(open(sys.argv[1]))
ds = doc["benchmark"]["datasets"][0]
ds["isl"], ds["osl"] = int(sys.argv[2]), int(sys.argv[3])
doc["benchmark"]["phases"][0]["duration"] = float(sys.argv[4])
print(yaml.safe_dump(doc, sort_keys=False))
EOF
)"
      KIND=synthetic; DS=synthetic; LABEL=synthetic-isl$ISL-osl$OSL; ;;
    *) echo "unknown workload $wl"; continue;;
  esac
  mkdir -p "$D"
  curl -sf -X POST "$URL/flush_cache" > /dev/null
  ( cd $INFX && env $COMMON RESULT_DIR=$D AGENTIC_OUTPUT_DIR=$D RESULT_FILENAME=result CONC=$CONC \
      SERVING_DATASET_KIND=$KIND SERVING_DATASET=$DS SERVING_SCENARIO_LABEL=$LABEL \
      SERVING_SCENARIO_YAML="$SC" bash -c 'python3 -m infx.serving.client prepare && python3 -m infx.serving.client run' ) \
      > "$D/client.log" 2>&1
  echo "$wl EXIT=$?" >> "$OUT/status.txt"
  curl -sf "$URL/metrics" > "$D/metrics_end.txt"
done
date > "$OUT/DONE"
