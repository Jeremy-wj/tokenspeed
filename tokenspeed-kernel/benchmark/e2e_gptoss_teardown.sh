#!/usr/bin/env bash
# Tear down TokenSpeed serve processes in this project's container only.
# GRACEFUL FIRST: the container's PID 1 is `sleep infinity`, which does NOT reap
# orphans, so SIGKILL'ing the orchestrator leaks the smg router's LISTEN socket
# (port stuck as an owner-less socket held by an unreaped zombie). Sending
# SIGINT/SIGTERM to the orchestrator (renamed "ts-serve") lets it cleanly stop
# the gateway + engine and release ports. Only then SIGKILL stragglers.
set -uo pipefail
CONTAINER="${CONTAINER:-jeremwan-tokenspeed-profiler}"
docker exec "$CONTAINER" bash -lc '
  # 0. Stop benchmark clients targeting this project server. A failed scheduler
  # can leave clients blocked indefinitely even after the server exits.
  bench_pids=$(ps -eo pid=,args= | awk \
    "\$2 ~ /python/ && \$3 == \"-m\" && \$4 == \"tokenspeed.cli\" && \
     \$5 == \"bench\" && \$6 == \"serve\" {print \$1}")
  [ -z "$bench_pids" ] || kill -TERM $bench_pids 2>/dev/null || true
  # 1. Graceful: signal the orchestrator + control server.
  pkill -INT  -x ts-serve   2>/dev/null || true
  pkill -TERM -x ts-serve   2>/dev/null || true
  pkill -TERM -x ts-control 2>/dev/null || true
  for _ in $(seq 1 12); do
    [ -z "$(pgrep -f ts-serve)" ] && break
    sleep 2
  done
  # 2. Force-kill any stragglers (engine, schedulers, router) by pattern + PID loop.
  for name in "ts-serve" "ts-control" "tokenspeed::sch" "smg_grpc_serv"; do
    pkill -9 -x "$name" 2>/dev/null || true
  done
  # A GPU abort can orphan the Python gRPC parent under container PID 1.
  orphan_pids=$(ps -eo pid=,args= | awk \
    "\$2 ~ /python$/ && \$3 == \"-m\" && \
     \$4 == \"smg_grpc_servicer.tokenspeed\" {print \$1}")
  [ -z "$orphan_pids" ] || kill -9 $orphan_pids 2>/dev/null || true
  for _ in $(seq 1 15); do
    pids=$(ps -eo pid=,stat=,comm= | awk -v self=$$ -v parent=$PPID \
      "\$1 != self && \$1 != parent && \$2 !~ /Z/ && \
       (\$3 == \"tokenspeed::sch\" || \$3 == \"smg_grpc_serv\" || \
        \$3 == \"ts-serve\" || \$3 == \"ts-control\") {print \$1}")
    [ -z "$pids" ] && break
    echo "$pids" | xargs -r kill -9 2>/dev/null || true
    sleep 2
  done
  rem=$(ps -eo pid,comm | grep -E "tokenspeed::sch|smg_grpc|ts-serve|ts-control" | grep -v defunct | grep -v grep | wc -l)
  echo "teardown: remaining_procs=$rem port8000=$(ss -tln 2>/dev/null | grep -c :8000) port8001=$(ss -tln 2>/dev/null | grep -c :8001)"
'
