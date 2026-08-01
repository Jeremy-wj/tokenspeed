#!/usr/bin/env bash
# Tear down TokenSpeed serving and benchmark processes in the selected container.
# Graceful shutdown is required because the profiler container's PID 1 does not
# reap orphaned serving processes or release their listening sockets.
set -uo pipefail

CONTAINER="${CONTAINER:-${TOKENSPEED_CONTAINER:-jeremwan-tokenspeed-profiler}}"
docker exec "$CONTAINER" bash -lc '
  bench_pids=$(ps -eo pid=,args= | awk \
    "\$2 ~ /python/ && \$3 == \"-m\" && \$4 == \"tokenspeed.cli\" && \
     \$5 == \"bench\" && \$6 == \"serve\" {print \$1}")
  [ -z "$bench_pids" ] || kill -TERM $bench_pids 2>/dev/null || true

  pkill -INT  -x ts-serve   2>/dev/null || true
  pkill -TERM -x ts-serve   2>/dev/null || true
  pkill -TERM -x ts-control 2>/dev/null || true
  for _ in $(seq 1 12); do
    [ -z "$(pgrep -f ts-serve)" ] && break
    sleep 2
  done

  for name in "ts-serve" "ts-control" "tokenspeed::sch" "smg_grpc_serv"; do
    pkill -9 -x "$name" 2>/dev/null || true
  done
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
  rem=$(ps -eo stat=,comm= | awk \
    "\$1 !~ /Z/ && (\$2 == \"tokenspeed::sch\" || \$2 == \"smg_grpc_serv\" || \
     \$2 == \"ts-serve\" || \$2 == \"ts-control\") {count++} \
     END {print count + 0}")
  echo "teardown: remaining_procs=$rem port8000=$(ss -tln 2>/dev/null | grep -c :8000) port8001=$(ss -tln 2>/dev/null | grep -c :8001)"
'
