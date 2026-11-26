#!/bin/bash
# Run a command (e.g. llama-server) and kill it if the machine starts running out of memory:
# swap grows by more than 4 GB, or free+inactive+purgeable RAM drops below MIN_FREE_GB (default 6).
# Samples memory every 3 s into watchdog.csv. macOS only (vm_stat, sysctl vm.swapusage).
#   tools/watchdog.sh llama-server -m model.gguf ...
MIN_FREE_GB=${MIN_FREE_GB:-6}
swap_mb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6)}'; }
swap0=$(swap_mb)
"$@" &
pid=$!
echo "t_s,rss_gb,avail_gb,swap_used_gb" > watchdog.csv
t0=$(date +%s)
while kill -0 $pid 2>/dev/null; do
  vs=$(vm_stat); page=$(sysctl -n hw.pagesize)
  pg() { echo "$vs" | awk -v k="$1" '$0 ~ k {gsub(/\./,"",$NF); print $NF}'; }
  avail=$(( ( $(pg "Pages free") + $(pg "Pages inactive") + $(pg "Pages purgeable") ) * page / 1073741824 ))
  swap=$(swap_mb)
  rss=$(( $(ps -o rss= -p $pid 2>/dev/null || echo 0) / 1048576 ))
  echo "$(( $(date +%s)-t0 )),$rss,$avail,$(( swap/1024 ))" >> watchdog.csv
  if [ $(( swap - swap0 )) -gt 4096 ] || [ "$avail" -lt "$MIN_FREE_GB" ]; then
    echo "watchdog: stopping $pid (free ${avail} GB, swap +$(( (swap-swap0)/1024 )) GB)" >&2
    kill $pid; sleep 5; kill -9 $pid 2>/dev/null
    exit 1
  fi
  sleep 3
done
wait $pid
