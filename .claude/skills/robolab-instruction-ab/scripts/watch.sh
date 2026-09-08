#!/usr/bin/env bash
# Live view of the A/B for a tmux pane: per-task tallies, driver tails, current runner progress.
# Usage: tmux new-session -d -s <name>_watch "bash <exp>/watch.sh"
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$EXP" || exit 1
SESSIONS=$(python3 -c "import json;print(' '.join(json.load(open('config.json'))['execution']['tmux_sessions']))")
while true; do
  clear
  echo "instruction A/B  |  $(date +%T)  |  sessions: $SESSIONS"
  python3 driver.py --csv-only >/dev/null 2>&1
  python3 status.py 2>/dev/null
  echo
  for s in $SESSIONS; do
    echo "--- driver $s:"; tail -n 2 "logs/driver_$s.log" 2>/dev/null | cut -c1-160
  done
  echo "--- runner progress:"
  for f in $(ls -t logs/s*_block*.log 2>/dev/null | head -2); do
    echo -n "  $(basename "$f"): "
    grep -o "\[RoboLab\] Running [A-Za-z_0-9]*: '[^']*'" "$f" | tail -1 | tr -d '\n'
    echo -n "  "
    tail -c 300 "$f" | tr '\r' '\n' | grep -oE "[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+" | tail -1
  done
  done_n=$(grep -hc "finished in" logs/driver_*.log 2>/dev/null | awk '{s+=$1} END{print s+0}')
  want_n=$(echo "$SESSIONS" | wc -w)
  [ "$done_n" -ge "$want_n" ] && { echo; echo ">>> ALL SESSIONS FINISHED"; echo "    python3 first_grab.py --exp $EXP   # verify the conditions differed"; echo "    .venv/bin/python analyze.py"; }
  sleep 30
done
