#!/bin/bash
# Launch the A/B in detached tmux sessions, one per GPU with disjoint blocks.
# Usage: bash launch_tmux.sh [s1|s2|all]   (default: all)
# Session names, GPUs and block split come from config.json ("execution.tmux_sessions").
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=$(python3 -c "import json;print(json.load(open('$EXP/config.json'))['repo'])")
NAMES=$(python3 - "$EXP/config.json" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
for name, s in cfg["execution"]["tmux_sessions"].items():
    print(name, s["gpu"], ",".join(str(b) for b in s["blocks"]), s.get("session_id", name[-1]))
PY
)

start() {  # name gpu blocks session_id
  tmux has-session -t "$1" 2>/dev/null && { echo "session $1 already exists"; return; }
  tmux new-session -d -s "$1" -c "$REPO" \
    "python3 $EXP/driver.py --session $4 --gpu $2 --blocks $3 2>&1 | tee -a $EXP/logs/driver_$1.log; echo DRIVER-EXIT; sleep 86400"
  echo "started tmux session $1 (gpu $2, blocks $3)"
}

want="${1:-all}"
i=0
while read -r name gpu blocks sid; do
  [ -z "$name" ] && continue
  i=$((i+1))
  case "$want" in
    all) start "$name" "$gpu" "$blocks" "$sid" ;;
    s$i) start "$name" "$gpu" "$blocks" "$sid" ;;
    "$name") start "$name" "$gpu" "$blocks" "$sid" ;;
  esac
done <<< "$NAMES"
tmux ls 2>/dev/null | grep -F "$(echo "$NAMES" | head -1 | cut -d' ' -f1 | sed 's/[0-9]*$//')"
