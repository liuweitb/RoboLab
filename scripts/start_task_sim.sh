#!/bin/bash
set -euo pipefail

# Resolve the repo root from this script's own location so the relative paths
# below (.env, .venv, policies/...) work no matter where the script is called from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# The uv installer drops the binary in ~/.local/bin, which non-login shells
# (cron, ssh one-shots, IDE terminals) do not pick up from ~/.profile.
case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *) PATH="$HOME/.local/bin:$PATH" ;;
esac

# Load environment variables from .env file, if present.
if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

# openpi checkout providing packages/openpi-client. Override in .env.
OPENPI_PATH="${OPENPI_PATH:-$(dirname "$REPO_ROOT")/openpi}"
# ECM checkout, needed by run_ecm.py. Optional: skipped if absent.
ECM_PATH="${ECM_PATH:-$(dirname "$REPO_ROOT")/ECM}"

DEVICE=cuda:6
POLICY=pi05
TASK=BananaInBowlTask
LIVESTREAM=2
INSTRUCTION_TYPE=vague
REMOTE_HOST=localhost
REMOTE_PORT=8000
SKIP_SYNC=0

usage() {
    echo "Usage: $0 [-p POLICY] [-l LIVESTREAM] [-d DEVICE] [-t TASK]"
    echo "          [-i INSTRUCTION_TYPE] [-r REMOTE_HOST] [--remote-port PORT]"
    echo "          [-s | --skip-sync]"
}

# Guard against a trailing flag with no value: a bare 'shift 2' when only one
# positional remains fails, leaves $# unchanged, and spins the loop forever.
need_val() {
    if [[ $# -lt 2 || -z "$2" ]]; then
        echo "[ERROR] $1 requires a value."
        usage
        exit 1
    fi
}

# Argument parse
while [[ $# -gt 0 ]]; do
    case $1 in
        -p | --policy)
            need_val "$@"
            POLICY="$2"
            shift 2
            ;;
        -l | --livestream)
            need_val "$@"
            LIVESTREAM="$2"
            shift 2
            ;;
        -d | --device)
            need_val "$@"
            DEVICE="$2"
            shift 2
            ;;
        -t | --task)
            need_val "$@"
            TASK="$2"
            shift 2
            ;;
        -i | --instruction-type)
            need_val "$@"
            INSTRUCTION_TYPE="$2"
            shift 2
            ;;
        -r | --remote-host)
            need_val "$@"
            REMOTE_HOST="$2"
            shift 2
            ;;
        --remote-port)
            need_val "$@"
            REMOTE_PORT="$2"
            shift 2
            ;;
        -s | --skip-sync)
            SKIP_SYNC=1
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            usage
            exit 1
            ;;
    esac
done

if ! [[ "$REMOTE_PORT" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] --remote-port must be a number, got '$REMOTE_PORT'."
    echo "        Did you mean: --remote-host $REMOTE_PORT ?"
    exit 1
fi

if curl -sf -m 5 -o /dev/null "http://$REMOTE_HOST:$REMOTE_PORT/healthz"; then
    echo "[INFO] Policy server healthy at $REMOTE_HOST:$REMOTE_PORT"
else
    echo "[WARN] No healthy policy server at $REMOTE_HOST:$REMOTE_PORT."
    echo "       The client will sit at 'Awaiting for server...' until one starts."
fi

# Another live run shares this .venv. 'uv sync' is exact — it rewrites
# site-packages underneath that process — so never sync while one is running.
LIVE_RUNS="$(pgrep -f 'policies/pi0_family/run.*\.py' 2>/dev/null || true)"
if [[ -n "$LIVE_RUNS" && "$SKIP_SYNC" == "0" ]]; then
    echo "[WARN] Another pi0_family run is live (pid $(echo "$LIVE_RUNS" | tr '\n' ' ')):"
    pgrep -af 'policies/pi0_family/run.*\.py' | cut -c1-120 | sed 's/^/       /'
    echo "       Skipping 'uv sync' — it would rewrite .venv under that process."
    echo "       Kill it first if you meant to re-sync: pkill -f pi0_family"
    SKIP_SYNC=1
fi

# Check if port 49100 is already held by a stale Isaac Sim process
if [[ "$LIVESTREAM" != "0" ]] && ss -lntp 2>/dev/null | grep -qE ":49100([[:space:]]|$)"; then
    echo "[WARN] Port 49100 is already held by a stale Isaac Sim:"
    ss -lntp 2>/dev/null | grep -E ":49100([[:space:]]|$)" | sed 's/^/       /'
    echo "       Livestream will be a coin flip until it is gone: pkill -f pi0_family"
fi

# Isaac Sim's RTX multigpumanager enumerates and initialises *every* visible
# CUDA device -- `--device` only steers PhysX/torch, not the renderer. On this
# host that means all 10 GPUs, mixing Turing (2080 Ti) with Ampere (A6000), and
# boot wedges partway through RTX init. Pin visibility to the one requested GPU
# and re-index it to cuda:0, which is what it becomes once the mask is applied.
RUN_DEVICE="$DEVICE"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && "$DEVICE" =~ ^cuda:([0-9]+)$ ]]; then
    export CUDA_VISIBLE_DEVICES="${BASH_REMATCH[1]}"
    RUN_DEVICE="cuda:0"
    echo "[INFO] Pinned CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (requested $DEVICE -> $RUN_DEVICE)."
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "[INFO] Honouring inherited CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES; passing --device $RUN_DEVICE."
fi

echo "[INFO] Repo root: $REPO_ROOT"

# shellcheck disable=SC1091
set +u; source .venv/bin/activate; set -u

if [[ "$SKIP_SYNC" == "0" ]]; then
    uv sync --extra isaac50
    # openpi-client is not in uv.lock, so the sync above evicts it. Reinstall
    # after syncing, never before, or run.py dies on 'import openpi_client'.
    if [[ ! -d "$OPENPI_PATH/packages/openpi-client" ]]; then
        echo "[ERROR] No openpi-client at '$OPENPI_PATH/packages/openpi-client'."
        echo "        Set OPENPI_PATH in $REPO_ROOT/.env to your openpi checkout."
        exit 1
    fi
    uv pip install -e "$OPENPI_PATH/packages/openpi-client"

    # Same story for ecm (run_ecm.py). --no-deps on purpose: ECM pins
    # torch==2.11.0 and tensorflow, which would trample the torch 2.7.0+cu128
    # that IsaacSim 5.0 needs. Its client path only uses openai/PIL/pydantic,
    # all of which the isaac50 extra already provides.
    if [[ -d "$ECM_PATH" ]]; then
        uv pip install --no-deps -e "$ECM_PATH"
    else
        echo "[INFO] No ECM checkout at '$ECM_PATH'; skipping (run_ecm.py will not work)."
    fi
fi

# Preflight the import that run.py needs. It happens at module scope, after
# Isaac Sim has booted and outside run.py's try/except, so a missing
# openpi_client costs a minute of startup and then hangs Kit on port 49100
# because simulation_app.close() is never reached.
if ! uv run --no-sync python -c "import openpi_client" 2>/dev/null; then
    echo "[ERROR] 'openpi_client' is not importable from .venv."
    echo "        Install it: uv pip install -e \$OPENPI_PATH/packages/openpi-client"
    echo "        (re-run without --skip-sync, with OPENPI_PATH set in .env)"
    exit 1
fi

uv run --no-sync python policies/pi0_family/run.py \
	--policy "$POLICY" \
	--livestream "$LIVESTREAM" \
	--device "$RUN_DEVICE" \
	--task "$TASK" \
	--instruction-type "$INSTRUCTION_TYPE" \
	--remote-host "$REMOTE_HOST" \
	--remote-port "$REMOTE_PORT"
