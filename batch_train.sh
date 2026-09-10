#!/bin/bash
set -e

# ============================================================================
# Batch Street Gaussians Training Script
# ============================================================================
# Usage:
#   ./batch_train.sh <gpu_indices> <start_idx> <end_idx> <list_file>
#
# Arguments:
#   gpu_indices : Comma-separated list of GPU indices (e.g. "0,1,2,3")
#   start_idx   : Start index of the clip range (0-indexed, inclusive)
#   end_idx     : End index of the clip range (0-indexed, inclusive)
#   list_file   : Path to a text file with clip names (one per line)
#
# Example:
#   ./batch_train.sh "0,1,2,3" 0 15 ./clip_list.txt
#     -> Runs clips 0 through 15 from clip_list.txt on GPUs 0,1,2,3 (round-robin)
#
#   ./batch_train.sh "2" 10 10 ./clip_list.txt
#     -> Runs only clip 10 from clip_list.txt on GPU 2
# ============================================================================

if [ $# -ne 4 ]; then
    echo "Usage: $0 <gpu_indices> <start_idx> <end_idx> <list_file>"
    echo "Example: $0 \"0,1,2,3\" 0 15 ./clip_list.txt"
    exit 1
fi

# --- Parse arguments ---
IFS=',' read -ra GPU_INDICES <<< "$1"
START_IDX=$2
END_IDX=$3
LIST_FILE="$4"

# --- Paths ---
TEMPLATE_YAML="/mnt/yswang-wan22/code/street_gaussians-main-local/configs/example/L3_front.yaml"
TRAIN_SCRIPT="/mnt/yswang-wan22/code/street_gaussians-main-local/train.py"
DATASET_DIR="/mnt/yswang-wan22/dataset/streetgs_l3_frame100"
EXP_DIR="/mnt/yswang-wan22/exp/cycle_front4v_streetgs"

# --- Temp directory (unique per run) ---
TEMP_DIR="/tmp/streetgs_batch_$$"
mkdir -p "$EXP_DIR"
mkdir -p "$TEMP_DIR"

# --- Load clip list from file ---
if [ ! -f "$LIST_FILE" ]; then
    echo "ERROR: List file not found: $LIST_FILE"
    exit 1
fi

echo "==> Loading clip list from: $LIST_FILE"
mapfile -t ALL_CLIPS < <(grep -v '^\s*$' "$LIST_FILE" | sed 's/[[:space:]]*$//')
TOTAL_ALL=${#ALL_CLIPS[@]}
echo "    Total clips in list: $TOTAL_ALL"

# Validate range
if [ "$START_IDX" -lt 0 ] || [ "$END_IDX" -ge "$TOTAL_ALL" ] || [ "$START_IDX" -gt "$END_IDX" ]; then
    echo "ERROR: Invalid range [$START_IDX, $END_IDX]. Must be within [0, $((TOTAL_ALL - 1))]"
    exit 1
fi

COUNT=$((END_IDX - START_IDX + 1))
CLIPS=("${ALL_CLIPS[@]:$START_IDX:$COUNT}")
TOTAL=${#CLIPS[@]}

NUM_GPUS=${#GPU_INDICES[@]}
echo "==> Processing $TOTAL clips (index $START_IDX to $END_IDX)"
echo "==> Using ${NUM_GPUS} GPU(s): ${GPU_INDICES[*]}"
echo "==> Temp dir: $TEMP_DIR"
echo ""

# --- Queue coordination files ---
QUEUE_FILE="$TEMP_DIR/queue_counter"
LOCK_FILE="$TEMP_DIR/queue_lock"
echo "0" > "$QUEUE_FILE"

# --- Log dir ---
LOG_DIR="$TEMP_DIR/logs"
mkdir -p "$LOG_DIR"

# ============================================================================
# Atomically get the next job index from the shared queue.
# Returns the index, or -1 if no jobs remain.
# ============================================================================
get_next_idx() {
    local idx
    (
        flock -x 200
        idx=$(cat "$QUEUE_FILE")
        if [ "$idx" -ge "$TOTAL" ]; then
            echo "-1"
        else
            echo $((idx + 1)) > "$QUEUE_FILE"
            echo "$idx"
        fi
    ) 200>"$LOCK_FILE"
}

# ============================================================================
# Worker function — one per GPU.
# ============================================================================
run_worker() {
    local gpu=$1
    local gpu_log_dir="$LOG_DIR/gpu${gpu}"
    mkdir -p "$gpu_log_dir"

    while true; do
        local idx
        idx=$(get_next_idx)

        if [ "$idx" -eq -1 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $gpu: No more jobs, exiting."
            break
        fi

        local clip_name="${CLIPS[$idx]}"
        local source_path="$DATASET_DIR/$clip_name"
        local model_path="$EXP_DIR/${clip_name}_100f_front4v"
        local tmp_yaml="$TEMP_DIR/config_${clip_name}.yaml"

        # Generate per-clip YAML from template
        sed -e "s|source_path:.*|source_path: $source_path|" \
            -e "s|model_path:.*|model_path: $model_path|" \
            -e "s|gpus: \[[0-9, ]*\]|gpus: [$gpu]|" \
            "$TEMPLATE_YAML" > "$tmp_yaml"

        local log_file="$gpu_log_dir/${clip_name}.log"

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $gpu: [job $((idx + 1))/$TOTAL] Starting: $clip_name"

        CUDA_VISIBLE_DEVICES=$gpu python "$TRAIN_SCRIPT" --config "$tmp_yaml" \
            > "$log_file" 2>&1

        local exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $gpu: [job $((idx + 1))/$TOTAL] FINISHED: $clip_name"
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $gpu: [job $((idx + 1))/$TOTAL] FAILED (exit=$exit_code): $clip_name  -> see $log_file"
        fi
    done
}

# ============================================================================
# Launch one worker per GPU in parallel
# ============================================================================
echo "==> Launching ${NUM_GPUS} worker(s)..."
for gpu in "${GPU_INDICES[@]}"; do
    run_worker "$gpu" &
done

# Wait for all workers to complete
wait

echo ""
echo "============================================"
echo "==> All jobs completed!"
echo "==> Logs: $LOG_DIR"
echo "==> Temp configs: $TEMP_DIR"
echo "============================================"
