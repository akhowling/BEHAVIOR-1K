#!/usr/bin/env bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate behavior

echo "Using python: $(which python)"
python -c "import hydra; print('hydra ok')"

set -u

REPO_ROOT="$HOME/behavior-1k-solution"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/scripts:${PYTHONPATH:-}"
# TASK_JSONL="$REPO_ROOT/logs/behavior_tasks.jsonl"
SUPPORTED_TASKS_FILE="$REPO_ROOT/logs/pi05_supported_tasks.txt"


EVAL_PY="$HOME/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval.py"

MODEL_HOST="localhost"
INSTANCE_IDS=(0)

# More repeats means more logs per task/condition
# Clean baseline should only run once per task
CLEAN_REPEATS=1

# Different random seeds for corrupted runs
ATTACK_SEEDS=(17)

# name | action_random_prob | action_noise_sigma
ACTION_SWEEPS=(
  "action_noise_low|0.05|0.03"
  "action_noise_med|0.20|0.10"
  "action_noise_high|0.40|0.20"
)

# name | rgb_cutout_prob | rgb_cutout_num_boxes | rgb_cutout_box_frac
RGB_SWEEPS=(
  "rgb_cutout_small|1.0|1|0.10"
  "rgb_cutout_med|1.0|3|0.20"
  "rgb_cutout_large|1.0|5|0.30"
)

# Number of eval.py jobs running at the same time
MAX_JOBS="${MAX_JOBS:-2}"

# Timeout for each individual eval run
TIMEOUT_PER_RUN="${TIMEOUT_PER_RUN:-90m}"

# Start policy server from this script
START_SERVER="${START_SERVER:-1}"

SERVER_LOG="$REPO_ROOT/logs/pi05_server_parallel_$(date +%Y%m%d_%H%M%S).log"

SUMMARY_DIR="$REPO_ROOT/logs/pi05_collection_summaries"
mkdir -p "$SUMMARY_DIR"
SUMMARY="$SUMMARY_DIR/parallel_summary_$(date +%Y%m%d_%H%M%S).tsv"
LOCKFILE="$SUMMARY.lock"

echo -e "time\tcondition\ttask\tinstance\trepeat\tstatus\texit_code\trun_dir" > "$SUMMARY"

cd "$REPO_ROOT" || exit 1

# if [[ ! -f "$TASK_JSONL" ]]; then
#   echo "Missing task list: $TASK_JSONL"
#   echo "Run your dump script first."
#   exit 1
# fi

if [[ ! -f "$SUPPORTED_TASKS_FILE" ]]; then
  echo "Missing supported task list: $SUPPORTED_TASKS_FILE"
  exit 1
fi

if [[ ! -f "$EVAL_PY" ]]; then
  echo "Missing eval.py: $EVAL_PY"
  exit 1
fi

# mapfile -t TASKS < <(
#   python - <<PY
# import json
# from pathlib import Path

# path = Path("$TASK_JSONL")
# seen = set()

# with path.open() as f:
#     for line in f:
#         if not line.strip():
#             continue
#         row = json.loads(line)
#         task = row.get("activity_name")
#         if task and task not in seen:
#             seen.add(task)
#             print(task)
# PY
# )

mapfile -t TASKS < <(
  grep -v '^[[:space:]]*$' "$SUPPORTED_TASKS_FILE" | grep -vx 'make_microwave_popcorn'
)

echo "Found ${#TASKS[@]} tasks."
echo "Running up to MAX_JOBS=$MAX_JOBS eval jobs in parallel."
echo "Clean repeats: $CLEAN_REPEATS"
echo "Attack seeds: ${ATTACK_SEEDS[*]}"
echo "Action sweeps: ${#ACTION_SWEEPS[@]}"
echo "RGB sweeps: ${#RGB_SWEEPS[@]}"
echo "Summary: $SUMMARY"

SERVER_PID=""

cleanup() {
  echo
  echo "Cleaning up..."

  jobs -pr | xargs -r kill 2>/dev/null || true

  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "$START_SERVER" -eq 1 ]]; then
  echo "Starting PI server. Log: $SERVER_LOG"

  uv run scripts/serve_b1k.py \
    --task-checkpoint-mapping task_checkpoint_mapping_abs.json \
    policy:checkpoint \
    --policy.config pi_behavior_b1k_fast \
    --policy.dir /home/anuriha/models/checkpoint_1 \
    > "$SERVER_LOG" 2>&1 &

  SERVER_PID=$!

  echo "Server PID: $SERVER_PID"
  echo "Waiting for server to load..."
  sleep 60

  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server died during startup. Check:"
    echo "$SERVER_LOG"
    exit 1
  fi
else
  echo "START_SERVER=0, assuming server already running."
fi

write_summary_line() {
  local condition="$1"
  local task="$2"
  local instance_id="$3"
  local repeat_id="$4"
  local status="$5"
  local exit_code="$6"
  local run_dir="$7"

  {
    flock 200
    echo -e "$(date +%Y-%m-%d_%H:%M:%S)\t$condition\t$task\t$instance_id\t$repeat_id\t$status\t$exit_code\t$run_dir" >> "$SUMMARY"
  } 200>"$LOCKFILE"
}

should_skip_job() {
  local condition="$1"
  local task="$2"
  local repeat_id="$3"

  # Already ran these manually/currently
  if [[ "$task" == "picking_up_trash" && "$condition" == "clean" && "$repeat_id" == "1" ]]; then
    return 0
  fi

  if [[ "$task" == "picking_up_trash" && "$condition" == "action_noise_low" && "$repeat_id" == "action_noise_low_seed_17" ]]; then
    return 0
  fi

  return 1
}

run_eval_job() {
  local condition="$1"
  local log_root="$2"
  local task="$3"
  local instance_id="$4"
  local repeat_id="$5"
  shift 5
  local attack_args=("$@")

  local ts
  ts="$(date +%Y%m%d_%H%M%S_%N)"
  local eval_log_path

  if [[ "$condition" == "clean" ]]; then
    eval_log_path="$log_root"
  else
    eval_log_path="$log_root/$repeat_id"
  fi

  local run_dir="$REPO_ROOT/logs/pi05_run_commands/$condition/$task/instance_${instance_id}/rep_${repeat_id}_${ts}"

  mkdir -p "$run_dir"
  mkdir -p "$eval_log_path"

  local cmd=(
    python "$EVAL_PY"
    "log_path=$eval_log_path"
    "policy=websocket"
    "task.name=$task"
    "model.host=$MODEL_HOST"
    "eval_instance_ids=[$instance_id]"
    "${attack_args[@]}"
  )

  {
    echo "condition: $condition"
    echo "task: $task"
    echo "instance_id: $instance_id"
    echo "repeat_id: $repeat_id"
    echo "started_at: $(date)"
    echo "run_dir: $run_dir"
    echo "eval_log_path: $eval_log_path"
    echo
    printf 'command: '
    printf '%q ' "${cmd[@]}"
    echo
  } > "$run_dir/command.txt"

  echo "[START] $condition | $task | instance=$instance_id | rep=$repeat_id"

  timeout "$TIMEOUT_PER_RUN" "${cmd[@]}" \
    > "$run_dir/stdout.log" \
    2> "$run_dir/stderr.log"

  local exit_code=$?
  local status

  if [[ "$exit_code" -eq 0 ]]; then
    status="success"
  elif [[ "$exit_code" -eq 124 ]]; then
    status="timeout"
  else
    status="failed"
  fi

  {
    echo
    echo "finished_at: $(date)"
    echo "exit_code: $exit_code"
    echo "status: $status"
  } >> "$run_dir/command.txt"

  write_summary_line "$condition" "$task" "$instance_id" "$repeat_id" "$status" "$exit_code" "$run_dir"

  echo "[DONE]  $condition | $task | rep=$repeat_id | $status"
}

# Semaphore for limiting parallel jobs
FIFO="/tmp/pi05_parallel_fifo_$$"
mkfifo "$FIFO"
exec 9<>"$FIFO"
rm "$FIFO"

for ((i=0; i<MAX_JOBS; i++)); do
  echo >&9
done

submit_job() {
  local condition="$1"
  local log_root="$2"
  local task="$3"
  local instance_id="$4"
  local repeat_id="$5"

  if should_skip_job "$condition" "$task" "$repeat_id"; then
    echo "[SKIP]  $condition | $task | instance=$instance_id | rep=$repeat_id"
    write_summary_line "$condition" "$task" "$instance_id" "$repeat_id" "skipped_existing" "0" "already_ran_before_restart"
    return 0
  fi

  read -r -u 9

  {
    run_eval_job "$@"
    echo >&9
  } &
}

for task in "${TASKS[@]}"; do
  for instance_id in "${INSTANCE_IDS[@]}"; do

    # Clean baseline: only one run per task
    for clean_rep in $(seq 1 "$CLEAN_REPEATS"); do
      submit_job \
        "clean" \
        "$REPO_ROOT/logs/pi05_eval_clean" \
        "$task" \
        "$instance_id" \
        "$clean_rep" \
        "+attack_mode=none"
    done

    # Action noise sweeps
    for seed in "${ATTACK_SEEDS[@]}"; do
      for spec in "${ACTION_SWEEPS[@]}"; do
        IFS='|' read -r variant random_prob noise_sigma <<< "$spec"

        submit_job \
          "$variant" \
          "$REPO_ROOT/logs/pi05_eval_action_noise" \
          "$task" \
          "$instance_id" \
          "${variant}_seed_${seed}" \
          "+attack_mode=action_noise" \
          "+attack_seed=$seed" \
          "+action_random_prob=$random_prob" \
          "+action_noise_sigma=$noise_sigma"
      done
    done

    # RGB cutout sweeps
    for seed in "${ATTACK_SEEDS[@]}"; do
      for spec in "${RGB_SWEEPS[@]}"; do
        IFS='|' read -r variant cutout_prob num_boxes box_frac <<< "$spec"

        submit_job \
          "$variant" \
          "$REPO_ROOT/logs/pi05_eval_rgb_cutout" \
          "$task" \
          "$instance_id" \
          "${variant}_seed_${seed}" \
          "+attack_mode=rgb_cutout" \
          "+attack_seed=$seed" \
          "+rgb_cutout_prob=$cutout_prob" \
          "+rgb_cutout_num_boxes=$num_boxes" \
          "+rgb_cutout_box_frac=$box_frac"
      done
    done

  done
done

wait

echo
echo "All parallel runs finished."
echo "Summary:"
echo "$SUMMARY"
