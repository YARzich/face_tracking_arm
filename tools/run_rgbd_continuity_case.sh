#!/usr/bin/env bash
# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT
set -euo pipefail

usage() {
  cat <<'EOF'
Run the 164 s stress scenario from rest while dropping selected source-stamped
depth windows through the test-only RGB-D relay. The capture directory must
not already exist. This starts one Gazebo launch and cleans up only its own
process groups.

Usage:
  run_rgbd_continuity_case.sh [CAPTURE_DIR]

Default capture:
  /home/yaroslav/ros2_ws/log/rgbd_continuity_implementation_20261026/runtime/stress_rest_hold
EOF
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
if (($# > 1)); then usage >&2; exit 2; fi

log_root=/home/yaroslav/ros2_ws/log/rgbd_continuity_implementation_20261026
capture_dir=$(realpath -m "${1:-"$log_root/runtime/stress_rest_hold"}")
audit_file="${capture_dir%/*}/stress_rest_hold_relay_audit.jsonl"
runtime_log="${capture_dir%/*}/stress_rest_hold_runtime.log"
capture_log="${capture_dir%/*}/stress_rest_hold_capture.log"
relay_pid=
launch_pid=
recorder_pid=
observer_pid=

if pgrep -f '^gz sim( |$)' >/dev/null; then
  echo 'Gazebo is already running; refusing to start another simulation.' >&2
  exit 1
fi
for path in "$capture_dir" "$audit_file" "$runtime_log" "$capture_log"; do
  if [[ -e $path ]]; then echo "Refusing to overwrite: $path" >&2; exit 1; fi
done

set +u
source /opt/ros/jazzy/setup.bash
source /home/yaroslav/ros2_ws/install/setup.bash
set -u

stop_group() {
  local pid=$1 pgid
  [[ -n $pid ]] || return 0
  pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)
  [[ $pgid == "$pid" ]] || return 0
  kill -INT -- "-$pgid" 2>/dev/null || true
  sleep 2
  if kill -0 -- "-$pgid" 2>/dev/null; then kill -TERM -- "-$pgid" 2>/dev/null || true; sleep 2; fi
  if kill -0 -- "-$pgid" 2>/dev/null; then kill -KILL -- "-$pgid" 2>/dev/null || true; fi
  wait "$pid" 2>/dev/null || true
}

cleanup() {
  stop_group "$observer_pid"
  stop_group "$recorder_pid"
  stop_group "$relay_pid"
  stop_group "$launch_pid"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "${capture_dir%/*}"
setsid python3 "$(dirname "$0")/relay_rgbd_test_depth.py" \
  --drop-window 95:95.6 --drop-window 97:97.6 --drop-window 99:99.6 \
  --output-audit-jsonl "$audit_file" --max-wall-sec 600 \
  >"${capture_dir%/*}/stress_rest_hold_relay.log" 2>&1 &
relay_pid=$!
setsid ros2 launch face_tracking_arm tracking_vision_common.launch.py \
  mode:=rgbd scenario:=stress cycles:=1 headless:=true show_image:=false \
  start_pose:=rest rgbd_depth_topic:=/face_test/test_depth \
  >"$runtime_log" 2>&1 &
launch_pid=$!
setsid python3 "$(dirname "$0")/record_rgbd_depth_runtime.py" \
  --output "$capture_dir" --scenario stress --duration-sim-sec 164 \
  --max-wall-sec 600 --sample-hz 1 --capture-depth-window 94:101 \
  >"$capture_log" 2>&1 &
recorder_pid=$!

for _ in {1..600}; do
  [[ -f "$capture_dir/manifest.json" ]] && break
  if ! kill -0 "$recorder_pid" 2>/dev/null; then
    echo "Recorder exited before writing manifest; see $capture_log" >&2
    exit 1
  fi
  sleep 0.1
done
if [[ ! -f "$capture_dir/manifest.json" ]]; then
  echo "Timed out waiting for recorder manifest; see $capture_log" >&2
  exit 1
fi

setsid python3 "$(dirname "$0")/posture1d_observer.py" \
  --scenario stress --mode observe --duration 164 \
  --output "$capture_dir/posture1d_observer" \
  >"${capture_dir%/*}/stress_rest_hold_observer.log" 2>&1 &
observer_pid=$!
if wait "$observer_pid"; then observer_pid=; else status=$?; exit "$status"; fi
if wait "$recorder_pid"; then recorder_pid=; else status=$?; exit "$status"; fi
