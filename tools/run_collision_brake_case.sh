#!/usr/bin/env bash
# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  run_collision_brake_case.sh SCENARIO fixed DURATION OUTPUT X Y Z
  run_collision_brake_case.sh SCENARIO inband DURATION OUTPUT

Run headless tracking_sim from rest with the same temporary configs and
source-stamped observer against the normal Gazebo table scene. Existing Gazebo
processes and output paths are refused.
EOF
}

if (($# < 4)); then usage; exit 2; fi
scenario=$1; mode=$2; duration=$3; output=$4; shift 4
case "$mode" in
  fixed) (($# == 3)) || { usage; exit 2; } ;;
  inband) (($# == 0)) || { usage; exit 2; } ;;
  *) usage; exit 2 ;;
esac
python3 - "$duration" "$mode" "$@" "$(dirname "$0")" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[-1])
from posture1d_sequence import validate_runner_preflight
validate_runner_preflight(sys.argv[1], sys.argv[2], sys.argv[3:-1])
PY

output=$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$output")
if [[ -e "$output" ]]; then
  echo "Output path already exists: $output" >&2
  exit 1
fi
if pgrep -f '^gz sim( |$)' >/dev/null; then
  echo 'Gazebo is already running; refusing to start a second simulation.' >&2
  exit 1
fi

mkdir -p "$(dirname "$output")"
temp_dir=$(mktemp -d "$(dirname "$output")/.collision_brake_config.XXXXXX")
cleanup_temp() { rm -rf -- "$temp_dir"; }
trap cleanup_temp EXIT

set +u
source /opt/ros/jazzy/setup.bash
source /home/yaroslav/ros2_ws/install/setup.bash
set -u

share=$(ros2 pkg prefix face_tracking_arm)/share/face_tracking_arm
python3 - "$share" "$temp_dir" <<'PY'
from pathlib import Path
import sys
import yaml
share, directory = map(Path, sys.argv[1:])
servo = yaml.safe_load((share / 'config/moveit/collision_aware_servo.yaml').read_text())
tracking = yaml.safe_load((share / 'config/tracking.yaml').read_text())
servo.setdefault('recovery', {})['mode'] = 'disabled'
params = tracking['face_tracking_controller']['ros__parameters']
params['return_to_rest_delay_sec'] = 3600.0
params['face_distance_min_m'] = 0.37
params['face_distance_max_m'] = 0.43
(directory / 'collision_aware_servo.yaml').write_text(yaml.safe_dump(servo, sort_keys=False))
(directory / 'tracking.yaml').write_text(yaml.safe_dump(tracking, sort_keys=False))
PY

setsid ros2 launch face_tracking_arm tracking_sim.launch.py \
  headless:=true robot_model:=xarm6 start_pose:=rest camera_mode:=disabled \
  idle_behavior:=rest require_perception_status:=false test_face_scenario:=disabled \
  "collision_aware_servo_config_file:=$temp_dir/collision_aware_servo.yaml" \
  "tracking_config_file:=$temp_dir/tracking.yaml" &
launch_pid=$!
cleanup_launch() {
  launch_pgid=$(ps -o pgid= -p "$launch_pid" | tr -d ' ')
  if [[ "$launch_pgid" == "$launch_pid" ]]; then
    kill -INT -- "-$launch_pgid" 2>/dev/null || true
    sleep 2
    if kill -0 -- "-$launch_pgid" 2>/dev/null; then
      kill -TERM -- "-$launch_pgid" 2>/dev/null || true
      sleep 2
    fi
    if kill -0 -- "-$launch_pgid" 2>/dev/null; then
      kill -KILL -- "-$launch_pgid" 2>/dev/null || true
    fi
  fi
  wait "$launch_pid" 2>/dev/null || true
}
trap 'cleanup_launch; cleanup_temp' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

observer=(python3 "$(dirname "$0")/posture1d_observer.py"
  --scenario "$scenario" --mode "$mode" --duration "$duration"
  --output "$output")
case "$mode" in
  fixed) observer+=(--target "$1" "$2" "$3") ;;
esac
"${observer[@]}"
mkdir -p "$output/config"
cp -- "$temp_dir/collision_aware_servo.yaml" "$output/config/"
cp -- "$temp_dir/tracking.yaml" "$output/config/"
