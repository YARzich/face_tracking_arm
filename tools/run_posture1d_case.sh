#!/usr/bin/env bash
# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  run_posture1d_case.sh arm SCENARIO fixed|inband|sequence DURATION OUTPUT [X Y Z|WAYPOINTS.json]
  run_posture1d_case.sh rgbd SCENARIO observe DURATION OUTPUT [START_POSE]

The runner uses the existing tracking_sim or tracking_vision_common launch and
starts a passive/source-stamped observer. It refuses to start if a Gazebo sim
is already running.
EOF
}

if (($# < 5)); then usage; exit 2; fi
launch_kind=$1; scenario=$2; observer_mode=$3; duration=$4; output=$5
shift 5
table_enabled=${POSTURE1D_TABLE_ENABLED:-true}
case "$table_enabled" in
  true|false) ;;
  *) echo 'POSTURE1D_TABLE_ENABLED must be true or false.' >&2; exit 2 ;;
esac

case "$launch_kind:$observer_mode" in
  arm:fixed|arm:inband|arm:sequence|rgbd:observe) ;;
  *) usage; exit 2 ;;
esac
rgbd_start_pose=
if [[ "$launch_kind" == rgbd && $# -eq 1 ]]; then
  case "$1" in
    rest|zero) rgbd_start_pose=$1 ;;
    *) usage; exit 2 ;;
  esac
elif [[ "$launch_kind" == rgbd && $# -ne 0 ]]; then
  usage
  exit 2
fi
if [[ "$launch_kind" == rgbd && $# -gt 1 ]] ||
   [[ "$launch_kind" == arm && "$observer_mode" == fixed && $# -ne 3 ]] ||
   [[ "$launch_kind" == arm && "$observer_mode" == inband && $# -gt 0 ]] ||
   [[ "$launch_kind" == arm && "$observer_mode" == sequence && $# -ne 1 ]]; then
  usage
  exit 2
fi
if pgrep -f '^gz sim( |$)' >/dev/null; then
  echo 'Gazebo is already running; refusing to start a second simulation.' >&2
  exit 1
fi

# ROS setup scripts reference intentionally unset variables under nounset.
set +u
source /opt/ros/jazzy/setup.bash
source /home/yaroslav/ros2_ws/install/setup.bash
set -u

if [[ "$launch_kind" == arm ]]; then
  setsid ros2 launch face_tracking_arm tracking_sim.launch.py \
    headless:=true start_pose:=rest camera_mode:=disabled \
    "table_enabled:=$table_enabled" \
    test_face_scenario:=disabled &
else
  rgbd_launch_args=(mode:=rgbd scenario:="$scenario" cycles:=1
    headless:=true show_image:=false)
  if [[ -n "$rgbd_start_pose" ]]; then
    rgbd_launch_args+=("start_pose:=$rgbd_start_pose")
  fi
  setsid ros2 launch face_tracking_arm tracking_vision_common.launch.py \
    "${rgbd_launch_args[@]}" &
fi
launch_pid=$!
cleanup() {
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
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

observer=(python3 "$(dirname "$0")/posture1d_observer.py"
  --scenario "$scenario" --mode "$observer_mode"
  --duration "$duration" --output "$output")
if [[ "$observer_mode" == fixed ]]; then
  observer+=(--target "$1" "$2" "$3")
fi
if [[ "$observer_mode" == sequence ]]; then
  observer+=(--sequence "$1")
fi
"${observer[@]}"
