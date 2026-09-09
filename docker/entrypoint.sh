#!/usr/bin/env bash
# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

set -e
source /opt/face_tracking_ws/install/setup.bash
mkdir -p "$HOME"
exec python3 /opt/face_tracking_container/runtime.py "$@"
