# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

# Perception pipeline health and face availability for motion interlocking.
# header.stamp is the publication time; source_stamp is the last processed
# camera capture time and must not be refreshed by a watchdog publication.
uint8 WAIT=0
uint8 OK=1
uint8 NO_FACE=2
uint8 FACE_NO_DEPTH=3
uint8 DEPTH_STREAM_LOST=4
uint8 SYNC_ERROR=5
uint8 INVALID_INPUT=6
uint8 TF_UNAVAILABLE=7

std_msgs/Header header
builtin_interfaces/Time source_stamp
uint8 state
string reason
# Internal selector identity; -1 before an identity has been selected.
# NO_FACE retains the last selected identity until a different face is selected.
int64 target_id
