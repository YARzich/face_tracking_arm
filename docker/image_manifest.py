# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Identify the program bundled in an image, independently of Git and user settings."""

import hashlib
from pathlib import Path


# Runtime configurations can live anywhere under config, including nested
# profiles with their own calibration/models. Never discover image inputs there.
# Add a file here when application code starts reading a new bundled default.
PROGRAM_CONFIG_FILES = (
    'config/perception_requirements.txt',
    'config/tracking.yaml', 'config/face_appearances.yaml', 'config/face_detection_bridge.yaml',
    'config/face_depth_view.rviz', 'config/face_detection_view.rviz',
    'config/tracking_vision_view.rviz', 'config/tracking_vision_gazebo.xml',
    'config/control/controllers.yaml', 'config/control/initial_positions.yaml',
    'config/control/lite6_initial_positions.yaml',
    'config/control/start_poses/lite6_incident.yaml',
    'config/control/start_poses/lite6_joint1_limit.yaml',
    'config/control/start_poses/xarm6_folded.yaml',
    'config/control/start_poses/xarm6_joint1_limit.yaml',
    'config/control/start_poses/zero.yaml',
    'config/moveit/servo.yaml', 'config/moveit/collision_aware_servo.yaml',
    'config/moveit/joint_limits.yaml', 'config/moveit/kinematics.yaml',
    'config/moveit/lite6.srdf', 'config/moveit/xarm6.srdf',
)
# Match program COPY inputs in Dockerfile. Tests check coverage when COPY changes.
INPUT_FILES = (
    '.dockerignore', 'docker/Dockerfile', 'docker/entrypoint.sh', 'docker/runtime.py',
    'docker/fastdds.xml', 'CMakeLists.txt', 'package.xml', 'LICENSE', 'THIRD_PARTY_NOTICES.md',
) + PROGRAM_CONFIG_FILES
INPUT_DIRECTORIES = (
    'description', 'include', 'msg', 'src', 'python', 'scripts', 'launch', 'worlds',
    'docker/patches', 'docker/licenses',
)
CACHE_DIRECTORIES = frozenset((
    '__pycache__', '.cache', '.pytest_cache', '.mypy_cache', '.ruff_cache',
))
# Recursive file exclusions in .dockerignore, independent of a file's location.
IGNORED_SUFFIXES = frozenset((
    '.pyc', '.pyo', '.pyd', '.swp', '.swo', '.pem', '.key',
    '.onnx', '.pt', '.pth', '.safetensors', '.engine', '.bag', '.db3', '.mcap',
))


def excluded(path):
    """Exclude external hardware settings and transient files absent from the image."""
    return (any(part in CACHE_DIRECTORIES for part in path.parts)
            or path.suffix in IGNORED_SUFFIXES
            or path.name.endswith('~') or path.name in ('.DS_Store', 'Thumbs.db'))


def source_fingerprint(root):
    """Hash sorted relative names and contents; location, mtimes and Git do not matter."""
    root = Path(root)
    paths = {root / name for name in INPUT_FILES}
    for name in INPUT_DIRECTORIES:
        paths.update(path for path in (root / name).rglob('*') if path.is_file())
    checksum = hashlib.sha256(b'face_tracking_arm image inputs v1\0')
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root)
        if excluded(relative):
            continue
        checksum.update(relative.as_posix().encode() + b'\0')
        file_checksum = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                file_checksum.update(block)
        checksum.update(file_checksum.digest())
    return checksum.hexdigest()
