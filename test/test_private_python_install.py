# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Private ROS Python modules remain importable from installed entry points."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOOK_PATH = PROJECT_ROOT / 'env-hooks' / 'face_tracking_arm_pythonpath.dsv'
HOOK_REGISTRATION = 'ament_environment_hooks(env-hooks/face_tracking_arm_pythonpath.dsv)'


def test_installed_entrypoint_imports_private_module_with_environment_hook():
    """The package hook exposes lib/<package>, regardless of entrypoint symlinks."""
    cmake = (PROJECT_ROOT / 'CMakeLists.txt').read_text()
    assert HOOK_REGISTRATION in cmake, 'ament hook is not registered by CMake'
    assert HOOK_PATH.is_file(), 'ament hook for private Python modules is missing'
    operation, variable, relative_module_path = HOOK_PATH.read_text().strip().split(';')
    assert operation == 'prepend-non-duplicate'
    assert variable == 'PYTHONPATH'

    for symlink_entrypoint in (False, True):
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = Path(temp_dir)
            package_lib = prefix / 'lib' / 'face_tracking_arm'
            package_lib.mkdir(parents=True)
            (package_lib / 'face_tracking_perception').mkdir()
            (package_lib / 'face_tracking_perception' / '__init__.py').write_text(
                "MARKER = 'loaded'\n"
            )
            source_entrypoint = prefix / 'entrypoint_source.py'
            source_entrypoint.write_text(
                'from face_tracking_perception import MARKER\n'
                "assert MARKER == 'loaded'\n"
            )
            installed_entrypoint = package_lib / 'entrypoint.py'
            if symlink_entrypoint:
                installed_entrypoint.symlink_to(source_entrypoint)
            else:
                installed_entrypoint.write_bytes(source_entrypoint.read_bytes())

            # A copied launcher sees its installed sibling module; a symlinked
            # launcher resolves to the source directory and needs the hook.
            env = os.environ.copy()
            env['PYTHONPATH'] = ''
            result = subprocess.run(
                [sys.executable, str(installed_entrypoint)],
                check=False,
                env=env,
                cwd=prefix,
                capture_output=True,
                text=True,
            )
            if symlink_entrypoint:
                assert result.returncode != 0
                assert 'ModuleNotFoundError' in result.stderr
            else:
                assert result.returncode == 0, result.stderr

            # Use the module path declared by the ament hook.
            env['PYTHONPATH'] = os.pathsep.join(
                [str(prefix / relative_module_path), env.get('PYTHONPATH', '')]
            )
            subprocess.run(
                [sys.executable, str(installed_entrypoint)],
                check=True,
                env=env,
                cwd=prefix,
                capture_output=True,
                text=True,
            )
