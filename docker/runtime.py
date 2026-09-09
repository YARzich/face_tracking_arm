# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Container-side YAML adapter and camera calibration; never modify the input config."""

import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time

import yaml


CONFIG = Path('/input/hardware.yaml')
MODES = {'cpu': 'mono_cpu', 'stereo': 'stereo'}


def describe(config, mode):
    """Return only the host resources required by the selected camera mode."""
    if not isinstance(config, dict):
        raise ValueError('hardware.yaml: ожидается YAML с разделами настроек')
    for name in (MODES[mode], 'perception'):
        if not isinstance(config.get(name), dict):
            raise ValueError(f'{name}: ожидается раздел настроек')
    section = config[MODES[mode]]
    if section.get('source') not in ('usb', 'ros'):
        raise ValueError('source должен быть usb или ros')
    sides = ['left', 'right'] if mode == 'stereo' else ['left']
    for side in sides:
        camera = section.get(side)
        if not isinstance(camera, dict):
            raise ValueError(f'{MODES[mode]}.{side}: ожидается раздел камеры')
        if section['source'] == 'usb':
            for key in ('device', 'calibration_file'):
                if not isinstance(camera.get(key), str):
                    raise ValueError(f'{MODES[mode]}.{side}.{key}: ожидается строка')
    for key in ('model_dir', 'detector'):
        if not isinstance(config['perception'].get(key), str):
            raise ValueError(f'perception.{key}: ожидается строка')
    return {'source': section['source'], 'cameras': {s: section[s] for s in sides},
            'model_dir': config['perception']['model_dir'],
            'detector': config['perception']['detector']}


def resolved_config(config, mode, *, calibration=False, external_models=False):
    """Map host resources to their explicit container mounts."""
    config = copy.deepcopy(config)
    config['perception']['python_executable'] = '/opt/face_tracking_venv/bin/python'
    config['perception']['model_dir'] = (
        '/models' if external_models else '/opt/face_tracking_models')
    section = config[MODES[mode]]
    for index, side in enumerate(['left', 'right'] if mode == 'stereo' else ['left']):
        if section['source'] == 'usb':
            section[side]['device'] = f'/dev/video{index}'
            section[side]['calibration_file'] = (
                '' if calibration else f'/calibration/{side}.yaml')
    return config


def write_config(config, mode, *, calibration=False):
    path = Path('/tmp/hardware_resolved.yaml')
    resolved = resolved_config(config, mode, calibration=calibration,
                               external_models=Path('/models').is_dir())
    path.write_text(yaml.safe_dump(resolved))
    return path


def save_calibration(archive, destination, mode):
    """Extract only the calibration YAMLs, never archive paths or image files."""
    names = {'left.yaml': 'ost.yaml'} if mode == 'cpu' else {
        'left.yaml': 'left.yaml', 'right.yaml': 'right.yaml'}
    pending = {}
    with tarfile.open(archive, 'r:gz') as source:
        for output, name in names.items():
            member = source.getmember(name)
            if not member.isfile() or member.size > 1_000_000:
                raise ValueError('Некорректный файл калибровки')
            data = source.extractfile(member).read()
            if not isinstance(yaml.safe_load(data), dict):
                raise ValueError('Некорректный YAML калибровки')
            pending[output] = data
    for name, data in pending.items():
        temp = destination / (name + '.tmp')
        temp.write_bytes(data)
        temp.replace(destination / name)


def stop_children(children):
    for sig, timeout in ((signal.SIGINT, 8), (signal.SIGTERM, 3), (signal.SIGKILL, 1)):
        running = [child for child in children if child.poll() is None]
        for child in running:
            try:
                os.killpg(child.pid, sig)
            except ProcessLookupError:
                pass
        for child in running:
            try:
                child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass


def calibrate(config, args):
    if config[MODES[args.mode]]['source'] != 'usb':
        raise ValueError('Для source=ros используйте калибровку драйвера камеры')
    path = write_config(config, args.mode, calibration=True)
    settings = config[MODES[args.mode]]
    command = ['ros2', 'run', 'camera_calibration', 'cameracalibrator',
               '--size', args.size, '--square', str(args.square), '--no-service-check']
    if args.mode == 'cpu':
        command += ['--camera_name', settings['left']['camera_name'],
                    '--ros-args', '-r', 'image:=' + settings['left']['image_topic']]
    else:
        command += ['--approximate', str(settings['sync_slop_sec']), '--ros-args',
                    '-r', 'left:=' + settings['left']['image_topic'],
                    '-r', 'right:=' + settings['right']['image_topic']]
    children = []
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        children.append(subprocess.Popen([
            'ros2', 'launch', 'face_tracking_arm', 'hardware_camera.launch.py',
            f'config_file:={path}', 'camera_mode:=' + MODES[args.mode]], start_new_session=True))
        children.append(subprocess.Popen(command, start_new_session=True))
        while not stopping and all(child.poll() is None for child in children):
            time.sleep(.1)
    finally:
        stop_children(children)
    archive = Path('/tmp/calibrationdata.tar.gz')
    if not archive.exists():
        raise ValueError('Калибровка не сохранена: нажмите CALIBRATE, затем SAVE')
    save_calibration(archive, Path('/output'), args.mode)
    print('Калибровка сохранена в папке config/calibration/' + MODES[args.mode], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['describe', 'launch', 'calibrate', 'shell', 'help'])
    parser.add_argument('mode', choices=MODES, nargs='?', default='cpu')
    parser.add_argument('--camera-only', action='store_true')
    parser.add_argument('--show-image', action='store_true')
    parser.add_argument('--mock', action='store_true')
    parser.add_argument('--size', default='8x6')
    parser.add_argument('--square', type=float, default=.025)
    args = parser.parse_args()
    if args.operation == 'shell':
        os.execvp('bash', ['bash', '--noprofile', '--norc'])
    if args.operation == 'help':
        parser.print_help()
        return
    config = yaml.safe_load(CONFIG.read_text())
    if args.operation == 'describe':
        print(json.dumps(describe(config, args.mode)))
    elif args.operation == 'calibrate':
        calibrate(config, args)
    else:
        path = write_config(config, args.mode)
        os.execvp('ros2', [
            'ros2', 'launch', 'face_tracking_arm',
            'tracking_hardware_' + args.mode + '.launch.py', f'config_file:={path}',
            'camera_only:=' + str(args.camera_only).lower(),
            'show_image:=' + str(args.show_image).lower(),
            'mock_hardware:=' + str(args.mock).lower()])


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError, yaml.YAMLError, tarfile.TarError) as error:
        sys.exit('Ошибка: ' + str(error))
