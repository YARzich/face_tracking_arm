# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Host-side Docker orchestration. No ROS, YAML library or pip installation on the host."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import socket
import stat
import subprocess
import sys

from image_manifest import source_fingerprint


ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'face-tracking-arm:jazzy-v2'
CONTAINER = 'face-tracking-arm'
MODES = {'cpu': 'mono_cpu', 'stereo': 'stereo'}
MODELS = {'yunet': 'face_detection_yunet_2023mar.onnx',
          'yolov5n_face': 'yolov5n-face.onnx',
          'yolo_facev2n': 'yolo-facev2n-preweight.onnx'}


class ImageCompatibilityError(ValueError):
    """An installed image must be replaced by one for this checkout and architecture."""


def capture(command):
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def docker_connection():
    if not shutil.which('docker'):
        raise ValueError('Docker не установлен. Выполните шаг 1 в docs/hardware.md')
    context = capture(['docker', 'context', 'inspect', '--format', '{{.Endpoints.docker.Host}}'])
    endpoint = os.environ.get('DOCKER_HOST') or context.stdout.strip()
    if not endpoint.startswith('unix://'):
        raise ValueError('Нужен локальный Docker Engine на Linux, с unix:// подключением')
    command = ['docker', '--host', endpoint]
    result = capture(command + ['info', '--format', '{{json .}}'])
    if result.returncode and 'permission denied' in result.stderr.lower():
        print('Для доступа к Docker нужен sudo; введите пароль в терминале.', flush=True)
        subprocess.run(['sudo', '-v'], check=True)
        command = ['sudo', 'docker', '--host', endpoint]
        result = capture(command + ['info', '--format', '{{json .}}'])
    if result.returncode:
        raise ValueError('Docker недоступен: ' + result.stderr.strip())
    info = json.loads(result.stdout)
    if info['OSType'] != 'linux' or 'desktop' in info.get('OperatingSystem', '').lower():
        raise ValueError('Нужен Docker Engine на Linux; Docker Desktop не поддерживается')
    if any('rootless' in item for item in info.get('SecurityOptions', [])):
        raise ValueError('Для USB и общей сети нужен обычный Docker Engine, не rootless')
    return command


def architecture():
    try:
        return {'x86_64': 'amd64', 'aarch64': 'arm64'}[platform.machine()]
    except KeyError as error:
        raise ValueError('Поддерживаются компьютеры Intel/AMD 64-bit и ARM64') from error


def image_exists(docker):
    result = capture(docker + ['image', 'inspect', IMAGE])
    if result.returncode:
        return False
    image = json.loads(result.stdout)[0]
    if image['Architecture'] != architecture():
        raise ImageCompatibilityError(
            'Образ для другой архитектуры. Выполните ./run build на этом компьютере')
    labels = image['Config'].get('Labels') or {}
    if labels.get('io.face_tracking_arm.launch-schema') != '2':
        raise ImageCompatibilityError('Образ несовместим со скриптом. Выполните ./run build')
    if labels.get('io.face_tracking_arm.source-sha256') != source_fingerprint(ROOT):
        raise ImageCompatibilityError(
            'Docker-образ содержит другую или непроверенную версию программы. '
            'Выполните ./run build либо загрузите готовый образ для этой версии проекта. '
            'Изменения hardware.yaml и калибровок пересборки не требуют.')
    return True


def archive_path():
    return ROOT / 'dist' / f'face-tracking-arm-v2-{architecture()}.tar.gz'


def build(docker):
    source_argument = 'FACE_TRACKING_SOURCE_SHA256=' + source_fingerprint(ROOT)
    subprocess.run(docker + ['build', '--platform', 'linux/' + architecture(),
                             '--build-arg', source_argument,
                             '-f', str(ROOT / 'docker/Dockerfile'), '-t', IMAGE, str(ROOT)],
                   check=True)


def prepare(docker):
    incompatible = None
    try:
        if image_exists(docker):
            print('Образ готов. Заполните config/hardware.yaml '
                  'и запустите ./run cpu или ./run stereo')
            return
    except ImageCompatibilityError as error:
        incompatible = error
    archive = archive_path()
    if archive.is_file():
        checksum = archive.with_suffix(archive.suffix + '.sha256')
        expected = checksum.read_text().split() if checksum.is_file() else []
        if not expected or digest(archive) != expected[0]:
            raise ValueError('Архив повреждён или отсутствует файл .sha256 рядом с ним')
        subprocess.run(docker + ['image', 'load', '-i', str(archive)], check=True)
    elif incompatible:
        raise incompatible
    else:
        print('Готового архива нет. Собираю образ; первая сборка требует интернета.', flush=True)
        build(docker)
    if not image_exists(docker):
        raise ValueError('Подготовка не создала ожидаемый образ')


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def export_image(docker):
    prepare(docker)
    destination = archive_path()
    destination.parent.mkdir(exist_ok=True)
    temp = destination.with_suffix('.tmp')
    with temp.open('wb') as output:
        process = subprocess.Popen(docker + ['image', 'save', IMAGE], stdout=subprocess.PIPE)
        try:
            with gzip.GzipFile(fileobj=output, mode='wb', compresslevel=1) as zipped:
                shutil.copyfileobj(process.stdout, zipped, length=1024 * 1024)
        finally:
            process.stdout.close()
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=10)
        if process.wait():
            raise ValueError('Не удалось сохранить образ')
    temp.replace(destination)
    destination.with_suffix(destination.suffix + '.sha256').write_text(
        digest(destination) + '  ' + destination.name + '\n')
    print('Готовый образ: ' + str(destination))


def mount(source, destination, *, writable=False):
    buffer = io.StringIO()
    fields = ['type=bind', 'src=' + str(source), 'dst=' + destination]
    if not writable:
        fields.append('readonly')
    csv.writer(buffer, lineterminator='').writerow(fields)
    return ['--mount', buffer.getvalue()]


def host_path(value, config):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else config.parent / path).resolve()


def metadata(docker, config, mode):
    command = docker + ['run', '--rm', '--network', 'none', '--cap-drop', 'ALL',
                        '--user', f'{os.getuid()}:{os.getgid()}']
    command += mount(config, '/input/hardware.yaml')
    result = capture(command + [IMAGE, 'describe', mode])
    if result.returncode:
        raise ValueError(result.stderr.strip())
    return json.loads(result.stdout)


def resources(data, config, mode, *, calibration=False):
    """Translate selected camera paths into narrowly scoped mounts and device grants."""
    flags = mount(config, '/input/hardware.yaml')
    groups = set()
    devices = set()
    video_devices = {}
    if data['source'] == 'usb':
        for side, camera in data['cameras'].items():
            device = host_path(camera['device'], config)
            if not device.exists() or not stat.S_ISCHR(device.stat().st_mode):
                raise ValueError(f'Нет устройства камеры {device}. Посмотрите ./run devices')
            if device in devices:
                raise ValueError('Для стерео нужны два разных устройства камеры')
            devices.add(device)
            groups.add(device.stat().st_gid)
            # Vendor UVC controls identify hardware through the matching sysfs videoN.
            flags += ['--device', f'{device}:{device}:rw']
            video_devices[side] = str(device)
            if not calibration:
                value = camera['calibration_file'] or f'calibration/{MODES[mode]}/{side}.yaml'
                path = host_path(value, config)
                if not path.is_file():
                    command = shlex.join(['./run', 'calibrate', mode, '--config', str(config)])
                    raise ValueError(f'Нет калибровки {path}. Выполните {command}')
                flags += mount(path, f'/calibration/{side}.yaml')
        flags += ['--env', 'FACE_TRACKING_VIDEO_DEVICES=' + json.dumps(video_devices)]
    if calibration:
        if data['source'] != 'usb':
            raise ValueError('Для source=ros используйте калибровку драйвера камеры')
    elif not (data['detector'] == 'yunet' and data['model_dir'] in (
            '', '~/.cache/face_tracking_arm/models')):
        model_dir = host_path(data['model_dir'], config)
        if data['model_dir'] and model_dir.is_dir():
            if not (model_dir / MODELS[data['detector']]).is_file():
                raise ValueError(f'Нет выбранной модели в {model_dir}')
            flags += mount(model_dir / MODELS[data['detector']],
                           '/models/' + MODELS[data['detector']])
        else:
            raise ValueError('Укажите существующую папку perception.model_dir с выбранной моделью')
    for group in sorted(groups):
        flags += ['--group-add', str(group)]
    return flags


def display_options(environment):
    if not environment.get('DISPLAY'):
        raise ValueError('Нет графического экрана DISPLAY; запустите без --show-image')
    flags = ['--env', 'DISPLAY=' + environment['DISPLAY'],
             '--env', 'QT_X11_NO_MITSHM=1', '--env', 'QT_QPA_PLATFORM=xcb',
             '--hostname', socket.gethostname()]
    if Path('/tmp/.X11-unix').is_dir():
        flags += mount('/tmp/.X11-unix', '/tmp/.X11-unix')
    authority = Path(environment.get('XAUTHORITY') or str(Path.home() / '.Xauthority'))
    if authority.is_file():
        flags += mount(authority.resolve(), '/tmp/face_tracking.xauthority')
        flags += ['--env', 'XAUTHORITY=/tmp/face_tracking.xauthority']
    return flags


def execute(docker, args):
    config = args.config.expanduser().resolve()
    if not config.is_file():
        raise ValueError('Нет конфига: ' + str(config))
    mode = args.mode if args.operation == 'calibrate' else args.operation
    calibration = args.operation == 'calibrate'
    data = metadata(docker, config, mode)
    flags = resources(data, config, mode, calibration=calibration)
    if calibration:
        output = config.parent / 'calibration' / MODES[mode]
        output.mkdir(parents=True, exist_ok=True)
        flags += mount(output, '/output', writable=True)
    if args.show_image or calibration:
        flags += display_options(os.environ)
    if not calibration and not args.camera_only:
        # Permit the existing 50/40-priority control threads without host configuration.
        flags += ['--ulimit', 'rtprio=50', '--ulimit', 'memlock=-1']
    command = docker + ['run', '--rm', '--init', '--name', CONTAINER,
                        '--network', 'host', '--cap-drop', 'ALL',
                        '--security-opt', 'no-new-privileges',
                        '--user', f'{os.getuid()}:{os.getgid()}',
                        '--env', f'ROS_DOMAIN_ID={args.domain}',
                        '--stop-signal', 'SIGINT', '--stop-timeout', '20']
    command += flags + [IMAGE, 'calibrate' if calibration else 'launch', mode]
    if calibration:
        command += ['--size', args.size, '--square', str(args.square)]
    for option in ('camera_only', 'show_image', 'mock'):
        if getattr(args, option):
            command.append('--' + option.replace('_', '-'))
    if args.point_filter is not None:
        command += ['--point-filter', args.point_filter]
    # Handle Ctrl+C once here; the Docker client must not proxy the same signal again.
    process = subprocess.Popen(command, start_new_session=True)
    interrupted = False
    try:
        returncode = process.wait()
    except KeyboardInterrupt:
        capture(docker + ['stop', '--time', '20', CONTAINER])
        returncode = process.wait(timeout=25)
        interrupted = True
    if calibration and returncode == 0:
        print('Файлы калибровки: ' + str(output), flush=True)
        if any(camera.get('calibration_file') for camera in data['cameras'].values()):
            print('В конфиге уже указан calibration_file. Очистите его для новой '
                  'калибровки по умолчанию либо укажите путь к полученному файлу.', flush=True)
    return 130 if interrupted else returncode


def list_devices():
    paths = sorted(Path('/dev/v4l/by-id').glob('*')) or sorted(Path('/dev').glob('video*'))
    for path in paths:
        resolved = path.resolve()
        name = Path('/sys/class/video4linux') / resolved.name / 'name'
        print(f'{path}  {name.read_text().strip() if name.is_file() else ""}')
    if not paths:
        print('USB-камеры не найдены. Подключите камеру и повторите ./run devices')


def main():
    parser = argparse.ArgumentParser(description='Подготовка и запуск face_tracking_arm в Docker')
    parser.add_argument('operation', choices=['prepare', 'build', 'export', 'devices',
                                              'cpu', 'stereo', 'calibrate'])
    parser.add_argument('mode', nargs='?', choices=MODES, default='cpu')
    parser.add_argument('--config', type=Path, default=ROOT / 'config/hardware.yaml')
    parser.add_argument('--domain', type=int, default=71, help='ROS domain: по умолчанию 71')
    parser.add_argument('--camera-only', action='store_true', help='Проверить камеру без руки')
    parser.add_argument('--show-image', action='store_true', help='Открыть изображение камеры')
    parser.add_argument('--mock', action='store_true', help='Программный контроллер вместо руки')
    parser.add_argument('--point-filter', choices=['off', 'kalman', 'one_euro'],
                        help='Фильтр точки на этот запуск; без аргумента — из YAML')
    parser.add_argument('--size', default='8x6', help='Внутренние углы шахматной доски')
    parser.add_argument('--square', type=float, default=.025, help='Размер клетки доски в метрах')
    args = parser.parse_args()
    if args.point_filter is not None and args.operation not in MODES:
        parser.error('--point-filter используется только с cpu или stereo')
    if not 0 <= args.domain <= 101:
        parser.error('--domain должен быть от 0 до 101')
    if (not re.fullmatch(r'[0-9]+x[0-9]+', args.size)
            or min(map(int, args.size.split('x'))) < 2 or not 0 < args.square <= 1):
        parser.error('Проверьте --size (например 8x6) и --square (метры)')
    try:
        if args.operation == 'devices':
            list_devices()
            return 0
        docker = docker_connection()
        if args.operation in ('prepare', 'build', 'export'):
            {'prepare': prepare, 'build': build, 'export': export_image}[args.operation](docker)
            return 0
        if not image_exists(docker):
            raise ValueError('Сначала выполните ./run prepare')
        return execute(docker, args)
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        print('Ошибка: ' + str(error), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
