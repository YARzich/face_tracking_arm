#!/usr/bin/env bash
# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

# Запускает соседний ./run и по умолчанию показывает изображение камеры.
set -u

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
run_path="$script_dir/run"

usage() {
    cat <<'HELP'
Запуск face_tracking_arm:
  ./start.sh [rgb|rgbd|stereo|cpu] [параметры run]

Без режима запускается rgbd. Режимы rgb и cpu означают обычную RGB-камеру.
Изображение включено по умолчанию; --no-image выключает его,
--show-image включает. При нескольких таких флагах действует последний.
Параметры run, например --config, --domain, --point-filter,
--camera-only и --mock, передаются без изменений.
Скрипт использует готовый Docker-образ и сам не запускает сборку.

Примеры:
  ./start.sh
  ./start.sh stereo --config 'profiles/bench setup.yaml' --no-image

  -h, --help       показать эту справку
HELP
}

# Справка локальная: для неё не нужен Docker и не требуется рабочая директория проекта.
args=()
mode=rgbd
show_image=1
camera_only=0
mock=0
want_help=0

# Первый обычный аргумент задаёт режим; значения остальных параметров сохраняются дословно.
if (($#)) && [[ "$1" != -* ]]; then
    case "$1" in
        rgb|cpu) mode=cpu ;;
        rgbd|stereo) mode="$1" ;;
        *) printf 'Ошибка: неизвестный режим: %s\n' "$1" >&2; exit 2 ;;
    esac
    shift
fi

while (($#)); do
    case "$1" in
        --)
            if (($# > 1)); then
                printf 'Ошибка: после -- недопустимы дополнительные позиционные аргументы\n' >&2
                exit 2
            fi
            break
            ;;
        --config|--domain|--point-filter|--size|--square|--action|--marker-side-mm)
            args+=("$1")
            if (($# > 1)); then
                args+=("$2")
                shift 2
            else
                shift
            fi
            ;;
        --config=*|--domain=*|--point-filter=*) args+=("$1"); shift ;;
        --show-image) show_image=1; shift ;;
        --no-image) show_image=0; shift ;;
        -h|--help) want_help=1; shift ;;
        --camera-only) camera_only=1; args+=("$1"); shift ;;
        --mock) mock=1; args+=("$1"); shift ;;
        -* ) args+=("$1"); shift ;;
        *)
            printf 'Ошибка: лишний позиционный аргумент: %s\n' "$1" >&2
            exit 2
            ;;
    esac
done

if ((want_help)); then
    usage
    exit 0
fi

if [[ ! -x "$run_path" ]]; then
    printf 'Ошибка: рядом с start.sh не найден исполняемый run: %s\n' "$run_path" >&2
    exit 1
fi

if ((show_image)); then
    args+=(--show-image)
    image_status=включено
else
    image_status=выключено
fi

printf 'Запуск: режим=%s; изображение=%s; ' "$mode" "$image_status"
if ((camera_only)); then
    printf 'только камера\n'
elif ((mock)); then
    printf 'mock-контроллер, без реальной руки\n'
else
    printf 'работа с рукой\n'
fi

# exec сохраняет сигналы и код завершения дочернего ./run.
exec "$run_path" "$mode" "${args[@]}"
