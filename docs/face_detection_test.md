# Найти лицо в Gazebo: тест 2D

Этот тест запускает Gazebo, RGB-камеру и один детектор YuNet. Детектор находит
рамки лица в пикселях. Он не оценивает расстояние и не запускает руку.

## Запуск

Сначала выполните [подготовку](#подготовка). Затем откройте терминал:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch face_tracking_arm face_detection_test.launch.py detector:=yunet
```

Gazebo показывает модель человека. RViz показывает кадр, рамки лиц, центр
рамки, уверенность и время обработки. Камера неподвижна. Перемещайте в Gazebo
только человека.

Чтобы закрыть тест, нажмите Ctrl+C. Аргумент `headless:=true` отключает оба
окна. Аргумент `show_image:=false` отключает только RViz.

Параметры запуска:

| Аргумент | По умолчанию | Назначение |
|---|---|---|
| `detector` | `yunet` | `yunet`, `yolov5n_face` или `yolo_facev2n` |
| `confidence` | `0.6` | Минимальная уверенность детектора |
| `threads` | `2` | Потоки CPU для OpenCV |
| `model_dir` | `~/.cache/face_tracking_arm/models` | Каталог ONNX-файлов |
| `assets_dir` | `~/.cache/face_tracking_arm/gazebo_models` | Модель человека и текстуры |
| `python_executable` | `~/.cache/face_tracking_arm/perception_venv/bin/python` | Python с зависимостями |
| `headless` | `false` | Отключить оба окна |
| `show_image` | `true` | Показать окно RViz |

Один запуск использует один детектор. Завершите текущий запуск клавишами Ctrl+C,
прежде чем выбирать другую модель. Для проверки замените значение detector на
yolov5n_face или yolo_facev2n.

## Проверить результат

Проверьте рамки и метки кадра:

```bash
ros2 topic echo /face_test/detections --qos-reliability best_effort
```

Топик `/face_test/detections` имеет тип `vision_msgs/Detection2DArray`.
Координаты центра и размеры рамки заданы в пикселях исходного изображения.
X растёт вправо, Y — вниз. Пустой массив означает, что детектор не нашёл лицо.
Если лиц несколько, сообщение содержит все рамки.

Топик `/face_test/debug_image` содержит изображение с рамками. Узел публикует
его только при наличии подписчика. Топики `/face_test/image_raw` и
`/face_test/camera_info` содержат RGB-кадр и параметры камеры.

Кадр использует `face_test_camera_optical_frame`: X направлена вправо,
Y — вниз, Z — вперёд. Изображение и рамки сохраняют время `stamp` кадра и его
`frame_id`. Очередь имеет глубину 1. Медленная обработка пропускает кадры,
а не накапливает их. Сравнивайте метки времени, чтобы заметить устаревший
результат или остановку камеры.

Этот тест публикует только данные 2D. Он не публикует `/face/center` и не
проверяет оценку глубины. Для 3D-точки используйте
[тест оценки глубины](face_depth_tests.md).

## Подготовка

Соберите пакет по [инструкции](../CONTRIBUTING.md). Из корня репозитория
создайте отдельное Python-окружение и установите зависимости:

```bash
python3 -m venv --system-site-packages ~/.cache/face_tracking_arm/perception_venv
~/.cache/face_tracking_arm/perception_venv/bin/python -m pip install \
  -r config/perception_requirements.txt
python3 scripts/prepare_face_test_assets.py
```

Скрипт загружает модель человека и текстуры, около 25 МБ, в
`~/.cache/face_tracking_arm/gazebo_models`. Версия модели закреплена; скрипт
записывает источник в каталог.

Загрузите YuNet и проверьте SHA-256:

```bash
mkdir -p ~/.cache/face_tracking_arm/models
curl --fail --location --retry 3 \
  https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
  -o ~/.cache/face_tracking_arm/models/face_detection_yunet_2023mar.onnx
printf '%s  %s\n' \
  8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4 \
  "$HOME/.cache/face_tracking_arm/models/face_detection_yunet_2023mar.onnx" | sha256sum -c -
```

Проверка должна вывести `OK`. YuNet нужен для обычного запуска. YOLO-веса
не входят в это окружение и для YuNet не нужны.

## Другие ONNX-модели

`yolov5n_face` и `yolo_facev2n` требуют ONNX-файл в `model_dir`:
`yolov5n-face.onnx` или `yolo-facev2n-preweight.onnx`. Экспорт совместимого
FaceV2 checkpoint описан в [отдельном руководстве](facev2_export.md).

Проверенный FaceV2 checkpoint в этой сцене не выдаёт детекций. Для обычного
использования выберите YuNet. Условия использования моделей указаны в
[списке сторонних ресурсов](../THIRD_PARTY_NOTICES.md).
