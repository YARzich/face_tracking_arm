# Отдельный тест лица в Gazebo Harmonic

Сцена содержит статического человека OSRF Standing person и RGB-камеру
640×480, 30 кадров/с на высоте 1.65 м. Один выбранный детектор обрабатывает
изображение на CPU. Рамка и красная точка показывают результат нейросети;
поза человека из симулятора детектору неизвестна.

## Запуск

При первом запуске выполните [подготовку ниже](#подготовка).

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch face_tracking_arm face_detection_test.launch.py detector:=yunet
```

Открываются Gazebo и RViz с панелью изображения, рамкой, центральной точкой,
уверенностью и временем обработки. В Gazebo можно перемещать и поворачивать
**человека** стандартным инструментом Transform. Камера фиксирована; её
статический TF соответствует позе в SDF.

Другие модели выбираются при новом запуске:

```bash
ros2 launch face_tracking_arm face_detection_test.launch.py detector:=yolov5n_face
ros2 launch face_tracking_arm face_detection_test.launch.py detector:=yolo_facev2n
```

Предыдущий запуск сначала завершить Ctrl+C. `headless:=true` отключает оба
графических окна, сохраняя рендеринг камеры и ROS-выходы. `show_image:=false`
отключает только окно изображения. Дополнительные аргументы:

| Аргумент | По умолчанию | Назначение |
|---|---|---|
| `confidence` | `0.6` | Минимальная уверенность обнаружения |
| `threads` | `2` | Потоки CPU для расчёта модели |
| `model_dir` | `~/.cache/face_tracking_arm/models` | Каталог весов |
| `assets_dir` | `~/.cache/face_tracking_arm/gazebo_models` | Каталог модели человека |
| `python_executable` | `~/.cache/face_tracking_arm/perception_venv/bin/python` | Python с зависимостями |

## ROS-контракт

- `/face_test/image_raw`: `sensor_msgs/Image`, RGB из Gazebo.
- `/face_test/camera_info`: `sensor_msgs/CameraInfo`, параметры камеры.
- `/face_test/detections`: `vision_msgs/Detection2DArray`; каждая рамка содержит
  `bbox.center.position.x/y` — **точку в пикселях исходного изображения**,
  `bbox.size_x/y` — размер рамки, `results[0].hypothesis.score` — уверенность.
  Ось X направлена вправо, Y вниз. Пустой массив явно означает отсутствие лица.
  При нескольких лицах публикуются все рамки. Выбор ближайшего человека
  выполняется в [3D-сценариях](face_depth_tests.md).
- `/face_test/debug_image`: `sensor_msgs/Image`, результат с разметкой.
  Рисуется только при наличии подписчиков.
- `/clock`, `/tf_static`: время симуляции и неподвижная камера.

Детекции и изображение результата сохраняют `stamp` и `frame_id` исходного
кадра. Frame: `face_test_camera_optical_frame`; TF-цепочка:
`world → face_test_camera_link → face_test_camera_optical_frame`.
Все ROS-ноды используют симуляционное время. Подписка на изображения и выходы
детектора имеют `best_effort`, `keep_last=1`: медленный расчёт пропускает кадры,
а не накапливает очередь. Потребитель обязан проверять свежесть stamp: при
остановке камеры новые результаты не появляются.

```bash
ros2 topic echo /face_test/detections --qos-reliability best_effort
```

Это только 2D-детектор: расстояние не оценивается, `/face/center` не публикуется,
кинематический сценарий не запускается.

## Подготовка

Выполните [сборку пакета и установку ROS-зависимостей](../CONTRIBUTING.md).
Затем из папки репозитория:

```bash
python3 -m venv --system-site-packages ~/.cache/face_tracking_arm/perception_venv
~/.cache/face_tracking_arm/perception_venv/bin/python -m pip install -r config/perception_requirements.txt
python3 scripts/prepare_face_test_assets.py
```

Скрипт загружает модель человека и текстуры, около 25 МБ, с закреплённой версией
и атрибуцией. Python-библиотеки устанавливаются только в отдельное окружение;
`--system-site-packages` даёт ему доступ к установленным ROS-пакетам.

Загрузите YuNet, около 233 КБ, если файла ещё нет:

```bash
mkdir -p ~/.cache/face_tracking_arm/models
curl --fail --location --retry 3 \
  https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
  -o ~/.cache/face_tracking_arm/models/face_detection_yunet_2023mar.onnx
printf '%s  %s\n' \
  8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4 \
  "$HOME/.cache/face_tracking_arm/models/face_detection_yunet_2023mar.onnx" | sha256sum -c -
```

Проверка должна завершиться с `OK`. YOLO-модели для обычного запуска не нужны.

## Экспериментальные ONNX-модели

Параметр `detector` также принимает `yolov5n_face` и `yolo_facev2n`.
Для них поместите соответственно `yolov5n-face.onnx` или
`yolo-facev2n-preweight.onnx` в каталог `model_dir`.
Если есть только совместимый FaceV2 checkpoint, см. [инструкцию экспорта](facev2_export.md).
Эти модели не требуются для запуска YuNet.

## Ограничения и лицензии

YuNet используется по умолчанию. YOLO-варианты требуют своих ONNX-весов.
В проверенном checkpoint `yolo-facev2n-preweight.onnx` обнаружения лица
не подтвердились; снижение порога не считается исправлением качества модели.

[Источники моделей и условия использования](../THIRD_PARTY_NOTICES.md).
