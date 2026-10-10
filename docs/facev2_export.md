# Экспорт FaceV2 в ONNX

Скрипт переводит файл весов FaceV2 (checkpoint) из PyTorch в ONNX. Он не
дообучает модель и сохраняет исходный файл. Поддерживаются модели с блоками
YOLOv5 Conv/C3/SPPF/Detect и шестью выходными значениями.

Используйте только доверенный файл `.pt`. Он может содержать исполняемый
Python-код. Экспорт выполняется на CPU; совместимая CUDA-сборка не нужна.

## Подготовить YOLOv5

Из-за формата checkpoint скрипт требует зафиксированный commit YOLOv5 v6.0.
Скачайте исходный код и переключитесь на нужный commit:

```bash
mkdir -p ~/.cache/face_tracking_arm
git clone https://github.com/ultralytics/yolov5.git \
  ~/.cache/face_tracking_arm/yolov5-export
git -C ~/.cache/face_tracking_arm/yolov5-export checkout 956be8e642b5c10af4a1533e09084ca32ff4f21f
```

Создайте отдельное Python-окружение и установите зависимости закреплённого
исходного кода и ONNX Runtime:

```bash
python3 -m venv ~/.cache/face_tracking_arm/facev2-export-venv
source ~/.cache/face_tracking_arm/facev2-export-venv/bin/activate
python -m pip install -r ~/.cache/face_tracking_arm/yolov5-export/requirements.txt
python -m pip install onnx onnxruntime
```

Для экспорта на CPU установите CPU-сборки PyTorch и torchvision. Выберите
совместимую пару по [официальной странице PyTorch](https://pytorch.org/get-started/locally/).

## Выполнить экспорт

Из корня `face_tracking_arm` укажите каталог YOLOv5 и существующий checkpoint:

```bash
python3 scripts/export_facev2_onnx.py \
  --upstream-dir ~/.cache/face_tracking_arm/yolov5-export \
  --weights ~/.cache/face_tracking_arm/models/yolo-facev2n-preweight.pt
```

Скрипт сравнивает PyTorch и ONNX на тестовых входах. Он записывает `.onnx` рядом
с checkpoint только после успешного сравнения. Файл `.export.json` содержит
SHA-256 исходного и полученного файлов и результат сравнения.

Для дополнительной проверки на кадре добавьте `--image /полный/путь/кадр.png`.
Успешная конвертация подтверждает сходство выходов моделей. Она не подтверждает,
что детектор находит лицо. Проверенный FaceV2 checkpoint в тестовой сцене не дал
детекций. YuNet не требует экспорта.

[Вернуться к 2D-тесту лица](face_detection_test.md).
