# Экспорт FaceV2 в ONNX

Готовый ONNX можно перенести вместе с остальными весами: PyTorch при работе
детектора не нужен. Исходный PT сохранён. Скрипт экспорта предназначен только
для доверенного пользовательского checkpoint, поскольку старый формат содержит
сериализованные Python-объекты.

В поддерживаемом checkpoint используются обычные YOLOv5 Conv/C3/SPPF/Detect с
шестью выходными значениями. Код YOLO-FaceV2 v2.1 не содержит SPPF, а его
нынешний master использует другой Detect с landmarks. Для чтения этого файла
использован совместимый upstream YOLOv5 v6.0, с его штатной миграцией
`anchor_grid` и небольшой поправкой `Upsample` для нового PyTorch.

```bash
git clone --depth 1 --branch v6.0 https://github.com/ultralytics/yolov5.git ~/.cache/face_tracking_arm/yolov5-export
```

Export-only зависимости: совместимая пара PyTorch/torchvision, `onnx`, `pandas`,
`seaborn` и зависимости upstream. Устанавливайте их в отдельное окружение для экспорта. Для экспорта на CPU
CUDA-сборка не нужна. Готовый ONNX затем используется основным окружением детектора.

```bash
~/.cache/face_tracking_arm/perception_venv/bin/python scripts/export_facev2_onnx.py \
  --upstream-dir ~/.cache/face_tracking_arm/yolov5-export \
  --weights ~/.cache/face_tracking_arm/models/yolo-facev2n-preweight.pt
```

Скрипт проверяет commit upstream, сравнивает PyTorch и ONNX на тестовых входах,
записывает ONNX только после успешного сравнения и сохраняет SHA-256 исходного
и полученного файлов в `.export.json`. Опция `--image` добавляет сравнение на
кадре камеры. Экспорт не дообучает модель и не меняет лицензию весов.


[Вернуться к тесту детектора](face_detection_test.md).
