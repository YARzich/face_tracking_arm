# Экспорт FaceV2 в ONNX

Скрипт преобразует совместимый checkpoint FaceV2 из PyTorch в ONNX,
сохраняя исходный файл. Для запуска готового детектора PyTorch не нужен.
Используйте только доверенные checkpoint: формат PT может содержать
исполняемые Python-объекты.

Поддерживаются модели с блоками YOLOv5 Conv/C3/SPPF/Detect и шестью выходными
значениями. Для загрузки используется YOLOv5 v6.0:

```bash
git clone --depth 1 --branch v6.0 https://github.com/ultralytics/yolov5.git ~/.cache/face_tracking_arm/yolov5-export
```

Для экспорта нужны совместимые версии PyTorch и torchvision, `onnx`, `pandas`,
`seaborn` и зависимости YOLOv5. Установите их в отдельное виртуальное окружение
и активируйте его. Для экспорта на CPU CUDA не требуется.

Из папки проекта выполните:

```bash
python3 scripts/export_facev2_onnx.py \
  --upstream-dir ~/.cache/face_tracking_arm/yolov5-export \
  --weights ~/.cache/face_tracking_arm/models/yolo-facev2n-preweight.pt
```

Скрипт проверяет commit upstream, сравнивает PyTorch и ONNX на тестовых входах,
записывает ONNX только после успешного сравнения и сохраняет SHA-256 исходного
и полученного файлов в `.export.json`. Опция `--image` добавляет сравнение на
кадре камеры. Экспорт не дообучает модель и не меняет лицензию весов.

[Вернуться к тесту детектора](face_detection_test.md).
