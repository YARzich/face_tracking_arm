# Сторонние компоненты

Исходный код face_tracking_arm распространяется по [MIT](LICENSE).
Эта лицензия не заменяет лицензии сторонних ресурсов, моделей и библиотек.

## В репозитории и Docker-образе

| Компонент | Источник и версия | Лицензия и расположение |
| --- | --- | --- |
| Геометрия и параметры UFACTORY Lite 6 | [xarm_ros2, commit 3dc2b5e](https://github.com/xArm-Developer/xarm_ros2/tree/3dc2b5e8294758d96b54b15fa5920d581b7cbb3d/xarm_description) | BSD-3-Clause; [LICENSE](description/ufactory_lite6/LICENSE), [состав импорта](description/ufactory_lite6/UPSTREAM) |
| Драйвер xarm_ros2 и его C++ SDK | Тот же закреплённый commit и его submodule | Лицензии копируются из исходников в `/usr/share/doc/face_tracking_arm/third_party/` внутри образа |
| YuNet `face_detection_yunet_2023mar.onnx` | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | MIT; [копия лицензии](docker/licenses/YuNet-MIT.txt). Веса загружаются при сборке образа, SHA-256 проверяется в Dockerfile |

Исходные модели Lite 6 сохранены без изменений. При сборке из копий двух
collision-mesh удаляются только одинаковые треугольники; координаты и поверхность
сохраняются. Производные файлы также относятся к BSD-3-Clause ресурсам UFACTORY.

ROS, MoveIt, OpenCV и остальные системные зависимости устанавливаются пакетным
менеджером со своими лицензиями. Список прямых зависимостей — [package.xml](package.xml),
версии Python-библиотек — [perception_requirements.txt](config/perception_requirements.txt).

## Ресурсы отдельных экспериментов

Эти файлы не включены в репозиторий или основной Docker-образ.

| Ресурс | Лицензия и использование |
| --- | --- |
| [Standing person](https://github.com/osrf/gazebo_models/tree/8163eb4b5e7e21985c6591d1c0bfb56468c0093f/person_standing), Marina Kollmitz, MakeHuman | CC BY 3.0. Скрипт подготовки загружает mesh и текстуры без изменений и сохраняет LICENSE и ATTRIBUTION рядом с ними |
| [Depth Anything V2 Metric Hypersim Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small) | Apache-2.0 для варианта Small. Используется только в экспериментальном GPU-тесте; версии кода и весов закреплены в скрипте подготовки |
| [YOLOv5-face](https://github.com/deepcam-cn/yolov5-face) | Код upstream — GPL-3.0; выбранные пользователем ONNX-веса загружаются из внешнего каталога |
| [YOLO-FaceV2](https://github.com/Krasjet-Yu/YOLO-FaceV2) | README upstream указывает MIT для кода; лицензия конкретного checkpoint должна быть проверена отдельно |
| [YOLOv5 v6.0](https://github.com/ultralytics/yolov5/tree/956be8e642b5c10af4a1533e09084ca32ff4f21f) | GPL-3.0; внешний инструмент экспорта FaceV2. Код upstream не включён в пакет |

Основные CPU- и стереосценарии используют YuNet и не требуют YOLO или модели
нейросетевой глубины.
