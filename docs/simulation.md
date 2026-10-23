# Проверить кинематику в Gazebo

Этот запуск проверяет движение руки по искусственной 3D-точке. Он не использует
камеру и детектор лица. Для проверки изображений откройте
[сценарии со зрением](tracking_vision.md).

Сначала соберите пакет по [инструкции](../CONTRIBUTING.md). Затем выполните:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch face_tracking_arm tracking_sim.launch.py
```

По умолчанию запускаются Gazebo, xArm6 и искусственная цель `people`. Цель
показывается в Gazebo. Контроллер проверяет ограничения и столкновения.
Завершите запуск клавишами Ctrl+C.

Основные аргументы:

| Аргумент | Значения | По умолчанию | Назначение |
|---|---|---|---|
| `test_face_scenario` | `disabled`, `stationary`, `circle`, `people`, `walk_around`, `approach` | `people` | Источник искусственной точки |
| `robot_model` | `xarm6`, `lite6` | `xarm6` | Модель руки |
| `start_pose` | `rest`, `zero`, `folded`, `joint1_limit`, `incident` | `rest` | Начальная поза |
| `initial_positions_file` | путь к YAML | пусто | Углы J1–J6; заменяют `start_pose` |
| `servo_backend` | `collision_aware`, `standard` | `collision_aware` | Контроллер движения |
| `headless` | `true`, `false` | `false` | Запустить Gazebo без окон |
| `visualize_target` | `true`, `false` | `true` | Показать искусственную цель |
| `test_world_box` | `true`, `false` | `false` | Подменить round table коробкой только в начальной сцене MoveIt |
| `table_enabled` | `true`, `false` | `true` | Оставить или убрать штатный стол в Gazebo и сцене MoveIt |

`test_world_box:=true` — тестовый режим измерения столкновений. Он заменяет
объект `round_table` в сцене MoveIt коробкой 0.10 × 0.10 × 0.10 м с центром
`(0.50, 0.00, 1.15)` в `world`; модель мира Gazebo и видимый стол не меняются.
`table_enabled:=false` убирает штатный `round_table` из визуальной и физической
сцены Gazebo и из planning scene MoveIt. Пол и высота закрепления основания
робота сохраняются; ограничения суставов и самостолкновения остаются включены.
Это поддержано для штатных миров пакета; содержимое пользовательского `world_file`
автоматически не изменяется.

Пример: нулевая поза и неподвижная цель:

```bash
ros2 launch face_tracking_arm tracking_sim.launch.py \
  start_pose:=zero test_face_scenario:=stationary
```

## Начальная поза

`rest` берёт углы из конфигурации выбранной руки. `zero` задаёт нулевые углы.
`folded` доступна для xArm6. `incident` доступна только для Lite6 и проверяет
блокировку старта при пересечении звеньев. `joint1_limit` ставит J1 на +359.9°.

Задайте свои углы в радианах. YAML должен содержать ровно шесть суставов:

```yaml
initial_positions:
  joint1: 0.0
  joint2: 0.0
  joint3: 0.0
  joint4: 0.0
  joint5: 0.0
  joint6: 0.0
```

Передайте абсолютный путь:

```bash
ros2 launch face_tracking_arm tracking_sim.launch.py \
  initial_positions_file:=/полный/путь/angles.yaml
```

Launch проверяет формат и пределы углов. Он не оборачивает и не обрезает углы.
После загрузки контроллер отдельно проверяет столкновения в сцене.

Физическая модель Gazebo расширяет предел каждого вращательного сустава на
0.01 рад. Это меняет только hard stop Gazebo. Пределы в MoveIt, контроллере
и интерфейсах управления не меняются.

## Тестовые цели

`stationary` держит точку неподвижной. `circle` ведёт её по кругу. `people`
проходит через несколько точек, меняет цель и теряет её в конце. `walk_around`
ведёт точку по непрерывному кругу вокруг основания:

```bash
ros2 launch face_tracking_arm tracking_sim.launch.py \
  test_face_scenario:=walk_around
```

`people` длится 132 с. `walk_around` имеет радиус около 0.85 м и высоту точки
около 1.68 м. Сценарии используют искусственные координаты. Модель человека
и камеры не нужны.

`approach` повторяет плавный подход и отход по оси X перед роботом: точка идёт
от 0.90 м до 0.05 м от вертикальной оси основания, задерживается и отходит.
Высота точки — 1.68 м. Пять сантиметров — координата цели относительно оси
основания, не расстояние до экрана; контроллер может не достичь такой цели.
Обычный запуск без стола и с этим сценарием:

```bash
ros2 launch face_tracking_arm tracking_sim.launch.py table_enabled:=false test_face_scenario:=approach
```

`/face/center` публикует точку цели в метрах и frame `world`.
`/tracking/diagnostics` и `/servo_node/diagnostics` показывают состояние управления. Проверяйте суставы
и TF монитора. Название состояния не доказывает, что рука достигла цели.

## Проверить камеру в Gazebo

Режим калибровки запускает Gazebo с RGB-D камерой и MoveIt. Он не запускает
контроллер слежения. В первом терминале выполните:

```bash
ros2 launch face_tracking_arm tracking_sim.launch.py \
  calibration_mode:=true headless:=true camera_mode:=rgbd camera_rate:=30
```

Во втором терминале запустите калибровку:

```bash
ros2 launch face_tracking_arm mount_setup.launch.py \
  external_sim:=true mode:=rgbd use_sim_time:=true \
  marker_side_mm:=45 output_file:=/tmp/face_tracking_sim_calibration.yaml
```

Узел использует кадры RGB, CameraInfo и суставные положения с метками времени.
Он проверяет результат на отдельных позах. При успехе проверьте выходной YAML.
Подробные шаги и ограничения описаны в
[руководстве по калибровке крепления камеры](mount_setup.md). Сравните
`camera_mount` с TF из описания робота только после калибровки.

Размер метки в мире равен 45 мм. Передайте это значение через
`marker_side_mm:=45`. Другое значение меняет масштаб калибровки и может дать
неверный результат; тест не обязан обнаружить каждое такое расхождение.
Параметр `camera_rate` задаёт частоту камеры. Повторите запуск при 15 и 30 Гц.

## Проверить компактный разворот

Этот путь предназначен для восстановления поворота пары J1/J4 после исчерпания
полезного хода пары. Отставание наведения и ограничение другого сустава сами по
себе не запускают манёвр. Проверяйте фактические суставы и ориентацию монитора,
а не только название фазы.

Для проверки сложенной xArm6 и круговой цели откройте терминал в корневой папке
проекта `face_tracking_arm`. Задайте конфигурационный файл из этой папки:

```bash
ros2 launch face_tracking_arm tracking_sim.launch.py \
  robot_model:=xarm6 start_pose:=folded test_face_scenario:=walk_around \
  collision_aware_servo_config_file:="$PWD/config/moveit/collision_aware_servo.yaml"
```

Параметр `compact_maneuver.speed_scale` в этом YAML по умолчанию равен `1.0`.
Он меняет скорость вдоль пути, но не усиливает коррекцию отклонения. Скорость
снижается перед поворотами и у конца пути. На последнем участке к цели меняются
усиление и предел скорости. Ограничения
скорости, ускорения, рывка, суставов и проверки столкновений остаются в силе.
Значение должно быть положительным. Используйте `1.0` для штатной скорости.
Изменяйте его только для отдельного сравнения. Launch принимает
`collision_aware_servo_config_file`, поэтому правка YAML в рабочей копии
применится без пересборки пакета.

Режим `recovery.mode` выбирает поведение после исчерпания полезного поворота J1/J4:
`compact` разрешает компактный маршрут, `disabled` отключает его. Чтобы проверить
режим без пересборки, скопируйте штатный YAML, измените в копии `recovery.mode`,
затем укажите эту копию в уже существующем аргументе запуска:

```bash
cd ~/ros2_ws/src/face_tracking_arm
cp config/moveit/collision_aware_servo.yaml config/moveit/collision_aware_servo.local.yaml
sed -i 's/^  mode: compact/  mode: disabled/' config/moveit/collision_aware_servo.local.yaml
ros2 launch face_tracking_arm tracking_sim.launch.py \
  robot_model:=xarm6 start_pose:=folded test_face_scenario:=walk_around \
  collision_aware_servo_config_file:="$PWD/config/moveit/collision_aware_servo.local.yaml"
```

Без `recovery.mode` используется `compact`, в том числе в прежних пользовательских
YAML. Неизвестное значение отклоняется до запуска контроллера.
