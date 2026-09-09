# Разработка

Окружение: Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic для симуляции.
Команды ниже предполагают, что репозиторий находится в
`~/ros2_ws/src/face_tracking_arm`. Если workspace расположен иначе, замените
`~/ros2_ws` своим путём.

Для запуска реальной руки используйте [руководство Docker](docs/hardware.md).
Инструкции ниже нужны для разработки и симуляции на установленном ROS 2.

## Подготовка и сборка

После установки ROS 2 Jazzy:

```bash
sudo apt install python3-colcon-common-extensions python3-rosdep python3-venv ninja-build
```

Если rosdep ещё не настроен, один раз выполните `sudo rosdep init`.

```bash
source /opt/ros/jazzy/setup.bash
cd ~/ros2_ws
rosdep update --rosdistro jazzy
rosdep install --from-paths src/face_tracking_arm --ignore-src --rosdistro jazzy -y
colcon build --base-paths src/face_tracking_arm --packages-select face_tracking_arm \
  --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G Ninja
source install/setup.bash
```

Собирается только этот пакет. По умолчанию используется `RelWithDebInfo`:
оптимизированный C++ с отладочной информацией. Для симуляции с изображениями
нужна также [подготовка моделей и Python-окружения](docs/face_detection_test.md).

## Проверка изменений

```bash
source /opt/ros/jazzy/setup.bash
cd ~/ros2_ws
source install/setup.bash
colcon test --base-paths src/face_tracking_arm --packages-select face_tracking_arm \
  --event-handlers console_cohesion+
colcon test-result --test-result-base build/face_tracking_arm --verbose
```

Пакетные проверки включают C++/Python-тесты, форматирование, статический анализ,
проверку описания робота, launch-конфигурации и остановки компонентов на
программном контроллере GenericSystem. Они не требуют работающей руки,
Gazebo, GPU или скачанных нейросетевых весов. GitHub Actions выполняет ту же
сборку и проверки.

После изменения управления дополнительно проверьте статичную и движущуюся
цель, исчезновение лица, возврат, поиск и обход в [симуляции](docs/simulation.md).
Запускайте один сценарий за раз и завершайте его через Ctrl+C.
После изменений кода внутри Docker выполните `./run build`;
`./run export` обновит архив для переноса.

## Правила кода

- Сохраняйте публичные launch-аргументы, топики, сообщения и YAML-параметры;
  несовместимое изменение требует описанного перехода.
- Держите ROS-узлы и launch-файлы адаптерами. Вычисления и состояние выделяйте
  в небольшие модули с явными зависимостями.
- Для камер сохраняйте `stamp` и `frame_id`, используйте TF на время измерения.
  В симуляции последовательно передавайте `use_sim_time`.
- В управляющем цикле избегайте ожиданий, сетевых запросов и планирования.
  Соблюдайте ограничения скорости, ускорения, рывка и столкновений.
- Добавляйте изолированный регрессионный тест для исправления поведения;
  обновляйте документацию, если изменяется запуск или интерфейс.
- Соблюдайте стиль ROS 2. Для собственных исходников используйте короткие
  заголовки copyright и `SPDX-License-Identifier: MIT`.
- Не изменяйте импортированную геометрию UFACTORY без обновления сведений об
  источнике и соответствующих тестов.

В pull request укажите проблему, изменение поведения и выполненные проверки.
Для крупных изменений сначала обсудите решение в issue.

## Что не включать в репозиторий

Результаты сборки, Docker-архивы, калибровки, личные конфиги
`config/*.local.yaml`, локальный ROADMAP и история измерений исключены через
`.gitignore`. Нейросетевые веса и модели людей хранятся отдельно в
`~/.cache/face_tracking_arm`.

Не включайте чужие веса без сведений об их лицензии. Источники используемых
ресурсов описаны в [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
