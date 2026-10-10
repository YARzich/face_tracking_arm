# Разработка

Инструкции рассчитаны на Ubuntu 24.04 и ROS 2 Jazzy. Ниже `~/ros2_ws` — корень
workspace, а `src/face_tracking_arm` — каталог пакета. Если ваши пути отличаются,
замените их в командах.

Для запуска реальной руки используйте [руководство по оборудованию](docs/hardware.md).
Этот файл описывает установку зависимостей, сборку и пакетные проверки.

## Установить зависимости

Установите инструменты сборки:

```bash
sudo apt update
sudo apt install python3-colcon-common-extensions python3-rosdep ninja-build
```

Если `rosdep` ещё не настроен, выполните `sudo rosdep init` один раз. Затем
обновите список зависимостей и установите зависимости пакета:

```bash
source /opt/ros/jazzy/setup.bash
cd ~/ros2_ws
rosdep update --rosdistro jazzy
rosdep install --from-paths src/face_tracking_arm --ignore-src --rosdistro jazzy -y
```

Для симуляции с изображениями отдельно подготовьте модели и Python-среду по
[руководству по тесту детектора](docs/face_detection_test.md).

## Собрать пакет

Из корня `~/ros2_ws` выполните:

```bash
source /opt/ros/jazzy/setup.bash
cd ~/ros2_ws
colcon build --base-paths src/face_tracking_arm --packages-select face_tracking_arm \
  --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G Ninja
source install/setup.bash
```

Команда собирает пакет `face_tracking_arm`, а не весь workspace. CMake по
умолчанию выбирает `RelWithDebInfo`: оптимизированную сборку с отладочной
информацией. Флаг `CMAKE_EXPORT_COMPILE_COMMANDS=ON` создаёт
`compile_commands.json` для CLion.

## Проверить пакет

После сборки из корня workspace выполните:

```bash
source /opt/ros/jazzy/setup.bash
cd ~/ros2_ws
source install/setup.bash
colcon test --base-paths src/face_tracking_arm --packages-select face_tracking_arm \
  --event-handlers console_cohesion+
colcon test-result --test-result-base build/face_tracking_arm --verbose
```

Первая команда запускает тесты и проверки пакета. Вторая выводит их результат.
GitHub Actions выполняет те же сборку и пакетные проверки на ROS 2 Jazzy.
Эти проверки не являются испытанием реальной руки или полным тестом слежения в
Gazebo. Для сценариев с искусственной целью см. [руководство по симуляции](docs/simulation.md).
Для лиц в кадре используйте [руководство по слежению в Gazebo](docs/tracking_vision.md).

Команды `./run build` и `./run export` запускайте из каталога пакета:

```bash
cd ~/ros2_ws/src/face_tracking_arm
./run build
./run export
```

Первая команда собирает Docker-образ. Вторая создаёт архив образа для переноса.
Подробности приведены в [руководстве по оборудованию](docs/hardware.md).

## Правила изменений

- Сохраняйте имена публичных launch-аргументов, топиков, сообщений и параметров YAML.
  Для несовместимого изменения опишите переход.
- Оставляйте ROS-узлы и launch-файлы тонкими. Передавайте вычисления и состояние
  небольшим модулям с явными зависимостями.
- Сохраняйте `stamp` и `frame_id` в данных камеры. Преобразуйте точки через TF
  на время измерения. В симуляции используйте `use_sim_time` согласованно.
- Не добавляйте ожидание, сетевой запрос или планирование в управляющий цикл.
  Сохраняйте ограничения скорости, ускорения, рывка, суставов и столкновений.
- Добавляйте регрессионный тест для исправленного поведения. Обновляйте руководство,
  если меняется запуск или ROS-интерфейс.
- Следуйте стилю ROS 2. Для исходников проекта используйте заголовок авторских прав
  и `SPDX-License-Identifier: MIT`.
- Не меняйте импортированную геометрию UFACTORY без обновления источника и тестов.

В pull request укажите проблему, новое поведение и выполненные проверки.

## Не включать в репозиторий

Не добавляйте результаты сборки, архивы Docker, калибровки, личные конфиги
`config/*.local.yaml`, локальные планы и отчёты испытаний. Git игнорирует эти файлы.
Нейросетевые веса и модели людей хранятся отдельно в
`~/.cache/face_tracking_arm`.

Не добавляйте веса без сведений о лицензии. Источники ресурсов перечислены в
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
