# Синтетическая проверка паузы поиска

`tools/test_search_camera_hold.py` подключается к уже запущенной симуляции и
публикует только `/face/perception_status` и `/face/center`. Камера и RGB-D
потоки не нужны. После пяти секунд подтверждённого слежения за лицом сценарий
переходит в `NO_FACE`, дважды кратко сообщает о потере глубины в
`SEARCH_LOCAL`, затем делает то же в `SEARCH_SWEEP` и возвращает свежее лицо.
Команды `/servo_node/tracking_target_cmds`, обе диагностические темы и `/clock`
записываются для проверки фактических переходов.

В первом терминале из корня workspace запустите симуляцию:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch face_tracking_arm tracking_sim.launch.py \
  headless:=true start_pose:=rest camera_mode:=disabled \
  test_face_scenario:=disabled idle_behavior:=search_local_then_sweep \
  require_perception_status:=true
```

Во втором терминале из корня workspace выполните helper. Он завершится сам не
позднее чем через 120 секунд. Каталог результата должен быть новым: helper
откажется перезаписывать предыдущий прогон.

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
cd ~/ros2_ws/src/face_tracking_arm
python3 tools/test_search_camera_hold.py \
  --output ~/ros2_ws/log/rgbd_continuity_implementation_20261026/runtime/synthetic_hold
```

Успех — `result: PASS` в `report.json`: каждый перерыв подтверждён servo
диагностикой `motion_reference_state=HOLD`, поиск возобновился в той же фазе,
число запрошенных планов не изменилось во время перерыва, активная длительность
локальной фазы составила 5–8 секунд. После свежего `OK` helper ждёт новую
команду `FACE` и более позднюю servo диагностику с
`motion_reference_state=FACE` и `mode=TRACKING`.
`events.jsonl` содержит полную 20-Гц шкалу состояний с publication/source stamp,
переходы фаз и режимы реальных команд. Если фаза не наблюдалась к таймауту,
соответствующий критерий помечается `NOT_ASSESSED`; такой прогон не проходит.
