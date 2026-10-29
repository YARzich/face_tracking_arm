# Runtime-проверка POSTURE-1D

Запускать из каталога пакета. Runner сам поднимает существующий launch,
проверяет preflight, записывает source-stamped joints/TF/context и завершает
только запущенный им launch. Для fixed-F камера выключена; RGB-D работает
через штатный vision launch.

В режиме `sequence` observer сначала фиксирует исходную точку и публикует первую
цель. Отсчёт сценария начинается после свежего execution context, подтверждающего
реальное FACE-слежение за этой первой точкой. Ожидание ограничено прежним preflight
timeout. Длительности этапов и траектория цели после старта не меняются; REST при
запуске больше не попадает в первое окно оценки неподвижной цели.

```bash
tools/run_posture1d_case.sh arm far_hold fixed 60 /tmp/posture1d-far \
  3.052973146 1.907709351 1.75
python3 tools/analyze_posture1d_capture.py /tmp/posture1d-far --window 20 60

tools/run_posture1d_case.sh arm near_hold inband 60 /tmp/posture1d-near
python3 tools/analyze_posture1d_capture.py /tmp/posture1d-near --window 30 60

# Тот же inband 0.4 м с выключенным штатным столом:
POSTURE1D_TABLE_ENABLED=false tools/run_posture1d_case.sh \
  arm near_hold inband 60 /tmp/posture1d-near-no-table
python3 tools/analyze_posture1d_capture.py /tmp/posture1d-near-no-table \
  --window 30 60

tools/run_posture1d_case.sh arm far_lateral_return sequence 175 \
  /tmp/posture1d-sequence tools/posture1d_far_lateral_up_return.json
python3 tools/analyze_posture1d_capture.py /tmp/posture1d-sequence \
  --sequence tools/posture1d_far_lateral_up_return.json

# Проверка медленного возврата (40 с на переход); быстрый 175-секундный
# сценарий выше оставлен отдельным диагностическим прогоном.
tools/run_posture1d_case.sh arm far_lateral_return sequence 205 \
  /tmp/posture1d-sequence-slow tools/posture1d_far_lateral_up_return_slow.json
python3 tools/analyze_posture1d_capture.py /tmp/posture1d-sequence-slow \
  --sequence tools/posture1d_far_lateral_up_return_slow.json

tools/run_posture1d_case.sh rgbd stress_fast observe 126 /tmp/posture1d-rgbd-fast

# Отдельный контроль стартовой позы; без START_POSE launch сохраняет свой default.
tools/run_posture1d_case.sh rgbd stress observe 164 /tmp/posture1d-rgbd-stress-rest rest
```

Анализатор проверяет покрытие joints, синхронизацию context/TF, состояние FACE,
дистанцию и метрики P/E/W. Для статических окон размах каждой точки и путь в
каждом скользящем окне 10 с должны быть не больше 1 см. Камеру оценивают на её
штатной частоте. Если пропуски или рассинхронизация превышают порог, качество
записи неполное: непрерывное поведение по ней не оценивается.

Личные captures, отчёты, rosbag и QP snapshots хранить в ignored-каталоге
`/tmp/posture1d-RUN/` или в ignored-каталоге workspace, не в репозитории. План и
пороговые условия приведены в `ROADMAP.md` и `tools/posture1d_scenarios.json`.

При анализе sequence-записей первые строки acquisition до первой source-stamped FACE-команды
учитываются отдельно как начальный префикс без FACE. После первого FACE непрерывность
команды и свежесть source stamp проверяются на каждом оценённом context, включая конец
acquisition; потери позже и в hold не скрываются. Полное отсутствие FACE до конца acquisition означает
FAIL. Gaze gate для acquisition/recovery (последняя секунда) и fixed hold берётся из
acceptance definition: практический предел 1° (0.01745329 рад) учитывает измеренный шум
RGB-D feed-forward; ошибки выше 1° не проходят. Порог не переписывает результаты старых
captures.

Явный `control_time_wait=true` в execution context означает штатный control tick,
который не выполнял управление и не потреблял текущую F. Observer сохраняет такие
строки и считает их отдельно; анализатор не принимает их за потерю лица, но
физические измерения и joint coverage сохраняет. Между реально оценёнными context
не допускается разрыв больше 50 мс. Пустой context без этого флага и длинные серии
wait остаются `NOT_ASSESSED`; для старых сообщений отсутствующий флаг равен `false`.
Одна source-запись может быть ассоциирована с соседними joint-сэмплами; повторы
не считаются новыми source-событиями при проверке частоты и разрывов.

Для проверки лишних движений при неподвижной цели используйте обычную сцену:
руку с монитором и штатный стол. Сцены Gazebo и MoveIt содержат тот же стол;
дополнительных препятствий нет. `NOT_ASSESSED` означает неполную запись
и не считается PASS. Неподвижность оценивается после первых 20 секунд установления.

```bash
tools/run_collision_brake_case.sh table fixed 60 \
  /tmp/collision-brake-real-table-F .95 0 1.15
python3 tools/analyze_posture1d_capture.py /tmp/collision-brake-real-table-F \
  --window 20 60

tools/run_posture1d_case.sh arm low-table fixed 60 \
  /tmp/collision-brake-low-table-F .8 0 .84
python3 tools/analyze_posture1d_capture.py /tmp/collision-brake-low-table-F \
  --window 20 60

tools/run_collision_brake_case.sh table inband 60 \
  /tmp/collision-brake-table-inband
python3 tools/analyze_posture1d_capture.py /tmp/collision-brake-table-inband \
  --window 20 60
```

Проверять фактическое движение монитора, локтя и запястья: размах и путь
за любые 10 секунд должны укладываться в 1 см; повторных разворотов локтя
быть не должно. Наведение — в пределах 1°. Нужны полная запись данных,
сохранение удобной позы и отсутствие аварий, ошибок решателя и нарушений
физических пределов. Изменение опубликованной цели само по себе не доказывает
движение руки. Для недостижимой дистанции отдельно оценивать, что рука
продолжает наводиться и спокойно держит доступное положение.

Общий контроль движущейся цели и камеры — существующий прогон
`tools/run_posture1d_case.sh rgbd stress_fast observe 126 OUTPUT rest`.
К его фазам движения критерий неподвижности не применяется.
