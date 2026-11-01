# mini-WAM

Рабочий конвейер задания: **простая среда → PPO на состояниях → RGB-траектории → `policy` / `video` / `joint` на n шагов → closed-loop SR → среда на два действия**.

Начальная задача — действительно **одно действие**: синий агент стоит рядом с красным квадратом; нужно сделать правильный ход. Двухшаговая версия использует тот же код, но цель находится на расстоянии двух ходов. Это самостоятельный небольшой WAM baseline, а не воспроизведение diffusion-модели NanoWM или ShapeCraft из отчёта.

[Выполненные проверки, SR 100% и разбор ранних ошибок](reports/VERIFICATION.md).

## Что реализовано

| Этап | Реализация |
|---|---|
| Среда | Поле 5×5, четыре абсолютных действия: вверх, вправо, вниз, влево. RGB 30×30. Успех — достижение цели; успешная клетка становится зелёной |
| Эксперт | PPO с clipped objective, critic и GAE; вход — относительные координаты цели и остаток бюджета. Без oracle и маски правильных действий |
| Датасет | RGB, действия, награды, SR, длины, task IDs, valid/expert masks, manifest, SHA256, GIF |
| `policy` | По текущим пикселям предсказывает n действий; только action CE |
| `video` | По пикселям и последовательности действий предсказывает n кадров; только image CE. Действует через планирование в обученной модели |
| `joint` | Общие encoder и recurrent transition, action CE + image CE; действует через action head |
| Оценка | Первое предсказанное действие выполняется в настоящей среде, затем новое наблюдение и перепланирование |
| Диагностика | Failure traces, ошибки по направлениям, границы, качество кадров против копирования контекста, чувствительность к действию |
| Resume | Атомарные checkpoints, optimizer и RNG; проверка совпадения кода, параметров и данных |

Pixel-модель — небольшая CNN с рекуррентной свёрточной динамикой. Пиксели клеток усредняются без потерь из 30×30 до 5×5 RGB; фиксированная палитра задаёт image targets. Состояния PPO, награды и success labels в pixel-модель не передаются. Загружаемые веса, VAE, JAX, W&B, дисплей и внешние датасеты новому конвейеру не нужны.

```text
miniwam/                      # Среда, PPO, данные, модели, обучение, оценка, CLI
configs/smoke.json            # Быстрый one-step, n=3, все три режима
configs/one_step.json         # 3 режима × n={1,2,3}
configs/two_step_smoke.json   # Короткая проверка two-step
configs/two_step.json         # Полная серия на двух действиях
scripts/train_server.sh       # Последовательно one-step и two-step
scripts/prepare_git_export.py # Чистая копия для Git
tests/                       # Проверки среды, данных, objectives и resume
reports/                     # Компактные результаты выполненных проверок
runs/                        # Новые данные/веса/отчёты, исключены из Git
XLand-MiniGrid/               # Предыдущие исследования и исходники
Nano-World-Models/            # Предыдущие исследования и исходники
```

## Подготовить папку для Git

В рабочей папке есть два вложенных `upstream/.git` и несколько ГБ окружений, кешей и весов. При обычном `git add .` upstream-код может попасть в Git как ссылка на вложенный репозиторий. Для переноса **всех исходников обоих проектов обычными файлами** сделайте чистую копию:

```bash
# Из исходной mini-WAM; нужен только Python, пакеты устанавливать не надо.
python3 scripts/prepare_git_export.py --out ../mini-WAM-git --dry-run
python3 scripts/prepare_git_export.py --out ../mini-WAM-git
cd ../mini-WAM-git
git init
git add .
git status --short
# Далее самостоятельно commit, remote и push в свой репозиторий.
```

Исходная папка и её Git-истории сохраняются. Копируются код, конфигурации, документация, notebooks, небольшие данные и отчёты. Исключаются окружения, кеши, вложенные `.git`, новые `runs/`, checkpoint-веса, архивы, `.env` и файлы ≥95 МиБ. Список исключений: `git-export-manifest.json`. Новый конвейер создаёт свои данные/веса заново; старые NanoWM-веса переносятся отдельно при необходимости. Папка назначения должна быть новой и вне исходной mini-WAM.

## Установка на сервере

Все команды — **из корня клонированного репозитория**, Linux, Python 3.11. Для маленького baseline достаточно CPU; CUDA доступна отдельно. GPU-часть должна пройти `doctor` и smoke на вашем сервере: локальные проверки сделаны на CPU.

```bash
git clone <URL_ВАШЕГО_РЕПОЗИТОРИЯ> mini-WAM
cd mini-WAM
python3.11 -m venv .venv-wam
source .venv-wam/bin/activate
python -m pip install --upgrade pip setuptools

# Выберите ОДИН вариант torch.
# NVIDIA CUDA 12.6:
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
# Либо CPU:
# python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu

python -m pip install -e .
python -m miniwam --device cpu doctor
python -m unittest discover -s tests -v

# Для GPU:
nvidia-smi
python -m miniwam --device cuda doctor
```

Torch 2.7.1/cu126 взят из [официальной таблицы PyTorch](https://pytorch.org/get-started/previous-versions/#v271); там же есть cu118/cu128 для подходящего драйвера. Если CUDA недоступна, явный `--device cuda` завершится ошибкой. На Mac используйте обычный `pip install -e .` и `--device cpu`. Существующее `Nano-World-Models/.venv-macos/bin/python` тоже подходит для нового кода из корня.

## План запуска на сервере

1. **Короткий сквозной прогон:** PPO, датасет, три режима, итоговый SR.

   ```bash
   python -m miniwam --device cuda pipeline \
     --config configs/smoke.json --out runs/smoke --require-perfect
   cat runs/smoke/summary.md
   ```

2. **Полная одношаговая серия:** девять моделей, 3 режима × 3 горизонта, один датасет и одинаковые splits.

   ```bash
   python -m miniwam --device cuda pipeline \
     --config configs/one_step.json --out runs/one_step --require-perfect
   cat runs/one_step/summary.md
   ```

3. **Если SR ниже 100%, разобрать ошибки.** Результат сохраняется; `--require-perfect` возвращает код 2 при любом test SR <100%. Эксперт с неполным train/val SR останавливает конвейер до обучения WAM. Файлы диагностики описаны ниже.

4. **Перейти к двум действиям:** отдельно обучаются PPO и WAM, собирается новый датасет.

   ```bash
   python -m miniwam --device cuda pipeline \
     --config configs/two_step.json --out runs/two_step --require-perfect
   cat runs/two_step/summary.md
   ```

Запуск обеих основных серий одной командой:

```bash
bash scripts/train_server.sh
# Для CPU: DEVICE=cpu bash scripts/train_server.sh
```

Скрипт начнёт two-step после SR 100% во всех одношаговых режимах/горизонтах. Это критерий конечной серии, а не автоматический подбор параметров по test.

Для работы после закрытия SSH:

```bash
mkdir -p runs
nohup bash scripts/train_server.sh > runs/server.log 2>&1 &
echo $!
tail -f runs/server.log
```

При повторном запуске скрипта завершённые стадии переиспользуются, незавершённые восстанавливаются из `latest.pt`. PPO сохраняется каждый update; WAM — каждые `eval_every` updates. При ручном запуске добавьте `--resume`. Изменение кода/параметров требует **новой** папки. Продолжение после прерывания сохраняет исходный бюджет обучения. Для каждого процесса нужна отдельная папка результатов.

Можно выбрать диск, устройство и интерпретатор:

```bash
OUT_ROOT=/mnt/data/miniwam PYTHON=.venv-wam/bin/python DEVICE=cuda bash scripts/train_server.sh
```

## Протокол и формат данных

Для 5×5 одношаговый банк содержит 80 задач: 49 train / 15 val / 16 test. Двухшаговый — 124: 77 / 23 / 24. Разделение по **позиции цели** исключает одинаковые входные кадры между splits, включая промежуточные состояния. Каждый переход проверяется replay-аудитом.

Checkpoint выбирается по validation SR и offline validation loss. Эксперт должен получить train/val SR 100% до сбора данных. Test используется в итоговом отчёте; его кадры не участвуют в градиентах. Для изменения параметров ориентируйтесь на validation. После изучения test-ошибок новое научное сравнение требует заранее зафиксированного нового `split_seed`.

`dataset/{train,val,test}.npz` — числовые массивы, `allow_pickle=False`:

| Массив | Форма | Смысл |
|---|---|---|
| `observations` | `[E, depth+1, H, W, 3]`, uint8 | Начальный кадр и кадры после действий |
| `actions` | `[E, depth]` | `obs[t] --action[t]--> obs[t+1]`; padding = −1 |
| `valid` | `[E, depth]` | Настоящий переход; исключает terminal padding из loss |
| `expert_mask` | `[E, depth]` | Действие выбрал PPO; только эти метки входят в action CE |
| `success`, `lengths` | `[E]` | Реальный результат эпизода и длина |
| `rewards`, `task_ids` | `[E, depth]`, `[E]` | Метаданные для аудита, не входы WAM |

Первые четыре повтора каждой задачи перебирают четыре начальных действия. Пятый — чистая PPO-траектория. Затем используется PPO с долей случайных действий `exploration`. Неудачные эпизоды сохраняются: image-only модели нужны негативные примеры. Поэтому `behavior_sr` смеси ниже `expert_closed_loop_sr`; это разные политики. Все режимы получают один датасет, action loss случайных действий маскируется.

**n — горизонт предсказания; depth — бюджет среды.** На `depth=1` есть один настоящий future target при любом n, на `depth=2` — два. Padding не создаёт фиктивных обучающих шагов. `horizons` в JSON задаёт список n; больше настоящих future steps требует расширения задачи.

Image CE применяется к фиксированным цветам клеток с повышенным весом объектов. `video` перебирает до 4² планов, оценивает предсказанную дистанцию до видимой красной цели и появление зелёного успеха. Это известный визуальный критерий задачи. PPO, будущие состояния, SR labels и `env.step` внутри планировщика не используются. `joint` и `policy` действуют через action head.

## Результаты и разбор SR

```text
runs/one_step/
├── resolved_config.json, run.json, task_banks.json
├── expert/                   # best.pt, latest.pt, metrics.json, gate.json
├── dataset/                  # NPZ, manifest.json, audit.json, GIF
├── models/{mode}_n{n}_seed{seed}/
│   ├── best.pt, latest.pt, metrics.json, complete.json
│   └── test/
│       ├── evaluation.json   # SR, failures, направления, offline metrics
│       ├── forecast.npz      # context/truth/prediction/actions/valid
│       └── forecast_*_truth_left.gif
└── summary.json, summary.md
```

Если SR ниже 100%, сначала проверьте `expert/gate.json` и `dataset/audit.json`. В `models/*/metrics.json` видно, достигнут ли validation SR 100% или закончился бюджет. В `test/evaluation.json` для каждой ошибки записаны задача, действия, расстояния до/после и столкновения с границей.

Для кадров сравнивайте `foreground_cell_accuracy`, `forecast_mse` с `repeat_context_mse` и `action_sensitivity_mae`. Общая pixel accuracy может скрывать ошибку в клетке агента. `action_accuracy` считает PPO-метки; `open_loop_action_accuracy` — совпадение декодированной последовательности с записанными метками, включая возможное расхождение ветвей после первого действия. Основной критерий — closed-loop SR.

```bash
python -m miniwam audit --dataset runs/one_step/dataset
python -m miniwam --device cuda evaluate \
  --checkpoint runs/one_step/models/joint_n3_seed3407/best.pt \
  --dataset runs/one_step/dataset --split val --out runs/recheck_val
python -m miniwam --device cpu evaluate-expert \
  --checkpoint runs/one_step/expert/best.pt --split all --out runs/expert_all.json
```

При нехватке обучения создайте копию JSON с большим `updates` и новую папку запуска. Для нескольких training seeds: `"seeds": [3407, 3408, 3409]` — 27 моделей вместо 9. Память CUDA и время стадии записываются в `complete.json`; время на вашем сервере определяется smoke-прогоном. SR 100% на конечном тестовом банке не гарантирует успех на больших полях, с препятствиями или при другом seed.

## Отдельные стадии и проверки

```bash
python -m miniwam --device cpu expert --depth 1 --out runs/manual/expert
python -m miniwam --device cpu collect \
  --expert runs/manual/expert/best.pt --out runs/manual/dataset
python -m miniwam --device cpu train \
  --dataset runs/manual/dataset --mode joint --horizon 3 --out runs/manual/joint
python -m miniwam --device cpu evaluate \
  --checkpoint runs/manual/joint/best.pt --dataset runs/manual/dataset \
  --split test --out runs/manual/evaluation
python -m unittest discover -s tests -v
```

`--device` и `--threads` ставятся **до** имени команды. `auto` выбирает CUDA при наличии, иначе CPU; MPS включается явно. Все параметры основной серии задаются JSON. Справка: `python -m miniwam --help`.

Тесты проверяют полный банк задач, достижимость цели, эквивалентность state/RGB сред, terminal GAE, отсутствие утечки, выравнивание obs/action/next obs, маскирование padding/random labels, планирование на идеальной визуальной динамике, точное восстановление PPO/WAM после имитации прерывания и экспорт без вложенных Git-репозиториев.

Прежние материалы сохранены: [XLand-MiniGrid](XLand-MiniGrid/README.md), [Nano World Models](Nano-World-Models/README.md), [разбор Video Action Models](Nano-World-Models/docs/SUMMER_SCHOOL_GPU_RU.md). Их окружения и отчёты самостоятельны; новый конвейер не зависит от upstream-кода.

<details>
<summary>Исторические инструкции предыдущих экспериментов</summary>

Два отдельных направления работы:

| Проект | Что находится внутри |
|---|---|
| [XLand-MiniGrid](XLand-MiniGrid/README.md) | Рабочие RL²/PPO-эксперименты, ноутбук, датасеты, результаты и исходный репозиторий |
| [Nano World Models](Nano-World-Models/README.md) | Разбор статьи и кода, выполненные CPU/MPS-опыты, прогнозы с опубликованными весами, Jupyter и Colab |

```text
mini-WAM/
├── XLand-MiniGrid/
│   ├── upstream/       # GitHub-репозиторий xminigrid с его историей Git
│   ├── experiments/    # Наши скрипты и зависимости
│   ├── notebooks/      # Jupyter / Colab
│   ├── data/           # Наборы задач
│   ├── results/        # Метрики, логи, checkpoints и траектории
│   ├── docs/           # Разбор исследований и инструкции
│   ├── papers/         # Статья XLand-MiniGrid
│   └── .venv/          # Готовое окружение JAX / Python 3.11
└── Nano-World-Models/
    ├── upstream/       # Исходный GitHub-репозиторий NanoWM
    ├── experiments/    # CPU/MPS-скрипты, зависимости и патч
    ├── notebooks/      # Выполненный Jupyter и отдельный Colab
    ├── configs/        # Конфигурации моделей и обучения
    ├── data/           # Данные моделей мира
    ├── results/        # Метрики, checkpoints, изображения и анимации
    ├── docs/           # Разбор NanoWM, летней школы и оценка GPU
    ├── papers/         # Nano World Models и отчёт Video Action Models
    └── .venv-macos/    # Отдельное окружение PyTorch CPU/MPS
```

## Открыть готовый ноутбук

Из корня `mini-WAM`:

```bash
cd XLand-MiniGrid
JAX_PLATFORMS=cpu .venv/bin/python -m jupyterlab notebooks/trivial_1m_experiments.ipynb
```

Ядро: **XLand-MiniGrid (JAX CPU)**. Новый путь к ноутбуку:
[XLand-MiniGrid/notebooks/trivial_1m_experiments.ipynb](XLand-MiniGrid/notebooks/trivial_1m_experiments.ipynb).

Каждый проект использует собственные данные, результаты и окружение. Общими в корне остаются только служебные `.venv` (инструменты чтения PDF и `uv`), `.python` (управляемый Python) и `.cache/uv` (кеш установки пакетов). Это не окружение обучения NanoWM.

## Открыть NanoWM

Из корня `mini-WAM`:

```bash
cd Nano-World-Models
.venv-macos/bin/python -m jupyterlab notebooks/nanowm_local_experiments.ipynb
```

Ядро: **NanoWM (PyTorch CPU/MPS)**. [Подробный разбор и команды](Nano-World-Models/docs/NANOWM_REVIEW.md),
[отдельный ноутбук для Colab](Nano-World-Models/notebooks/nanowm_point_maze_colab.ipynb).

[Наглядный отчёт на русском](Nano-World-Models/notebooks/nanowm_local_experiments.html)
можно открыть без Jupyter: входные данные, движение агента, графики качества и выводы.

</details>
