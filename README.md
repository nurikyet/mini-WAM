# mini-WAM — эксперименты на сервере H100

**XLand-MiniGrid (`trivial-1m`) → PPO по состояниям → RGB-видео и действия → NanoWM в режимах `policy` / `video` / `joint` → SR в настоящей среде.**

[План экспериментов, постановка и диагностика](docs/SERVER_PLAN_RU.md). Новый код: [`miniwam/server/`](miniwam/server/). Обучение через этот entry point разрешено только на Linux с CUDA. В этой версии серверное обучение ещё не выполнено; SR 100% — критерий приёмки, а не заявленный результат.

XLand-MiniGrid — **среда и набор задач**, NanoWM — **модель мира**. Используются настоящие исходники обоих проектов, закреплённые в [`vendor/`](vendor/README.md), с лицензиями и SHA256. Повторно клонировать upstream-проекты не нужно.

Для `trivial-1m` явно выбран исторический `trivial_1m_v2`, ревизия Hugging Face и SHA256 закреплены в коде. Подмены на `trivial-21k` нет. Для начала используется контролируемое подмножество с видимой целью, ровно в 1/2/3 действиях от успеха. **Это отладочная постановка, не весь миллион задач со скрытой целью.**

## Запуск после загрузки этой папки на GitHub

Подключение `gangway` предварительно настройте по выданной вам серверной инструкции. Ключ остаётся на вашем компьютере.

```bash
# На вашем компьютере — только подключение:
ssh gangway

# ВСЁ НИЖЕ — уже в SSH-сессии на сервере:
cd ~
git clone <URL_ВАШЕГО_РЕПОЗИТОРИЯ> mini-WAM
cd ~/mini-WAM
bash scripts/setup_server.sh
```

Скрипт создаст `~/mini-WAM/.venv-server`, установит зависимости **через уже настроенное зеркало**, проверит CUDA/BF16 и настоящий NanoWM, скачает и проверит `trivial_1m_v2` и frozen SD-VAE. Настройки прокси и индекса пакетов сохраняются. `sudo` не нужен.

Если Python 3.11 доступен как `python3`:

```bash
SERVER_PYTHON=python3 bash scripts/setup_server.sh
```

Сначала короткая проверка всей цепочки на сервере:

```bash
bash scripts/train_server.sh smoke
cat runs/server/smoke/summary.md
```

У smoke очень маленький NanoWM и 20 обновлений WAM: проверяется работоспособность, **SR 100% от него не требуется**. PPO при этом должен пройти свой gate.

Основная одношаговая задача, все три режима:

```bash
mkdir -p logs
nohup bash scripts/train_server.sh one-step \
  > logs/one-step.log 2>&1 < /dev/null &
echo $!
tail -f logs/one-step.log
```

После завершения:

```bash
cat runs/server/one_step/summary.md
cat runs/server/one_step/diagnosis.json
```

Полная последовательность с переходом к двум и трём действиям **только после SR 100% предыдущей стадии**:

```bash
nohup bash scripts/train_server.sh all \
  > logs/all.log 2>&1 < /dev/null &
```

`all` повторно использует завершённый one-step. Также доступны отдельные `two-step` и `three-step`. Повтор той же команды продолжает прерванное обучение из `latest.pt`. После перезапуска сервера сохранённые окружение и результаты остаются в вашем HOME; процесс нужно запустить заново.

**Если изменили код или конфиг — используйте новую папку запуска.** Например:

```bash
OUT_ROOT=runs/server-v2 bash scripts/train_server.sh one-step
```

## Что именно обучается

| Стадия | Настоящая длина эпизода | Горизонты | Моделей при одном seed |
|---|---:|---|---:|
| `one-step` | 1 действие | n=1 | 3 |
| `two-step` | 2 действия | n=1,2 | 6 |
| `three-step` | 3 действия | n=1,2,3 | 9 |

Одношаговый эпизод содержит только один настоящий переход. Фиктивные n=2,3 из padding не считаются отдельными экспериментами. В многошаговых эпизодах после терминального состояния targets маскируются.

| Режим | Обучение | Как выбирается действие |
|---|---|---|
| `policy` | Только CE действий PPO по текущему RGB | Первый выход action head |
| `video` | Только diffusion v-loss будущих кадров, условие — действия | MPC по прогнозам кадров и заданной визуальной цели |
| `joint` | Action CE + diffusion v-loss, общий NanoWM | Первый выход action head |

Основная модель: upstream **NanoWM-S/4**, 12 слоёв, width 384, 6 attention heads; frozen SD-VAE FP32, RGB 128×128 → latent 4×16×16. Добавлен собственный action-query decoder. Это явная адаптация, не исходный код экспериментов летней школы. Основной NanoWM обучается с нуля; готовый Point Maze checkpoint не используется как модель XLand.

PyTorch PPO и NanoWM работают на H100. Маленький векторизованный XLand-симулятор работает в JAX на CPU: это намеренный выбор для данной простой среды, без конкуренции двух фреймворков за GPU-память.

## Данные и результаты

```text
configs/server/                 # smoke, one_step, two_step, three_step
miniwam/server/                 # полный новый конвейер
vendor/                         # upstream NanoWM + XLand, обычные файлы
runs/server/<stage>/
  banks/                        # конечные train/val/test, источник trivial-1m
  expert/                       # PPO, SR, ошибки, gate
  dataset/<split>/              # observations.npy, actions.npy, masks, SR, metadata, MP4
  latents/                      # frozen VAE cache и качество реконструкции
  models/<mode>_n<n>_seed<seed>/  # best/latest, train/val/test, прогнозы, failure traces
  summary.md                    # фактически измеренный SR
  diagnosis.json                # проверки причин неуспеха
```

Checkpoint выбирается по validation. Test оценивается после выбора модели. SR считается через выполнение действий в реальном симуляторе; отдельно сохраняются greedy и sampled T=1 для моделей с action head. `--require-perfect` возвращает код 2, если хотя бы один финальный SR меньше 100%; результаты сохраняются.

Для отчёта без нового обучения:

```bash
.venv-server/bin/python -m miniwam.server report --out runs/server/one_step
```

Проверить воспроизведение переходов и контрольные суммы:

```bash
.venv-server/bin/python -m miniwam.server audit --out runs/server/one_step
```

Квота сервера — 100 ГБ. Сохраняются только `best.pt` и `latest.pt` каждой модели, а не каждый checkpoint за всю историю. Следите за `du -sh ~` и `du -sh runs .venv-server ~/.cache`. Свободное место общей NFS не равно вашей квоте. По умолчанию один model seed; для подтверждения результата задайте несколько seeds в новом конфиге и новую папку запуска.

## Проверки и история

Для нового кода предусмотрены проверки контрактов среды, масок, objectives, action alignment, планировщика и resume. На сервере:

```bash
.venv-server/bin/python -m unittest discover -s tests -p test_server_contracts.py -v
```

Текущий статус проверки и ограничения — в [плане](docs/SERVER_PLAN_RU.md). Старый самостоятельный CNN baseline, его конфиги и ранее полученные CPU-отчёты сохранены для истории. Они **не являются результатами этого серверного конвейера**. Его прежнее описание: [LEGACY_BASELINE.md](docs/LEGACY_BASELINE.md).
