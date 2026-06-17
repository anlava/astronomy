# Деплой на Yandex Cloud VM

Эта директория содержит скрипты и инструкции для ручного деплоя проекта на Yandex Cloud VM и запуска полного пайплайна на сплите `target` (~94M строк).

## 1. Создание VM в Yandex Cloud

Рекомендуемая конфигурация:
- **Платформа**: Intel Ice Lake / AMD EPYC (любая)
- **vCPU**: 16
- **RAM**: 128 GB
- **Диск**: 500 GB SSD (NVMe)
- **ОС**: Ubuntu 22.04 LTS
- **Публичный IP**: обязательно
- **Прерываемость**: для первого прогона (`compute_all_features.py`) лучше **непрерываемая** VM, так как итоговый parquet пишется только в конце.

После создания VM подключитесь по SSH:

```bash
ssh <user>@<vm-ip>
```

## 2. Подготовка диска (если диск > 200 GB и не примонтирован в /home)

Если вы добавили отдельный диск под данные, примонтируйте его, например, в `/data`:

```bash
lsblk
sudo mkfs.ext4 /dev/vdb
sudo mkdir -p /data
sudo mount /dev/vdb /data
sudo chown $USER:$USER /data
```

Добавьте в `/etc/fstab` для автоматического монтирования.

## 3. Быстрый старт

На свежей Ubuntu 22.04 выполните:

```bash
# Скачайте скрипт настройки
wget https://raw.githubusercontent.com/anlava/astronomy/main/deploy/setup_vm.sh
chmod +x setup_vm.sh
./setup_vm.sh
```

Скрипт:
- установит Python 3.11 и зависимости системы;
- клонирует форк `anlava/astronomy` в `~/astronomy`;
- создаст виртуальное окружение `~/astronomy/.venv`;
- установит Python-зависимости из `requirements.txt`.

### 3.1. Настройка git credentials (если планируете пушить с VM)

Для публичного форка клонирование работает без авторизации. Если нужен `git push`, настройте один из способов:

**GitHub CLI (рекомендуется):**
```bash
sudo apt install gh -y
gh auth login
gh auth setup-git
```

**HTTPS + PAT:**
```bash
git config --global credential.helper store
git remote set-url fork https://<TOKEN>@github.com/anlava/astronomy.git
```

**SSH:**
```bash
ssh-keygen -t ed25519 -C "vm-deploy"
cat ~/.ssh/id_ed25519.pub
# добавить ключ в GitHub → Settings → SSH and GPG keys
git remote set-url fork git@github.com:anlava/astronomy.git
```

## 4. Доступ к HuggingFace из РФ

CDN HuggingFace (`cas-bridge.xethub.hf.co`, CloudFront) может отдавать 403 для российских IP. Варианты решения:

### 4.1. VPN / прокси на стороне VM

Самый надёжный способ — поднять WireGuard/OpenVPN или настроить HTTP-прокси на VM, находящейся вне РФ. После этого:

```bash
export HTTPS_PROXY=http://<proxy-host>:<port>
```

### 4.2. Зеркало HuggingFace (hf-mirror.com)

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

> **Примечание**: работоспособность зеркала из РФ не гарантируется, проверяйте `curl $HF_ENDPOINT`.

### 4.3. Ручная загрузка датасета

Если автоматическая загрузка невозможна, скачайте `snad-space/ztf-m-dwarf-flares-2025` локально и загрузите на VM через `rsync` / `scp` в `~/astronomy/data`.

## 5. Скачивание датасета с HuggingFace

Перед запуском `compute_all_features.py` полезно отдельно скачать датасет, чтобы не ловить 403 посреди вычислений.

### 5.1. Проверка доступа

```bash
cd ~/astronomy
./deploy/test_hf_access.sh
```

Скрипт проверяет доступность `huggingface.co` и `hf-mirror.com`, а затем пытается загрузить первые 1000 строк сплита `train`.

### 5.2. Скачивание через datasets (рекомендуется)

```bash
cd ~/astronomy
source .venv/bin/activate

# Скачать target (~30 GB сырых данных, 314 parquet-файлов)
python deploy/download_dataset.py --split target --num-proc 4

# Или train (~0.8 GB) для быстрой проверки
python deploy/download_dataset.py --split train --num-proc 4
```

Датасет кэшируется в `~/.cache/huggingface`. Убедитесь, что на диске достаточно места (target требует ~50–70 GB под кэш + parquet).

### 5.3. Если HuggingFace недоступен из РФ

Установите одну из переменных окружения перед скачиванием:

```bash
# Вариант A: зеркало (проверьте работоспособность)
export HF_ENDPOINT=https://hf-mirror.com

# Вариант B: HTTP-прокси
export HTTPS_PROXY=http://<proxy-host>:<port>
export HF_HUB_ENABLE_HF_TRANSFER=1

# Вариант C: VPN на уровне системы/сетевого интерфейса VM
```

Проверка зеркала:

```bash
curl -I $HF_ENDPOINT/datasets/snad-space/ztf-m-dwarf-flares-2025
```

### 5.4. Ручная загрузка

Если автоматическое скачивание невозможно:

1. Скачайте датасет локально (с VPN):
   ```bash
   huggingface-cli download snad-space/ztf-m-dwarf-flares-2025 --repo-type dataset --local-dir ./ztf-m-dwarf-flares-2025
   ```
2. Загрузите на VM:
   ```bash
   rsync -avz --progress ./ztf-m-dwarf-flares-2025 <user>@<vm-ip>:/data/hf_cache/
   ```
3. При запуске `compute_all_features.py` HF-кэш должен указывать на эту директорию.

## 6. Извлечение признаков для `target`

```bash
cd ~/astronomy
screen -S features
./deploy/run_compute_features.sh
```

Процесс занимает **3–8 часов** и создаёт `~/astronomy/data/all_features.parquet` (~80–90 GB для 201 колонки).

Отключитесь от screen: `Ctrl+A, D`. Вернуться: `screen -r features`.

## 7. Запуск active learning пайплайна

```bash
cd ~/astronomy
screen -S al
./deploy/run_active_learning.sh
```

Пайплайн:
- загружает `all_features.parquet` с 201 колонкой (`COLS_TO_USE`);
- формирует `known_flares` (2214 объекта) и forced negatives из `expert_labels.txt`;
- запускает CatBoost в CPU-режиме;
- пишет результаты в `~/astronomy/active_learning_output_target/`.

## 8. Мониторинг

```bash
# Загрузка CPU/RAM/disk
htop
nvme smart-log /dev/vda   # или df -h

# Логи пайплайна
tail -f ~/astronomy/active_learning_output_target/pipeline.log
```

## 9. После завершения

Результаты:
- `~/astronomy/data/all_features.parquet` — признаки для target
- `~/astronomy/active_learning_output_target/` — чекпоинты, отчёты, probability reports
- `~/astronomy/sample_plots/` — графики (если включены)

Скачать результаты локально:

```bash
rsync -avz --progress <user>@<vm-ip>:~/astronomy/active_learning_output_target/ ./output_target/
```
