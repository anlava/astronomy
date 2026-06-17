#!/usr/bin/env bash
set -euo pipefail

# Скрипт настройки свежей Ubuntu 22.04 VM для про astronomy.
# Запускать от имени обычного пользователя с sudo.

REPO_URL="https://github.com/anlava/astronomy.git"
INSTALL_DIR="$HOME/astronomy"
PYTHON_VERSION="3.11"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "Обновление пакетов..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    software-properties-common \
    curl \
    wget \
    git \
    htop \
    screen \
    tmux \
    rsync \
    libffi-dev \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    liblzma-dev \
    zlib1g-dev

log "Установка Python ${PYTHON_VERSION} из deadsnakes..."
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-distutils

# Убедимся, что pip установлен
python${PYTHON_VERSION} -m ensurepip --upgrade || true

log "Клонирование репозитория..."
if [ -d "$INSTALL_DIR" ]; then
    log "Директория $INSTALL_DIR уже существует, обновляем..."
    cd "$INSTALL_DIR"
    git pull origin main
else
    git clone "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

log "Создание виртуального окружения..."
python${PYTHON_VERSION} -m venv .venv
source .venv/bin/activate

log "Обновление pip/setuptools/wheel..."
pip install --upgrade pip setuptools wheel

log "Установка Python-зависимостей..."
pip install -r requirements.txt

log "Проверка импортов..."
python -c "import polars, numpy, pandas, catboost, pywt, numba; print('OK')"

log "Настройка завершена. Рабочая директория: $INSTALL_DIR"
log "Активировать окружение: source $INSTALL_DIR/.venv/bin/activate"
