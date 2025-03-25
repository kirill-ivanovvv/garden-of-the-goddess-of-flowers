#!/bin/bash

# Создаем группу процессов
cleanup() {
    echo "Завершение всех процессов..."
    pkill -P $$  # Убиваем все дочерние процессы
    exit 0
}

trap cleanup SIGINT SIGTERM

# Запускаем процессы
python flower-generator/src/train.py &
train_pid=$!

python flower-generator/src/monitor.py &
monitor_pid=$!

# Ждем завершения всех фоновых процессов
wait -n $train_pid $monitor_pid
