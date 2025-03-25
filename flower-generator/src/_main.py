import torch

# Создаём случайный тензор размером 5x3
x = torch.rand(5, 3)

# Выводим результат
print("Случайный тензор:")
print(x)

# Дополнительная проверка установки CUDA
print("\nДополнительная информация:")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA доступно: {torch.cuda.is_available()}")
