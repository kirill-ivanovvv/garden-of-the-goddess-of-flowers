import torch
from model import Generator
import matplotlib.pyplot as plt
import numpy as np
import os

# Конфигурация
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100
model_path = "../outputs/models/generator_epoch_50.pth"  # Укажите ваш путь

# Загрузка модели
generator = Generator().to(device)
generator.load_state_dict(torch.load(model_path, map_location=device))
generator.eval()

# Генерация
with torch.no_grad():
    noise = torch.randn(1, latent_dim, device=device)
    generated = generator(noise).cpu().numpy()[0]
    
    # Денормализация [-1, 1] -> [0, 1]
    generated = np.transpose(generated, (1, 2, 0))
    generated = (generated * 0.5) + 0.5

# Сохранение
os.makedirs("../outputs/images", exist_ok=True)
plt.figure(figsize=(4, 4))
plt.imshow(generated)
plt.axis('off')
plt.savefig("../outputs/images/generated_flower.png", bbox_inches='tight', pad_inches=0)
plt.close()

print("Цветок сгенерирован и сохранён в outputs/images/generated_flower.png")
