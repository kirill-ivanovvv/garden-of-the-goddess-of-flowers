import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class FlowerDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = [
            os.path.join(root_dir, f) 
            for f in os.listdir(root_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.transform = transforms.Compose([
            transforms.Resize((16, 16)),  # Фиксированный размер
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
            return self.transform(img)
        except Exception as e:
            print(f"Error loading {self.image_paths[idx]}: {str(e)}")
            # Возвращаем нулевой тензор в случае ошибки
            return torch.zeros(3, 16, 16)
