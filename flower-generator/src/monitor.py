import os
import time
import imageio
from PIL import Image

def create_progress_gif():
    # Пути к файлам
    samples_dir = "../outputs/train_samples"
    output_gif = "../outputs/training_progress.gif"
    
    print("Starting training monitor...")
    
    while True:
        try:
            # Получаем список всех файлов сэмплов
            sample_files = [f for f in os.listdir(samples_dir) 
                          if f.startswith('epoch_') and f.endswith('.png')]
            
            if not sample_files:
                print("No samples found. Waiting...")
                time.sleep(10)
                continue
            
            # Сортируем по номеру эпохи
            sample_files.sort(key=lambda x: int(x.split('_')[1]))
            
            # Создаем список изображений для GIF
            images = []
            for filename in sample_files:
                try:
                    img_path = os.path.join(samples_dir, filename)
                    img = Image.open(img_path)
                    images.append(img)
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
            
            # Сохраняем GIF (если есть изображения)
            if images:
                images[0].save(
                    output_gif,
                    save_all=True,
                    append_images=images[1:],
                    duration=50,  # 50ms между кадрами
                    loop=0        # Бесконечный цикл
                )
                print(f"Updated GIF with {len(images)} samples")
            
            # Проверяем каждые 30 секунд
            time.sleep(30)
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            time.sleep(10)

if __name__ == "__main__":
    # Создаем необходимые директории
    os.makedirs("../outputs/train_samples", exist_ok=True)
    os.makedirs("../outputs", exist_ok=True)
    
    create_progress_gif()
