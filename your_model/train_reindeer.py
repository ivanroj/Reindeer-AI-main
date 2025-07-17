import os
import torch
from ultralytics import YOLO

def main():
    # Проверяем доступность CUDA
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA не доступна. Установите версию PyTorch с поддержкой CUDA: "
            "https://pytorch.org/get-started/locally/"
        )
    # Используем первую видеокарту (GPU 0)
    device = 0
    print(f"Using GPU device: {device} (CUDA)")

    # Путь к файлу data.yaml
    data_yaml = os.path.join(os.getcwd(), 'data.yaml')
    # Выбор предобученной модели (yolov8n.pt, yolov8s.pt и т.д.)
    base_model = 'yolov8s.pt'

    # Параметры обучения
    epochs = 50         # число эпох
    imgsz = 640         # размер кадра
    batch = 16          # размер батча
    project = 'runs/train'
    name = 'reindeer_detector'

    # Инициализация и обучение модели
    model = YOLO(base_model)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,    # GPU 0
        project=project,
        name=name,
        verbose=True
    )

if __name__ == '__main__':
    main()