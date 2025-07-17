from ultralytics import YOLO
import os


def run_inference(model_path, source_dir, out_dir='runs/detect/inference', conf=0.6):
    # Загружаем модель
    model = YOLO(model_path)

    # Прогоняем предсказание
    results = model.predict(
        source=source_dir,
        conf=conf,
        save=True,        # сохранять прорисованные изображения
        save_txt=True,   # текстовые метки можно отключить
        project=out_dir,  # куда сохранять
        name='inference'  # подпапка внутри project
    )

    count = len(results[0].boxes)
    # путь к сохранённому изображению:
    image_out = os.path.join(out_dir, 'inference', os.path.basename(source_dir))
    return image_out, count

if __name__ == '__main__':
    model_path = 'runs/train/reindeer_detector4/weights/best.pt'
    source_dir = 'dataset/images/val'  # можно и одиночный файл
    run_inference(model_path, source_dir)