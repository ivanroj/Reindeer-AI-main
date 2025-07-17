import os
import glob
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from your_model.inference import run_inference

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'runs/detect/web'
MODEL_PATH = 'runs/train/reindeer_detector4/weights/best.pt'
CONF_THRESHOLD = 0.6

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_files = request.files.getlist('files')
        results = []  # список словарей: {{filename, out_path, count}}
        total_count = 0
        for file in uploaded_files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                src_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(src_path)
                # Запускаем инференс по одному файлу
                out_dir = app.config['RESULT_FOLDER']
                # run_inference сохраняет изображения в out_dir/inference
                run_inference(MODEL_PATH, src_path, out_dir, conf=CONF_THRESHOLD)
                # В результирующей папке будет файл с тем же имя
                candidates = glob.glob(os.path.join(out_dir, 'inference*'))
                # Берём ту, что была создана последней
                latest = max(candidates, key=os.path.getctime)
                processed_file = os.path.join(latest, filename)

                # Копируем её в статик
                dst = os.path.join('static/results', filename)
                os.replace(processed_file, dst)

                image_out, count = run_inference(MODEL_PATH, src_path, out_dir, conf=CONF_THRESHOLD)
                total_count += count
                results.append({
                    'filename': filename,
                    'url': url_for('static', filename=f"results/{filename}"),
                    'count': count,
                    'found': count > 0
                })
        # Переносим все processed изображения в static/results/
        os.makedirs('static/results', exist_ok=True)
        for r in results:
            src = os.path.join(app.config['RESULT_FOLDER'], 'inference', r['filename'])
            dst = os.path.join('static/results', r['filename'])
            if os.path.exists(src):
                os.replace(src, dst)
        return render_template('result.html', results=results, total_count=total_count)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)