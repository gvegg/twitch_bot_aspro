from flask import Flask, render_template, request, send_from_directory, redirect
import os
import json
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'mov', 'avi'}

# Создаем необходимые директории
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return redirect('/stream')

@app.route('/stream')
def stream():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    try:
        with open('positions.json', 'r') as f:
            positions = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        positions = {}
    
    # Инициализация позиций для новых файлов
    for file in files:
        if file not in positions:
            positions[file] = {
                'x': 0,
                'y': 0,
                'width': 300,
                'height': 200
            }
    
    return render_template('stream.html', files=files, positions=positions)


@app.route('/control', methods=['GET', 'POST'])
def control():
    if request.method == 'POST':
        # Обработка загрузки файла
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    # Показываем список загруженных файлов
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('control.html', files=files)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/save_position', methods=['POST'])
def save_position():
    # Сохраняем новые позиции элементов
    data = request.get_json()
    with open('positions.json', 'w') as f:
        json.dump(data, f)
    return 'OK'

@app.route('/get_positions')
def get_positions():
    # Возвращаем сохраненные позиции
    try:
        with open('positions.json', 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

if __name__ == '__main__':
    app.run(debug=True, port=5000)
