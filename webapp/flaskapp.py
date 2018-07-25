import os
from flask import Flask, flash,request, redirect, url_for
from flask import send_from_directory
from flask import render_template
from werkzeug.utils import secure_filename
# from backend.ProcessVoucher import processIncomingFile
from backend.ProcessVoucher import processIncomingFile

TEMP = 'uploads'
UPLOAD_FOLDER = 'webapp/uploads'
IMAGE_DUMP_FOLDER = '../backend/imagedump'
CROPPED_DUMP_FOLDER = '../backend/chardump'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['IMAGE_DUMP_FOLDER'] = IMAGE_DUMP_FOLDER
app.config['CROPPED_DUMP_FOLDER'] = CROPPED_DUMP_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/process',methods=['GET','POST'])
def process():
    if request.method == 'POST':
        file_name=request.form['name']
        print(file_name)
        account_number,date,mobile,cropped_file_path = processIncomingFile(file_name)
        # print(acc,y,z)
        all_cropped_file_path = []
        for path in cropped_file_path:
            all_cropped_file_path.append('http://127.0.0.1:5000/' + path)

        file_location = 'http://127.0.0.1:5000/uploads/' + file_name
        processed_file_stage1 = 'http://127.0.0.1:5000/imagedump/' + 'matches.jpg'
        processed_file_stage2 = 'http://127.0.0.1:5000/imagedump/' + 'aligned.jpg'
        return render_template('index.html', **locals())


@app.route('/',methods=['GET'])
def welcomePage():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # return render_template('index.html', file_location=file_location)
            return redirect(url_for('uploaded_file',filename=filename))
    return render_template('index.html')


@app.route('/show/<filename>')
def uploaded_file(filename):
    # return send_from_directory(app.config['UPLOAD_FOLDER'],filename)
    file_name = filename
    file_location = 'http://127.0.0.1:5000/uploads/' + filename
    return render_template('index.html',**locals())

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(TEMP, filename)

# @app.route('/process/show/<original_file>/<processed_file_stage1>')
# def processed_file(original_file,processed_file_stage1):
#     uploaded_file(original_file)
#     processed_file_stage1 = 'http://127.0.0.1:5000/imagedump/' + processed_file_stage1
#     return render_template('index.html',**locals())

@app.route('/imagedump/<filename>')
def send_processed_file(filename):
    print(filename)
    return send_from_directory(IMAGE_DUMP_FOLDER, filename)

@app.route('/backend/chardump/<filename>')
def send_cropped_file(filename):
    # temp_file = filename.split('/')
    print(filename)
    return send_from_directory(CROPPED_DUMP_FOLDER, filename)


def runApp():
    app.run()

if __name__ == '__main__':
    app.run()