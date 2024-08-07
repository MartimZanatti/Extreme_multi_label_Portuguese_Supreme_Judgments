from flask import Flask, request, send_file, jsonify, send_from_directory
import os
import tempfile
import subprocess
#from pypandoc.pandoc_download import download_pandoc
#download_pandoc()

app = Flask(__name__)
app.config['JSON_AS_ASCII']=False

@app.route('/', methods=["GET"])
@app.route('/<path:path>', methods=["GET"])
def send_report(path=None):
    if path is None:
        path = "index.html"
    return send_from_directory('static', path)

@app.route("/", methods=["POST"])
def handle_post():
    _, file_extension = os.path.splitext(request.files["file"].filename)
    uploaded_file = tempfile.NamedTemporaryFile(suffix=file_extension, delete=False)
    request.files["file"].save(uploaded_file)
    area = request.form.get("area")
    uploaded_file.flush()
    uploaded_file.close()


    #caixa preta
    resp = subprocess.Popen(f"python black-box-cli.py {uploaded_file.name} {file_extension} {area}", shell=True, stdout=subprocess.PIPE).stdout.read()
    os.unlink(uploaded_file.name)
    return resp




if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8999)

