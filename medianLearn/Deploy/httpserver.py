from flask import Flask, request, jsonify
import io
import cv2 as cv
from PIL import Image

app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello World!'


@app.route('/test', methods=['POST'])
def test():
    name = request.form.get("name")

    file = request.files.get("file")
    img_bype = file.read()
    img = Image.open(io.BytesIO(img_bype))


    return jsonify({'name': name, 'filesize': img.size})


if __name__ == '__main__':
    app.run()
