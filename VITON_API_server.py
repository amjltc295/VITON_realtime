from flask import Flask, request
from PIL import Image
import pickle

app = Flask(__name__)


@app.route("/", methods=["POST"])
def home():
    print(request.files)
    img = Image.open(request.files['files'])
    print(img)
    img_pickle = pickle.dumps(img)
    return img_pickle
