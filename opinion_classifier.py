from flask import (Blueprint, flash, g, redirect,
                   render_template, request, session, url_for)

from werkzeug.exceptions import abort
from werkzeug.utils import secure_filename

from flask import current_app as app
import os

from app.predict import *

bp = Blueprint('opinion_classifier', __name__)


@bp.route('/', methods=("GET", "POST"))
@bp.route('/index', methods=("GET", "POST"))
def index():
    body = None
    if request.method == "POST":
        body = request.form['userOpinion']

    res = Predict(body)
    return render_template('opinion_classifier/single_estimator.html', result = res)

@bp.route("/thong-ke-quan-diem", methods=("GET", "POST"))
def list_opinion():
    res = None
    if request.method == "POST":
        if request.files:

            file = request.files["formFile"]

            if file:

                file_path = os.path.join(
                    app.config["UPLOAD_FOLDER"], file.filename)
                
                file.save(file_path)

                res = PredictList(file_path, os.path.splitext(file.filename))

                print("res:",res)

                # if os.path.exists(file_path):
                #     os.remove(file_path)

    return render_template("opinion_classifier/list-opinion.html", result = res)

@bp.route('/chi-tiet/<text>', methods=("GET", "POST"))
def opinion_detail(text : str):
    vocab : dict[str, str] = feature_vocabulary
    origin_text : str = text
    for key in feature_vocabulary:
        print(text, key)
        if text.find(key) != -1:
            if vocab[key] == "Positive":
                 text = text.replace(key, f'<span class="font-positive">{key}</span>')
            if vocab[key] == "Negative":
                text = text.replace(
                    key, f'<span class="font-negative">{key}</span>')
            if vocab[key] == "Neutral":
                text = text.replace(
                    key, f'<span class="font-neutral">{key}</span>')
            print(text)
    return render_template("opinion_classifier/detail.html", result = (origin_text, text))

@bp.route("/test")
def test():
    return "<h1>Hello world</h1>"