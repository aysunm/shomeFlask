from time import sleep
from flask import Flask, stream_with_context, request, Response, flash, render_template, redirect, url_for, jsonify, \
    json
from app.model import init_model, predict_model

app = Flask(__name__)
app.secret_key = '!$w4wW~o|~9OVFQ'  # !!change this with random key!!

init_model()

def stream_template(template_name, **context):
    app.update_template_context(context)
    t = app.jinja_env.get_template(template_name)
    rv = t.stream(context)
    rv.disable_buffering()
    return rv


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict')
def predict():
    sentence = request.args.get("cmd")
    return app.response_class(response=json.dumps(predict_model(sentence)), mimetype='application/json', status=200)
