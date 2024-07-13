from flask import Flask, render_template, url_for

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    return "Hello, Flaskbook!"

@app.route("/hello/<name>",
           methods=["GET", "POST"],
           endpoint="hello-endpoint")
def hello(name):
    return f"Hello, {name}!"

@app.route("/name/<name>")
def show_name(name):
    return render_template("index.html",
                           name=name)

with app.test_request_context():
    print(url_for("index"))
    print(url_for("hello-endpoint", name="world"))
    print(url_for("show_name", name="ichiro", page="1")) # このpageはフロントからバックへ追加の情報を渡すイメージ