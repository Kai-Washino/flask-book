from flask import Flask, current_app, g, render_template, url_for, request, redirect
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

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/contact/complete", methods = ["GET", "POST"])
def contact_complete():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        description = request.form["description"]        
        return redirect(url_for("contact_complete"))
    
    return render_template("contact_complete.html")

with app.test_request_context():
    print(url_for("index"))
    print(url_for("hello-endpoint", name="world"))
    print(url_for("show_name", name="ichiro", page="1")) # このpageはフロントからバックへ追加の情報を渡すイメージ

