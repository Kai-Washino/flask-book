import logging
import os
from email_validator import validate_email, EmailNotValidError
from flask_debugtoolbar import DebugToolbarExtension
from flask import (
    Flask, 
    current_app, 
    g, 
    render_template, 
    url_for, 
    request, 
    redirect,
    flash,
)

from flask_mail import Mail, Message

app = Flask(__name__)
app.config["SECRET_KEY"] = "jadksjfaiwejfkljk"
app.logger.setLevel(logging.DEBUG)

app.config["DEBUG_TB_INTERCEPT_REDIRECTS"] = False

app.config["MAIL_SERVER"] = os.environ.get("MAIL_SERVER")
app.config["MAIL_PORT"] = os.environ.get("MAIL_PORT")
app.config["MAIL_USE_TLS"] = os.environ.get("MAIL_USE_TLS")
app.config["MAIL_USERNAME"] = os.environ.get("MAIL_USERNAME")
app.config["MAIL_PASSWORD"] = os.environ.get("MAIL_PASSWORD")
app.config["MAIL_DEFAULT_SENDER"] = os.environ.get("MAIL_DEFAULT_SENDER")

mail = Mail(app)

toolbar = DebugToolbarExtension(app)

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
        is_valid = True

        if not username:
            flash("ユーザ名は必須です")
            is_valid = False

        if not email:
            flash("メールアドレスは必須です")
            is_valid = False

        try:
            validate_email(email)
        except EmailNotValidError:
            flash("メールアドレスの形式で入力してください")
            is_valid = False

        if not description:
            flash("問い合わせ内容は必須です")
            is_valid = False
        
        if not is_valid:            
            return redirect(url_for("contact"))
        
        send_email(
            email,
            "問い合わせありがとうございました。",
            "contact_mail",
            username=username,
            description=description,
        )
        
        flash("問い合わせありがとうございました．")
        return redirect(url_for("contact_complete"))
    
    return render_template("contact_complete.html")

with app.test_request_context():
    print(url_for("index"))
    print(url_for("hello-endpoint", name="world"))
    print(url_for("show_name", name="ichiro", page="1")) # このpageはフロントからバックへ追加の情報を渡すイメージ

def send_email(to, subject, template, **kwargs):
    """メールを送信する関数"""
    msg = Message(subject, recipients=[to])
    msg.body = render_template(template + ".txt", **kwargs)
    msg.html = render_template(template + ".html", **kwargs)
    mail.send(msg)