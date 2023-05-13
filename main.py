from flask import Flask, render_template, redirect, url_for, flash
from flask_bootstrap import Bootstrap
from flask_ckeditor import CKEditor
from datetime import date
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, current_user, logout_user
from forms import CreatePostForm, RegisterForm, LoginForm, DetectForm, ContactForm, ResponseForm
from functools import wraps
from flask import abort
import smtplib
from user_data import UserData
import pandas as pd
from xgboost import XGBClassifier
import numpy as np
import os
import pickle
import html


MY_EMAIL = "lirontheprog@gmail.com"
MY_PASSWORD = "ciexaniegletpvnu"


app = Flask(__name__)
app.config['SECRET_KEY'] = 'ndu8r4huncyh352tnemsfh78h'
ckeditor = CKEditor(app)
Bootstrap(app)

##CONNECT TO DB
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///website.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)


def send_email(email, phone, message):
    """
    Sends an email to a person who asked me a question in the 'Contact Me' page.
    :param email: the user's email.
    :param phone: the user's phone number.
    :param message: the message that I want to send to the person.
    :return: None.
    """
    email_message = f"Subject:New Message\n\nTo: {email}\nYour Phone Number: {phone}\n\n{message}"
    with smtplib.SMTP("smtp.gmail.com", port=587) as connection:
        connection.starttls()
        connection.login(user=MY_EMAIL, password=MY_PASSWORD)
        connection.sendmail(
            from_addr=MY_EMAIL,
            to_addrs=email,
            msg=f"{email_message}")


def activate_model(user_data):
    """
    Opens the dataframe and prepares the data for the training. Then, activates the XGBClassifier model on the data
    the user entered in my website, and returns an answer if he is healthy or sick.
    :param user_data: the object of the UserData class that contains the user information.
    :return: 'Healthy' if the person was found healthy, or 'Sick' if the person was found with cardiovascular disease.
    """
    cardio_disease_df = pd.read_csv("static/data/cardio_train.csv", delimiter=";").set_index("id")

    # drop duplicate rows.
    cardio_disease_df.drop_duplicates()

    # There was some exceptional data in the ap_hi and ap_lo columns, and here I remove it.
    cardio_disease_df = cardio_disease_df.drop(cardio_disease_df[cardio_disease_df.ap_hi > 250].index)
    cardio_disease_df = cardio_disease_df.drop(cardio_disease_df[cardio_disease_df.ap_hi < 40].index)
    cardio_disease_df = cardio_disease_df.drop(cardio_disease_df[cardio_disease_df.ap_lo > 200].index)
    cardio_disease_df = cardio_disease_df.drop(cardio_disease_df[cardio_disease_df.ap_lo < 20].index)

    # the age in the dataframe is in days, so to get it in years I divided the values in 365.
    cardio_disease_df["age"] = cardio_disease_df["age"] / 365

    has_cardio_disease_column = "cardio"
    co = cardio_disease_df.corr()[has_cardio_disease_column][:].abs().sort_values(ascending=False)[1:10]
    train_columns = [column for column in co.index]
    print(f"Train columns:\n{train_columns}")

    train_inputs, train_labels, max_values_list = normalize_data(cardio_disease_df, train_columns,
                                                                 has_cardio_disease_column)

    new_test_input = np.array([user_data.ap_hi, user_data.ap_lo, user_data.age,
                               user_data.cholesterol, user_data.weight, user_data.glucose, user_data.active,
                               user_data.smoke, user_data.height])

    new_test_input = np.array([test_input / max_values for test_input, max_values in zip(new_test_input, max_values_list)])
    print(new_test_input)

    xgb_clf = XGBClassifier(
        colsample_bytree=0.8,
        learning_rate=0.1,
        max_depth=3,
        n_estimators=300,
        subsample=0.8
    )

    filename = 'xgb_finalized_model.sav'
    if not os.path.isfile(filename):
        xgb_clf.fit(train_inputs, train_labels)
        pred = xgb_clf.predict(new_test_input.reshape(1, 9))

        print("save model at local drive: xgb_finalized_model.sav")
        with open(filename, 'wb') as fn:
            pickle.dump(xgb_clf, fn)
    else:
        with open(filename, 'rb') as infile:
            xgb_clf = pickle.load(infile)
            pred = xgb_clf.predict(new_test_input.reshape(1, 9))

    if pred == [0]:
        return "Healthy"
    return "Sick"


def normalize_data(df, input_columns, output_column):
    """
    Normalizes the input and label's data and makes a list of the max value in each column in the dataframe.
    :param df: the cardiovascular disease dataframe.
    :param input_columns: the columns that are being entered to the model as the input data.
    :param output_column: the label column data that describes if the person has a heart disease or not.
    :return: the columns and labels data after normalizing the data and converting the data to numpy arrays, and
    a list of the max value in each column in the dataframe.
    """
    max_list = []
    for field in input_columns:
        print("field", field)
        max_list.append(max(df[field]))
        df[field] = df[field] / max(df[field])

    cardio_disease_info = df[input_columns].to_numpy()
    has_cardio = df[output_column].to_numpy()
    return cardio_disease_info, has_cardio, max_list


def convert_user_data(cholesterol, glucose, smoke, active):
    """
    Converts the data the user enters in the gender, cholesterol, glucose, smoke, alcohol and active columns to the
    format of the train data.
    :param cholesterol: the cholesterol level of the person.
    :param glucose: the glucose level of the person.
    :param smoke: 'Yes' if the person smokes or 'No' if not.
    :param active: 'Yes' if the person does any activity or 'No' if not.
    :return: The gender, cholesterol, glucose, smoke, alcohol and active value fields in the format of the train data.
    """
    if cholesterol == "less than 200 mg/dL":
        cholesterol = 1
    elif cholesterol == "200 to 239 mg/dL":
        cholesterol = 2
    else:
        cholesterol = 3

    if glucose == "less than 100 mg/dL":
        glucose = 1
    elif glucose == "100 to 125 mg/dL":
        glucose = 2
    else:
        glucose = 3

    if smoke == "No":
        smoke = 0
    else:
        smoke = 1

    if active == "No":
        active = 0
    else:
        active = 1

    return cholesterol, glucose, smoke, active


# Website functions & classes
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    name = db.Column(db.String(100))


class BlogPost(db.Model):
    __tablename__ = "blog_posts"
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(250), unique=True, nullable=False)
    subtitle = db.Column(db.String(250), nullable=False)
    date = db.Column(db.String(250), nullable=False)
    body = db.Column(db.Text, nullable=False)
    img_url = db.Column(db.String(250), nullable=False)


class MessageWaiting(db.Model):
    __tablename__ = "emails_waiting"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(250), unique=True, nullable=False)
    phone = db.Column(db.String(250), nullable=False)
    message = db.Column(db.Text, nullable=False)


db.create_all()


#admin-only decorator
def admin_only(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        #If id is not 1 then return abort with 403 error
        if current_user.id != 1:
            return abort(403)
        return f(*args, **kwargs)
    return decorated_function


@app.route('/', methods=["GET", "POST"])
def home():
    """
    Renders the home page and handles form submissions for cardio disease detection.
    :return: If a valid form is submitted, it redirects to the 'result' page.
    Otherwise, it renders the 'index.html' template with the form.
    """
    form = DetectForm()
    if form.validate_on_submit():
        # Get user health data
        age = form.age.data
        height = form.height.data
        weight = form.weight.data
        ap_hi = form.ap_hi.data
        ap_lo = form.ap_lo.data
        cholesterol = form.cholesterol.data
        glucose = form.glucose.data
        smoke = form.smoke.data
        active = form.active.data

        cholesterol, glucose, smoke, active = convert_user_data(cholesterol, glucose, smoke, active)

        user_data = UserData(age, height, weight, ap_hi, ap_lo, cholesterol, glucose, smoke, active)
        user_status = activate_model(user_data)

        return redirect(url_for("result", user_status=user_status))

    return render_template("index.html", form=form)


@app.route('/register', methods=["GET", "POST"])
def register():
    """
    Renders the registration page and handles user registration.
    :return: If a valid form is submitted, it redirects to the home page.
    Otherwise, it renders the 'register.html' template with the form.
    """
    form = RegisterForm()
    if form.validate_on_submit():
        if User.query.filter_by(email=form.email.data).first():
            flash("You've already signed up with that email, log in instead!")
            return redirect(url_for("login"))

        hash_and_salted_password = generate_password_hash(
            form.password.data,
            method="pbkdf2:sha256",
            salt_length=8
        )

        new_user = User(
            email=form.email.data,
            name=form.name.data,
            password=hash_and_salted_password
        )
        db.session.add(new_user)
        db.session.commit()

        #Log in and authenticate user after adding details to the database.
        login_user(new_user)

        return redirect(url_for("home"))

    return render_template("register.html", form=form)


@app.route('/login', methods=["GET", "POST"])
def login():
    """
    Renders the login page and handles user authentication.
    :return: If a valid form is submitted and the user is authenticated, it redirects
             to the home page. Otherwise, it renders the 'login.html' template with
             the form.
    """
    form = LoginForm()
    if form.validate_on_submit():
        # Login and validate the user.
        email = form.email.data
        password = form.password.data

        user = User.query.filter_by(email=email).first()
        if not user:
            flash("That email does not exist, please try again.")
            return redirect(url_for('login'))
        elif not check_password_hash(pwhash=user.password, password=password):
            flash('Password incorrect, please try again.')
            return redirect(url_for('login'))
        else:
            login_user(user)
            return redirect(url_for("home"))

    return render_template("login.html", form=form)


@app.route('/logout')
@login_required
def logout():
    """
    Logs out the currently authenticated user.
    :return: It redirects the user to the home page after logging out.
    """
    logout_user()
    return redirect(url_for('home'))


@app.route('/show-all-posts', methods=["GET", "POST"])
def show_all_posts():
    """
    Renders a page that displays all the health posts.
    :return: It renders the 'show_all_posts.html' template with the posts from the database.
    """
    posts = BlogPost.query.all()
    return render_template("show_all_posts.html", all_posts=posts)


@app.route('/results/<user_status>')
def result(user_status):
    """
    Renders the result page with the user status (has/doesn't have cardio disease).
    :param user_status: 'Healthy' if the person was found healthy, or 'Sick' if he was found with cardio disease.
    :return: It renders the 'result.html' template and displays the result according to the user_status.
    """
    return render_template("result.html", user_status=user_status)


@app.route("/post/<int:post_id>")
def show_post(post_id):
    """
    Renders a specific post page.
    :param post_id: The unique identifier of the post.
    :return: It renders the 'post.html' template with the requested post.
    """
    requested_post = BlogPost.query.get(post_id)
    return render_template("post.html", post=requested_post)


@app.route("/about")
def about():
    """
    Renders the about page with some details about the website, and me as the creator of this website.
    :return: It renders the 'about.html' template.
    """
    return render_template("about.html")


@app.route("/contact", methods=['GET', 'POST'])
def contact():
    """
    Renders the contact page and handles form submissions for users' messages. If a valid form is
    submitted, adds the user data to my database.
    :return: If a valid form is submitted, it renders the 'contact.html' template
             with a success message. Otherwise, it renders the 'contact.html' template
             with the contact form.
    """
    form = ContactForm()
    if form.validate_on_submit():
        if MessageWaiting.query.filter_by(email=form.email.data).first():
            flash("You can't send me another email until I come back to you.")
            return redirect(url_for("contact"))
        else:
            email = form.email.data
            phone_number = form.phone_number.data
            message = form.message.data

            new_user_message = MessageWaiting(
                email=email,
                phone=phone_number,
                message=message,
            )
            db.session.add(new_user_message)
            db.session.commit()

            return render_template("contact.html", form=form, msg_sent=True)

    return render_template("contact.html", form=form, msg_sent=False)


@app.route("/new-post", methods=["GET", "POST"])
@admin_only
def add_new_post():
    """
    Renders the page for adding a new post and handles form submissions. If a valid form is
    submitted, adds the post data to my database.
    :return: If a valid form is submitted, it redirects to the 'show_all_posts' page.
             Otherwise, it renders the 'make-post.html' template with the form.
    """
    form = CreatePostForm()
    if form.validate_on_submit():
        new_post = BlogPost(
            title=form.title.data,
            subtitle=form.subtitle.data,
            body=form.body.data,
            img_url=form.img_url.data,
            date=date.today().strftime("%B %d, %Y")
        )
        db.session.add(new_post)
        db.session.commit()
        return redirect(url_for("show_all_posts"))
    return render_template("make-post.html", form=form)


@app.route("/edit-post/<int:post_id>", methods=["GET", "POST"])
@admin_only
def edit_post(post_id):
    """
    Renders the page for editing an existing post and handles form submissions.
    :param post_id: The id of the post that will be edited.
    :return: If a valid form is submitted, it redirects to the 'show_post' page with the post edited.
             Otherwise, it renders the 'make-post.html' template with the form for editing.
    """
    post = BlogPost.query.get(post_id)
    edit_form = CreatePostForm(
        title=post.title,
        subtitle=post.subtitle,
        img_url=post.img_url,
        body=post.body
    )
    if edit_form.validate_on_submit():
        post.title = edit_form.title.data
        post.subtitle = edit_form.subtitle.data
        post.img_url = edit_form.img_url.data
        post.body = edit_form.body.data
        db.session.commit()
        return redirect(url_for("show_post", post_id=post.id))

    return render_template("make-post.html", form=edit_form, is_edit=True)


@app.route("/delete/<int:post_id>")
@admin_only
def delete_post(post_id):
    """
    Deletes a post from the database and redirects to the 'show_all_posts' page.
    :param post_id: The id of the post to be deleted.
    :return: It redirects to the 'show_all_posts' page.
    """
    post_to_delete = BlogPost.query.get(post_id)
    db.session.delete(post_to_delete)
    db.session.commit()
    return redirect(url_for('show_all_posts'))


@app.route('/messages-waiting', methods=["GET", "POST"])
@admin_only
def show_all_messages():
    """
    Renders the page that displays all the waiting messages from users.
    :return: It renders the 'show_all_messages.html' template with the users' messages from the database.
    """
    messages = MessageWaiting.query.all()
    return render_template("show_all_messages.html", all_messages=messages)


@app.route("/message/<int:message_id>", methods=["GET", "POST"])
@admin_only
def show_message(message_id):
    """
    Renders an individual message page and handles form submissions for response about the current message.
    If a valid response form is submitted, sends an email to the user that answers his question
    and deletes his question details from the database.
    :param message_id: The id of the message to be displayed and responded to.
    :return: If a valid response form is submitted, it redirects to the 'show_all_messages' page.
             Otherwise, it renders the 'message.html' template with the message details and form.
    """
    requested_message = MessageWaiting.query.get(message_id)
    form = ResponseForm()
    if form.validate_on_submit():
        message_to_user = form.message_to_user.data
        message_to_user = message_to_user.replace('<p>', '').replace('</p>', '')
        message_to_user = html.unescape(message_to_user)
        message_to_user.encode('utf-8')

        user_message = MessageWaiting.query.get(message_id)
        email = user_message.email
        phone = user_message.phone

        send_email(email, phone, message_to_user)

        delete_message(message_id)

        return redirect(url_for('show_all_messages'))

    return render_template("message.html", user_message=requested_message, form=form)


@app.route("/delete/<int:message_id>")
@admin_only
def delete_message(message_id):
    """
    Deletes a user message with the given message id from the database.
    :param message_id: The id of the message to be deleted.
    :return: It redirects to the 'show_all_messages' page after deleting the current message.
    """
    message_to_delete = MessageWaiting.query.get(message_id)
    db.session.delete(message_to_delete)
    db.session.commit()
    # return redirect(url_for('show_all_messages'))


if __name__ == "__main__":
    app.run(debug=True)
