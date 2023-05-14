from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, PasswordField, IntegerField, FloatField, SelectField
from wtforms.validators import DataRequired, URL, Email
from flask_ckeditor import CKEditorField


class CreatePostForm(FlaskForm):
    """
    A form class for creating new posts.
    """
    title = StringField("Blog Post Title", validators=[DataRequired()])
    subtitle = StringField("Subtitle", validators=[DataRequired()])
    img_url = StringField("Blog Image URL", validators=[DataRequired(), URL()])
    body = CKEditorField("Blog Content", validators=[DataRequired()])
    submit = SubmitField("Submit Post")


class RegisterForm(FlaskForm):
    """
    A form class for user registration.
    """
    email = StringField("Email", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired()])
    name = StringField("Name", validators=[DataRequired()])
    submit = SubmitField("Sign Me Up!")


class LoginForm(FlaskForm):
    """
    A form class for user login.
    """
    email = StringField("Email", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Let Me In!")


class DetectForm(FlaskForm):
    """
    A form class for cardiovascular disease detection.
    """
    age = FloatField("Age", validators=[DataRequired()])
    height = IntegerField("Height", validators=[DataRequired()])
    weight = FloatField("Weight", validators=[DataRequired()])
    ap_hi = IntegerField("Systolic blood pressure", validators=[DataRequired()])
    ap_lo = IntegerField("Diastolic blood pressure", validators=[DataRequired()])

    cholesterol = SelectField("Enter your cholesterol level:",
                              choices=["less than 200 mg/dL", "200 to 239 mg/dL", "above 240 mg/dL"],
                              validators=[DataRequired()])
    glucose = SelectField("Enter your glucose level:",
                          choices=["less than 100 mg/dL", "100 to 125 mg/dL", "above 125 mg/dL"],
                          validators=[DataRequired()])
    smoke = SelectField("Do you smoke?",
                        choices=["Yes", "No"],
                        validators=[DataRequired()])
    active = SelectField("Do you exercise?",
                         choices=["Yes", "No"],
                         validators=[DataRequired()])
    submit = SubmitField("Detect!", render_kw={"onclick": "loader()"})


class ContactForm(FlaskForm):
    """
    A form class for users to contact me if they have questions.
    """
    email = StringField("Email Address", validators=[DataRequired(), Email()])
    phone_number = StringField("Phone Number", validators=[DataRequired()])
    message = StringField("Message", validators=[DataRequired()])
    submit = SubmitField("Send")


class ResponseForm(FlaskForm):
    """
    A form class for sending a response to users that asked me a question in the 'Contact Me' page.
    """
    message_to_user = CKEditorField("Answer For The User:", validators=[DataRequired()])
    submit = SubmitField("Send To User")


