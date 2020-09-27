# -*- encoding: utf-8 -*-
"""
License: MIT
Copyright (c) 2019 - present AppSeed.us
"""

from flask_wtf          import FlaskForm
from flask_wtf.file     import FileField, FileRequired
from wtforms            import StringField, TextAreaField, SubmitField, PasswordField
from wtforms.validators import InputRequired, Email, DataRequired
from wtforms.validators import ValidationError, DataRequired, Email, EqualTo,Length

class LoginForm(FlaskForm):
	username    = StringField  (u'Username'        , validators=[DataRequired()])
	password    = PasswordField(u'Password'        , validators=[DataRequired()])

class RegisterForm(FlaskForm):
	name        = StringField  (u'Name'      )
	username    = StringField  (u'Username'  , validators=[DataRequired()])
	password    = PasswordField(u'Password'  , validators=[DataRequired()])
	email       = StringField  (u'Email'     , validators=[DataRequired(), Email()])


                
                
class EditProfileForm(FlaskForm): 
    name = StringField('Real Name', validators=[Length(0, 64)])
    about_me = TextAreaField('About Me')
    localisation = TextAreaField('Localisation')
    job = TextAreaField('Job')
    submit = SubmitField('Submit')
    #photo = FileField()
    
   

