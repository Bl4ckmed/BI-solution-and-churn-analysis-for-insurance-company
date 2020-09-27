# -*- encoding: utf-8 -*-
"""
License: MIT
Copyright (c) 2019 - present AppSeed.us
"""

from app         import db
from flask_login import UserMixin
from datetime import datetime
from hashlib import md5
class User(UserMixin, db.Model):

    id       = db.Column(db.Integer,     primary_key=True)
    user     = db.Column(db.String(64),  unique = True)
    email    = db.Column(db.String(120), unique = True)
    password = db.Column(db.String(500))
    files=db.relationship('Files', lazy='dynamic')
    about_me = db.Column(db.String(140))
    localisation = db.Column(db.String(140))
    job = db.Column(db.String(140))
    last_seen = db.Column(db.DateTime, default=datetime.utcnow)
    def __init__(self, user, email, password):
        self.user       = user
        self.password   = password
        self.email      = email

    def __repr__(self):
        return str(self.id) + ' - ' + str(self.user)

    def save(self):

        # inject self into db session    
        db.session.add ( self )

        # commit change and save the object
        db.session.commit( )
    def avatar(self, size):
        digest = md5(self.email.lower().encode('utf-8')).hexdigest()
        return 'https://www.gravatar.com/avatar/{}?d=identicon&s={}'.format(
            digest, size)

        return self 
class Files(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(140))
    size = db.Column(db.String(140))
    hash = db.Column(db.String(140))
    date = db.Column(db.String(140))
    counter = db.Column(db.String(140))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))    

    def __repr__(self):
        return '<Post {}>'.format(self.body)