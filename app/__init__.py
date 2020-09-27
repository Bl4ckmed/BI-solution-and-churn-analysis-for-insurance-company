# -*- encoding: utf-8 -*-
"""
License: MIT
Copyright (c) 2019 - present AppSeed.us
"""

import os

from flask            import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login      import LoginManager
from flask_bcrypt     import Bcrypt
import hashlib
import os
import sqlite3
from datetime import date
from flask import Flask ,render_template ,request, redirect, url_for ,g ,jsonify ,send_from_directory , send_file
from flask_bootstrap import Bootstrap

# Grabs the folder where the script runs.
basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
bootstrap = Bootstrap(app)

app.config.from_object('app.configuration.Config')
app.config['UPLOAD_FOLDER'] = 'app/uploads'

DATABASE='/home/www/app/database.db'
db = SQLAlchemy  (app) # flask-sqlalchemy
migrate = Migrate(app, db)
bc = Bcrypt      (app) # flask-bcrypt

lm = LoginManager(   ) # flask-loginmanager
lm.init_app(app) # init the login manager


# Setup database
@app.before_first_request
def initialize_database():
    db.create_all()

# Import routing, models and Start the App
from app import views, models
if __name__ == '__main__':
    app.run(debug=True)