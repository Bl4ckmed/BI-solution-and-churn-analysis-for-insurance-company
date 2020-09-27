# -*- encoding: utf-8 -*-
"""
License: MIT
Copyright (c) 2019 - present AppSeed.us
"""

# Python modules
import logging
import hashlib
import os
import sqlite3
from datetime import date
from flask import Flask ,render_template ,request, redirect, url_for, g,jsonify, send_from_directory, send_file

# Flask modules
from flask import render_template, request, url_for, redirect, send_from_directory,flash
from flask_login import login_user, logout_user, current_user, login_required
from werkzeug.exceptions import HTTPException, NotFound, abort

# App modules
from app import app, lm, db, bc
from app.models import User ,Files
from app.forms  import LoginForm, RegisterForm,EditProfileForm
import mysql.connector


# connecting to the mysql server
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password=""
)
print(mydb)
mycursor = mydb.cursor(buffered=True)
mycursor.execute("USE bd_avplus")
def get_db():
	db = getattr(g, '_database', None)
	if db is None:
		db = g._database = sqlite3.connect('app/database.db')
	return db

#Database query function to return raw data from database
def query_db(query, args=(), one=False):
	cur = get_db().execute(query, args)
	rv = cur.fetchall()
	cur.close()
	return (rv[0] if rv else None) if one else rv



# provide login manager with load_user callback
@lm.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Logout user
@app.route('/logout.html')
def logout():
    logout_user()
    return redirect(url_for('index'))
    

 
@app.route('/explore')
@login_required
def explore():
    files = Files.query.all()
    return render_template('pages/upload.html', title='Explore', files=files)
                           

   

@app.route('/edit_profile.html', methods=['GET', 'POST'])
@login_required
def edit_profile():
    form = EditProfileForm()
    if form.validate_on_submit():
        current_user.user = form.name.data
        current_user.about_me = form.about_me.data
        current_user.localisation = form.localisation.data
        current_user.job = form.job.data
        db.session.add(current_user)
        db.session.commit()
        flash('Your profile has been updated.')
        return redirect(url_for('index', user=current_user.user))
    form.name.data = current_user.user
    form.about_me.data = current_user.about_me
    form.localisation.data = current_user.localisation
    form.job.data = current_user.job
    return render_template('pages/edit_profile.html', form=form)
    

    

# Register a new user
@app.route('/register.html', methods=['GET', 'POST'])
def register():
    
    # skip the registration for authenticated users
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    # declare the Registration Form
    form = RegisterForm(request.form)

    msg = None

    if request.method == 'GET': 

        return render_template( 'pages/register.html', form=form, msg=msg )

    # check if both http method is POST and form is valid on submit
    if form.validate_on_submit():

        # assign form data to variables
        username = request.form.get('username', '', type=str)
        password = request.form.get('password', '', type=str) 
        email    = request.form.get('email'   , '', type=str) 

        # filter User out of database through username
        user = User.query.filter_by(user=username).first()

        # filter User out of database through username
        user_by_email = User.query.filter_by(email=email).first()

        if user or user_by_email:
            msg = 'Error: User exists!'
        
        else:         

            pw_hash = password #bc.generate_password_hash(password)

            user = User(username, email, pw_hash)

            user.save()

            msg = 'User created, please <a href="' + url_for('login') + '">login</a>'     

    else:
        msg = 'Input error'     

    return render_template( 'pages/register.html', form=form, msg=msg )

# Authenticate user
@app.route('/login.html', methods=['GET', 'POST'])
def login():
    
    # skip the login for authenticated users
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    # Declare the login form
    form = LoginForm(request.form)

    # Flask message injected into the page, in case of any errors
    msg = None

    # check if both http method is POST and form is valid on submit
    if form.validate_on_submit():

        # assign form data to variables
        username = request.form.get('username', '', type=str)
        password = request.form.get('password', '', type=str) 

        # filter User out of database through username
        user = User.query.filter_by(user=username).first()

        if user:
            
            #if bc.check_password_hash(user.password, password):
            if user.password == password:
                login_user(user)
                return redirect(url_for('index'))
            else:
                msg = "Wrong password. Please try again."
        else:
            msg = "Unkkown user"

    return render_template( 'pages/login.html', form=form, msg=msg )

# revenue

# App main route + generic routing
@app.route('/', defaults={'path': 'index.html'})

@app.route('/<path>')
def index(path):

    if not current_user.is_authenticated:
        return redirect(url_for('login'))

    content = None

    #try:

        # try to match the pages defined in -> pages/<input file>
    return render_template( 'pages/'+path )
    
    #except:
        
       # return render_template( 'pages/error-404.html' )

# Return sitemap 
import pandas as pd
import pyodbc
import numpy as np
import os

##MODEL PREDICTION
class display(object):
    """Display HTML representation of multiple objects"""
    template = """<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>{0}</p>{1}
    </div>"""

    def __init__(self, *args):
        self.args = args

    def _repr_html_(self):
        return '\n'.join(self.template.format(a, eval(a)._repr_html_())
                         for a in self.args)

    def __repr__(self):
        return '\n\n'.join(a + '\n' + repr(eval(a))
                           for a in self.args)




@app.route('/confusion', methods=['GET','POST'])
def confusion():
    # !/usr/bin/env python
    # coding: utf-8

    # In[1]:

    import pandas as pd
    import pyodbc
    import numpy as np
    import os



    # mycursor.execute("CREATE TABLE adherantchurn as SELECT id,statutsEnrolementCaAvp,suspensionImAvp,typePIDaAvp FROM adherentavpsfm ")
    sql_querry2 = 'SELECT * FROM adherantchurn'

    adherantt = pd.read_sql(sql_querry2, con=mydb)
    adherantt.head()

    # In[4]:

    # mycursor.execute("CREATE TABLE classchurn as SELECT adherent_avp_id,sexeDaAvp,domaineActiviteDaAvp,dateNaissanceDaAvp FROM adherentcomplementavpsfm ")
    sql_queryyy = 'SELECT * FROM adherantcomplechurn'
    dfff = pd.read_sql(sql_queryyy, con=mydb)
    dfff.head()

    # In[6]:

    # mycursor.execute("CREATE TABLE classchurn as SELECT montantCotisationCaAvp,adherent_avp_id,nomClasseAvp FROM classeavpsfm ")

    sql_query1 = ' SELECT * FROM classchurn'
    classe = pd.read_sql(sql_query1, con=mydb)
    classe.head()

    # In[24]:

    # mycursor.execute("DROP TABLE paimentchurn")
    # mycursor.execute("CREATE TABLE paimentchurn as SELECT id, adherent_avp_id,montant_global,sourceImAvp FROM paiementavpsfm ")
    sql_query3 = ' SELECT * FROM paimentchurn'
    paiment = pd.read_sql(sql_query3, con=mydb)
    paiment.head()

    # In[25]:

    # mycursor.execute("CREATE TABLE bureauchurn as SELECT adresse,label,id FROM bureauavpsfm ")
    ql_query = 'SELECT * FROM bureauchurn'
    bureau = pd.read_sql(ql_query, con=mydb)
    bureau.head()

    # In[26]:

    # mycursor.execute("CREATE TABLE adressechurn as SELECT adherent_avp_id,villeDaAvp,discritDaAvp,communeDaAvp FROM adresseavpsfm ")
    sql_query = 'SELECT * FROM adressechurn'
    adresse = pd.read_sql(sql_query, con=mydb)
    adresse.head()

    # In[27]:

    # mycursor.execute("CREATE TABLE cotisationchurn as SELECT comptabilisation_avp_id,montantPayeImAvp,paiement_avp_id FROM cotisationavpsfm ")
    sql_query4 = ' SELECT * FROM cotisationchurn'
    cotisation = pd.read_sql(sql_query4, con=mydb)
    cotisation.head()

    # In[46]:

    # mycursor.execute("DROP TABLE comptabilischurn ")
    # mycursor.execute("CREATE TABLE comptabilischurn as SELECT classe_comptabilite_avp_id,adherent_avp_id,montant_cotisation, trimestre, solde, compteDebit, compteCredit FROM comptabilisationavpsfm ")
    sql_query5 = ' SELECT * FROM comptabilischurn'
    comptabilis = pd.read_sql(sql_query5, con=mydb)
    comptabilis.head()

    # In[47]:

    df = display('dfff', 'adherantt', 'pd.merge(dfff, adherantt, left_on="adherent_avp_id", right_on="id")')
    df

    # In[48]:

    data = pd.merge(dfff, adherantt, left_on="adherent_avp_id", right_on="id").drop('id', axis=1)
    data.head()

    # In[49]:

    df = display('data', 'bureau', 'pd.merge(data, bureau, left_on="adherent_avp_id", right_on="id")')
    df

    # In[50]:

    data1 = pd.merge(data, bureau, left_on="adherent_avp_id", right_on="id").drop('id', axis=1)
    data1.head()

    # In[51]:

    dfm = display('data1', 'classe', 'pd.merge(data1, classe, left_on="adherent_avp_id", right_on="adherent_avp_id")')
    dfm

    # In[52]:

    data2 = pd.merge(data1, classe, left_on="adherent_avp_id", right_on="adherent_avp_id")
    data2.head()

    # In[53]:

    dfm = display('cotisation', 'paiment', 'pd.merge(cotisation, paiment, left_on="paiement_avp_id", right_on="id")')
    dfm

    # In[54]:

    data3 = pd.merge(cotisation, paiment, left_on="paiement_avp_id", right_on="id").drop('paiement_avp_id', axis=1)
    data3.head()

    # In[55]:

    daa4 = display('data2', 'data3', 'pd.merge(data2, data3, left_on="adherent_avp_id", right_on="adherent_avp_id")')
    daa4

    # In[56]:

    data4 = pd.merge(data2, data3, left_on="adherent_avp_id", right_on="adherent_avp_id")
    data4.head()

    # In[57]:

    ddfm = display('data4', 'comptabilis',
                   'pd.merge(data3, comptabilis, left_on="adherent_avp_id", right_on="adherent_avp_id")')
    ddfm

    # In[58]:

    data5 = pd.merge(data4, comptabilis, left_on="adherent_avp_id", right_on="adherent_avp_id")
    data5.head()

    # In[59]:

    data5['Age'] = data5['dateNaissanceDaAvp']

    # In[60]:

    # churn_df_final = data5[['montantPayeImAvp','domaineActiviteDaAvp','suspensionImAvp','typePIDaAvp','adresse','label','montant_global','montantCotisationCaAvp','sourceImAvp','trimestre','compteDebit','compteCredit','Age']]
    # churn_df_final= churn_df_final.reset_index(drop=True)

    # In[61]:

    churn_df_final = data5[
        ['montantPayeImAvp', 'suspensionImAvp', 'montant_global', 'montantCotisationCaAvp', 'trimestre']]
    churn_df_final = churn_df_final.reset_index(drop=True)

    # In[62]:

    churn_df_final

    # In[89]:

    churn_df_final['trimestre'] = pd.factorize(churn_df_final['trimestre'])[0]

    # In[64]:

    # churn_df_final['trimestre']=pd.factorize(churn_df_final['trimestre'])[0]
    # churn_df_final['sourceImAvp']=pd.factorize(churn_df_final['sourceImAvp'])[0]
    # churn_df_final['label']=pd.factorize(churn_df_final['label'])[0]
    # churn_df_final['adresse']=pd.factorize(churn_df_final['adresse'])[0]
    # churn_df_final['domaineActiviteDaAvp']=pd.factorize(churn_df_final['domaineActiviteDaAvp'])[0]
    # churn_df_final['typePIDaAvp']=pd.factorize(churn_df_final['typePIDaAvp'])[0]

    # churn_df_final

    # In[65]:

    churn_df_final['trimestre'] = churn_df_final['trimestre'].astype('int')

    # In[66]:

    # churn_df_final['trimestre'] = churn_df_final['trimestre'].astype('int')
    # churn_df_final['domaineActiviteDaAvp'] = churn_df_final['domaineActiviteDaAvp'].astype('int')
    # churn_df_final['adresse'] = churn_df_final['adresse'].astype('int')
    # churn_df_final['label'] = churn_df_final['label'].astype('int')
    # churn_df_final['sourceImAvp'] = churn_df_final['sourceImAvp'].astype('int')
    # churn_df_final['typePIDaAvp'] = churn_df_final['typePIDaAvp'].astype('int')

    # In[67]:

    X = np.asarray(churn_df_final[['montantPayeImAvp', 'montantCotisationCaAvp', 'montant_global', 'trimestre']])
    X[0:5]

    # In[68]:

    # X = np.asarray(churn_df_final[['montantPayeImAvp','domaineActiviteDaAvp','montantCotisationCaAvp','typePIDaAvp','adresse','label','montant_global','sourceImAvp','trimestre','compteDebit','compteCredit','Age']])
    # X[0:5]

    # In[69]:

    y = np.asarray(churn_df_final['suspensionImAvp'])
    y[0:5]

    # In[72]:

    from sklearn import preprocessing
    X = preprocessing.StandardScaler().fit(X).transform(X)
    X

    # In[73]:

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    print('Train set:', X_train.shape, y_train.shape)
    print('Test set:', X_test.shape, y_test.shape)

    # In[74]:

    # Import Random Forest Model
    from sklearn.ensemble import RandomForestClassifier

    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=150, n_jobs=2)

    # In[75]:

    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # In[76]:

    yhat = y_pred

    # In[77]:

    # Import scikit-learn metrics module for accuracy calculation
    from sklearn import metrics
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", (metrics.accuracy_score(y_test, yhat)) * 100, '%')

    # In[79]:

    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix
    import itertools
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        labels = 'Exited', 'Retained'
        sizes = [data5.suspensionImAvp[data5['suspensionImAvp'] == 1].count(),
                 data5.suspensionImAvp[data5['suspensionImAvp'] == 0].count()]
        explode = (0, 0.1)
        fig1, (ax1,ax2) = plt.subplots(2)
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')


        a=plt.imshow(cm, interpolation='nearest', cmap=cmap)
        #a.figure.show()

        plt.title("Proportion of customer churned and retained", size=20)
        plt.title("Confusion matrix", size=20)

        #plt.show()

        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        # fig3.figure.savefig("aaa.png")
        output = io.BytesIO()
        FigureCanvasSVG(fig1).print_svg(output)
        return Response(output.getvalue(), mimetype="image/svg+xml")

    print(confusion_matrix(y_test, yhat, labels=[1, 0]))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, yhat, labels=[1, 0])
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    #plt.figure()


    # In[66]:

    import itertools
    import numpy as np

    from matplotlib.ticker import NullFormatter
    import pandas as pd
    import numpy as np
    import matplotlib.ticker as ticker
    from sklearn import preprocessing

    # In[80]:
    plot_confusion_matrix(cnf_matrix, classes=['churn=1', 'churn=0'], normalize=False, title='Confusion matrix')

    labels = 'Exited', 'Retained'
    sizes = [data5.suspensionImAvp[data5['suspensionImAvp'] == 1].count(),
             data5.suspensionImAvp[data5['suspensionImAvp'] == 0].count()]
    explode = (0, 0.1)
    plt.figure()
    fig, ax1 = plt.subplots(figsize=(9, 9))
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')
    plt.title("Proportion of customer churned and retained", size=20)




    #plt.show()
    #fig.savefig("C:/Users/asus/Desktop/SFM/WEB APP/app/static/Hello.png")
    output = io.BytesIO()
    FigureCanvasSVG(fig).print_svg(output)
    return Response(output.getvalue(), mimetype="image/svg+xml")



@app.route('/conf', methods=['GET','POST'])
def conf():
    # !/usr/bin/env python
    # coding: utf-8

    # In[1]:

    import pandas as pd
    import pyodbc
    import numpy as np
    import os



    # mycursor.execute("CREATE TABLE adherantchurn as SELECT id,statutsEnrolementCaAvp,suspensionImAvp,typePIDaAvp FROM adherentavpsfm ")
    sql_querry2 = 'SELECT * FROM adherantchurn'

    adherantt = pd.read_sql(sql_querry2, con=mydb)
    adherantt.head()

    # In[4]:

    # mycursor.execute("CREATE TABLE classchurn as SELECT adherent_avp_id,sexeDaAvp,domaineActiviteDaAvp,dateNaissanceDaAvp FROM adherentcomplementavpsfm ")
    sql_queryyy = 'SELECT * FROM adherantcomplechurn'
    dfff = pd.read_sql(sql_queryyy, con=mydb)
    dfff.head()

    # In[6]:

    # mycursor.execute("CREATE TABLE classchurn as SELECT montantCotisationCaAvp,adherent_avp_id,nomClasseAvp FROM classeavpsfm ")

    sql_query1 = ' SELECT * FROM classchurn'
    classe = pd.read_sql(sql_query1, con=mydb)
    classe.head()

    # In[24]:

    # mycursor.execute("DROP TABLE paimentchurn")
    # mycursor.execute("CREATE TABLE paimentchurn as SELECT id, adherent_avp_id,montant_global,sourceImAvp FROM paiementavpsfm ")
    sql_query3 = ' SELECT * FROM paimentchurn'
    paiment = pd.read_sql(sql_query3, con=mydb)
    paiment.head()

    # In[25]:

    # mycursor.execute("CREATE TABLE bureauchurn as SELECT adresse,label,id FROM bureauavpsfm ")
    ql_query = 'SELECT * FROM bureauchurn'
    bureau = pd.read_sql(ql_query, con=mydb)
    bureau.head()

    # In[26]:

    # mycursor.execute("CREATE TABLE adressechurn as SELECT adherent_avp_id,villeDaAvp,discritDaAvp,communeDaAvp FROM adresseavpsfm ")
    sql_query = 'SELECT * FROM adressechurn'
    adresse = pd.read_sql(sql_query, con=mydb)
    adresse.head()

    # In[27]:

    # mycursor.execute("CREATE TABLE cotisationchurn as SELECT comptabilisation_avp_id,montantPayeImAvp,paiement_avp_id FROM cotisationavpsfm ")
    sql_query4 = ' SELECT * FROM cotisationchurn'
    cotisation = pd.read_sql(sql_query4, con=mydb)
    cotisation.head()

    # In[46]:

    # mycursor.execute("DROP TABLE comptabilischurn ")
    # mycursor.execute("CREATE TABLE comptabilischurn as SELECT classe_comptabilite_avp_id,adherent_avp_id,montant_cotisation, trimestre, solde, compteDebit, compteCredit FROM comptabilisationavpsfm ")
    sql_query5 = ' SELECT * FROM comptabilischurn'
    comptabilis = pd.read_sql(sql_query5, con=mydb)
    comptabilis.head()

    # In[47]:

    df = display('dfff', 'adherantt', 'pd.merge(dfff, adherantt, left_on="adherent_avp_id", right_on="id")')
    df

    # In[48]:

    data = pd.merge(dfff, adherantt, left_on="adherent_avp_id", right_on="id").drop('id', axis=1)
    data.head()

    # In[49]:

    df = display('data', 'bureau', 'pd.merge(data, bureau, left_on="adherent_avp_id", right_on="id")')
    df

    # In[50]:

    data1 = pd.merge(data, bureau, left_on="adherent_avp_id", right_on="id").drop('id', axis=1)
    data1.head()

    # In[51]:

    dfm = display('data1', 'classe', 'pd.merge(data1, classe, left_on="adherent_avp_id", right_on="adherent_avp_id")')
    dfm

    # In[52]:

    data2 = pd.merge(data1, classe, left_on="adherent_avp_id", right_on="adherent_avp_id")
    data2.head()

    # In[53]:

    dfm = display('cotisation', 'paiment', 'pd.merge(cotisation, paiment, left_on="paiement_avp_id", right_on="id")')
    dfm

    # In[54]:

    data3 = pd.merge(cotisation, paiment, left_on="paiement_avp_id", right_on="id").drop('paiement_avp_id', axis=1)
    data3.head()

    # In[55]:

    daa4 = display('data2', 'data3', 'pd.merge(data2, data3, left_on="adherent_avp_id", right_on="adherent_avp_id")')
    daa4

    # In[56]:

    data4 = pd.merge(data2, data3, left_on="adherent_avp_id", right_on="adherent_avp_id")
    data4.head()

    # In[57]:

    ddfm = display('data4', 'comptabilis',
                   'pd.merge(data3, comptabilis, left_on="adherent_avp_id", right_on="adherent_avp_id")')
    ddfm

    # In[58]:

    data5 = pd.merge(data4, comptabilis, left_on="adherent_avp_id", right_on="adherent_avp_id")
    data5.head()

    # In[59]:

    data5['Age'] = data5['dateNaissanceDaAvp']

    # In[60]:

    # churn_df_final = data5[['montantPayeImAvp','domaineActiviteDaAvp','suspensionImAvp','typePIDaAvp','adresse','label','montant_global','montantCotisationCaAvp','sourceImAvp','trimestre','compteDebit','compteCredit','Age']]
    # churn_df_final= churn_df_final.reset_index(drop=True)

    # In[61]:

    churn_df_final = data5[
        ['montantPayeImAvp', 'suspensionImAvp', 'montant_global', 'montantCotisationCaAvp', 'trimestre']]
    churn_df_final = churn_df_final.reset_index(drop=True)

    # In[62]:

    churn_df_final

    # In[89]:

    churn_df_final['trimestre'] = pd.factorize(churn_df_final['trimestre'])[0]

    # In[64]:

    # churn_df_final['trimestre']=pd.factorize(churn_df_final['trimestre'])[0]
    # churn_df_final['sourceImAvp']=pd.factorize(churn_df_final['sourceImAvp'])[0]
    # churn_df_final['label']=pd.factorize(churn_df_final['label'])[0]
    # churn_df_final['adresse']=pd.factorize(churn_df_final['adresse'])[0]
    # churn_df_final['domaineActiviteDaAvp']=pd.factorize(churn_df_final['domaineActiviteDaAvp'])[0]
    # churn_df_final['typePIDaAvp']=pd.factorize(churn_df_final['typePIDaAvp'])[0]

    # churn_df_final

    # In[65]:

    churn_df_final['trimestre'] = churn_df_final['trimestre'].astype('int')

    # In[66]:

    # churn_df_final['trimestre'] = churn_df_final['trimestre'].astype('int')
    # churn_df_final['domaineActiviteDaAvp'] = churn_df_final['domaineActiviteDaAvp'].astype('int')
    # churn_df_final['adresse'] = churn_df_final['adresse'].astype('int')
    # churn_df_final['label'] = churn_df_final['label'].astype('int')
    # churn_df_final['sourceImAvp'] = churn_df_final['sourceImAvp'].astype('int')
    # churn_df_final['typePIDaAvp'] = churn_df_final['typePIDaAvp'].astype('int')

    # In[67]:

    X = np.asarray(churn_df_final[['montantPayeImAvp', 'montantCotisationCaAvp', 'montant_global', 'trimestre']])
    X[0:5]

    # In[68]:

    # X = np.asarray(churn_df_final[['montantPayeImAvp','domaineActiviteDaAvp','montantCotisationCaAvp','typePIDaAvp','adresse','label','montant_global','sourceImAvp','trimestre','compteDebit','compteCredit','Age']])
    # X[0:5]

    # In[69]:

    y = np.asarray(churn_df_final['suspensionImAvp'])
    y[0:5]

    # In[72]:

    from sklearn import preprocessing
    X = preprocessing.StandardScaler().fit(X).transform(X)
    X

    # In[73]:

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    print('Train set:', X_train.shape, y_train.shape)
    print('Test set:', X_test.shape, y_test.shape)

    # In[74]:

    # Import Random Forest Model
    from sklearn.ensemble import RandomForestClassifier

    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=150, n_jobs=2)

    # In[75]:

    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # In[76]:

    yhat = y_pred

    # In[77]:

    # Import scikit-learn metrics module for accuracy calculation
    from sklearn import metrics
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", (metrics.accuracy_score(y_test, yhat)) * 100, '%')

    # In[79]:

    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix
    import itertools
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)

        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    print(confusion_matrix(y_test, yhat, labels=[1, 0]))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, yhat, labels=[1, 0])
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    fig = plt.figure(figsize=(5, 5))

    plot_confusion_matrix(cnf_matrix, classes=['Bad Clients', 'Good Clients'], normalize=False, title='Confusion matrix')


    output = io.BytesIO()
    FigureCanvasSVG(fig).print_svg(output)
    return Response(output.getvalue(), mimetype="image/svg+xml")



####INPUT STATISTICS

def revenue():

    # connecting to the mysql server
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password=""
    )
    print(a)
    print("hello")
    print(mydb)
    mycursor = mydb.cursor(buffered=True)
    mycursor.execute("USE bd_avplus")
    a = mycursor.execute("SELECT SUM(montantPayeImAvp) FROM cotisationavpsfm")
    myList = mycursor.fetchall()
    revenue =str(myList[0][0])
    return revenue

def nbrclients():
    # connecting to the mysql server
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password=""
    )
    print("hello")
    print(mydb)
    mycursor = mydb.cursor(buffered=True)
    mycursor.execute("USE bd_avplus")
    a = mycursor.execute("SELECT COUNT(*) FROM adherentavpsfm")
    myList = mycursor.fetchall()
    nbr=str(myList[0][0])
    return nbr


@app.route('/inputstatistics')
def input():
    rev = revenue()
    nbr = nbrclients()
    return render_template('pages/inputstatistique.html', value=rev, nbr=nbr)

@app.route('/client')
def situationcilents():
    # connecting to the mysql server
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password=""
    )
    mycursor = mydb.cursor(buffered=True)
    mycursor.execute("USE bd_avplus")
    print(mydb)
    df_enrollement = pd.read_sql(
        'SELECT `statutsEnrolementCaAvp` , COUNT(*) totalCount FROM `adherentavpsfm` GROUP BY `statutsEnrolementCaAvp`',
        con=mydb)

    df_enrollement = df_enrollement.loc[df_enrollement['statutsEnrolementCaAvp'] != 'BloquÃ©']
    df_enrollement = df_enrollement.loc[df_enrollement['statutsEnrolementCaAvp'] != 'BloquÃ©_Responsable']
    df_enrollement = df_enrollement.loc[df_enrollement['statutsEnrolementCaAvp'] != 'Correction Blocage']
    df_enrollement = df_enrollement.loc[df_enrollement['statutsEnrolementCaAvp'] != 'RefusÃ©']
    df_enrollement = df_enrollement.loc[df_enrollement['statutsEnrolementCaAvp'] != 'Dossier rÃ©vu']

    df_enrollement['statutsEnrolementCaAvp'] = df_enrollement['statutsEnrolementCaAvp'].replace(
        ['AcceptÃ©', 'AffiliÃ©', 'ValidÃ©'], ['Accepté', 'Affilié', 'Validé'])
    # df_enrollement.set_index('statutsEnrolementCaAvp')
    #df_enrollement.head(10)
    fig = df_enrollement.plot.bar(x='statutsEnrolementCaAvp', y='totalCount', rot=70,
                            title="Clients situation");
   # fig.figure.savefig("C:\Users\ asus\Desktop\SFM\WEB APP\ app\static\images")
    return "hello"

@app.route("/modflow/x.svg")
def plot_svg():
    """ renders the plot on the fly.
    """
    fig = plt.figure(figsize=(5, 5))
    axis = fig.add_subplot(1, 1, 1)
    y = [i for i in range(40) ]
    axis.plot(y)

    output = io.BytesIO()
    FigureCanvasSVG(fig).print_svg(output)
    return Response(output.getvalue(), mimetype="image/svg+xml")


@app.route('/sitemap.xml')
def sitemap():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'sitemap.xml')
    
#Upload get and post method to save files into directory
@app.route("/upload",methods=['GET','POST'])
def upload():
	if request.method == 'GET':
			return render_template('pages/upload.html')
	elif request.method == 'POST':
			file = request.files['file']
			filename = file.filename
			directory=os.path.join(app.config['UPLOAD_FOLDER'], filename)
			file.save(directory)
			size=os.path.getsize(directory)
			filehash=hashlib.sha1(directory.encode()).hexdigest()
			if (db_insert(filename,size,filehash,current_user.id)):
						#change filename to hash
					os.rename(directory,os.path.join(app.config['UPLOAD_FOLDER'], filehash))
                    #flash('File saved succesfully', 'succes')  
					return render_template( 'pages/upload.html')
                        
			#except:
			 #       return 'File not Found'
            
            #except:
            #return render_template( 'pages/error-404.html' )
 

@app.route("/download/<filehash>",methods=['GET'])
def download(filehash):
		#filehash is sha1 hash stored in database of file.Increase download counter
		data=query_db('select * from files where hash=?',[filehash])
		counter=int(data[0][5])+1
		try:
			get_db().execute("update files SET counter = ? WHERE hash=?", [counter,filehash])
			get_db().commit()
			return send_from_directory(app.config['UPLOAD_FOLDER'], data[0][3])
			#return send_file(os.path.join(app.config['UPLOAD_FOLDER'], data[0][3]),attachment_filename=data[0][1],as_attachment=True)
		except:
			return 'File not Found'

@app.route("/server-usage",methods=['GET'])
def server_usage():
	data=query_db('select * from files')
	bandwidth=0
	for i in data:
		bandwidth+=int(i[5])*int(i[2]) #Multiplying counter with size of file to get bandwidth amount
	return jsonify(bandwidthusage=str(bandwidth/1024.0)+" KB")

@app.route("/disk-usage",methods=['GET'])
def disk_usage():
	data=query_db('select * from files')
	diskspace=0
	for i in data:
		diskspace+=int(i[2])
	return jsonify(diskusage=str(diskspace/50.0)+" MB") 
    
#Its a simple function just return number of files link should be /db for application
@app.route("/db")
def db_table():
	data=query_db('select * from files')
	return jsonify(values=data)
	#for user in data:
    #	return user['filename'], user['size']

def db_insert(filename,size,filehash,user_id):
    filename=str(filename)
    size=int(size)
    filedate=str(date.today())
    file_exist=query_db('select * from files where hash=?',[filehash])
    if not file_exist:
        get_db().execute("insert into files (filename,size,hash,date,counter,user_id) values (?,?,?,?,?,?)", [filename,size,filehash,filedate,0,user_id])
        get_db().commit()
        return True


import io
import random
from flask import Response, request
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_svg import FigureCanvasSVG

#Implementing the uzf script and creating multiple endpoints from it : 
import os
import sys
import glob
import platform
import shutil
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd


#@app.route("/av/<string:av>", methods=['GET', 'POST'])
#def test(av="hello"):
#    if av == "test":
#       return "hello"
#    return "hello"

import mysql.connector
# HERE STARTS SFM
@app.route('/av/<string:av>', methods=['GET', 'POST'])
def connect(av="test"):

    if av == 'read':
        sql_file = open("bd_avplus.app1.12082020-2359.sql", "r").read()

    elif av == 'query':
        mycursor = mydb.cursor(buffered=True)
        mycursor.execute("USE bd_avplus")
        print("aha")
        df_total = pd.read_sql('SELECT * FROM adherentavpsfm JOIN  adherentcomplementavpsfm ON adherentavpsfm.id =adherentcomplementavpsfm.adherent_avp_id join adresseavpsfm ON adherentcomplementavpsfm.adherent_avp_id = adresseavpsfm.adherent_avp_id',con=mydb)
        print("aha2")
        print(df_total.head())
        df_total = df_total[df_total.typeDateNaissance == 'Date de naissance']
        # new dataframe containing the relevant columns
        adherent = df_total[
            ['adherent_avp_id', 'sexeDaAvp', 'typeDateNaissance', 'dateNaissanceDaAvp', 'numeroPIDaAvp', 'nbrenfant',
             'numeroCarteCaAvp', 'dateAffiliationCaAvp', 'dateDernierRelanceCaAvp', 'paysDaAvp', 'villeDaAvp',
             'regionDaAvp']].copy()
        # drop redundant columns
        adherent = adherent.loc[:, ~adherent.columns.duplicated()]

        return "hello"

    return "hello"


# total revenue
@app.route('/revenue', methods=['GET', 'POST'])



# suspension
@app.route('/suspension', methods=['GET', 'POST'])
def suspension():
    print("testtttt")
    # connecting to the mysql server
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password=""
    )
    print("hello")
    print(mydb)
    mycursor = mydb.cursor(buffered=True)
    mycursor.execute("USE bd_avplus")
    b = mycursor.execute("SELECT SUM(1) FROM adherentavpsfm WHERE suspensionImAvp=0")
    mylist = mycursor.fetchall()
    return str(mylist[0][0])

def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
@app.teardown_appcontext
def close_connection(exception):
	db = getattr(g, '_database', None)
	if db is not None:
		db.close()
    
