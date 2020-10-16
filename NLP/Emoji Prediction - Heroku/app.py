from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField
from wtforms.validators import NumberRange

import numpy as np  
from tensorflow.keras.models import load_model
import joblib

# import keras
import json
	

def sentences_to_indices(X, word_to_index, max_len):
    m = X.shape[0]                         
    X_indices = np.zeros((m,max_len))
    for i in range(m):                               
        sentence_words = [i.lower() for i in X[i].split()]
        j = 0
        for w in sentence_words:
            X_indices[i, j] = word_to_index[w]
            j = j+1
      
    
    return X_indices


app = Flask(__name__)
# Configure a secret SECRET_KEY
# We will later learn much better ways to do this!!
app.config['SECRET_KEY'] = 'someRandomKey'


# REMEMBER TO LOAD THE MODEL AND THE SCALER!
flower_model = load_model("my_model")


# Now create a WTForm Class
# Lots of fields available:
# http://wtforms.readthedocs.io/en/stable/fields.html
class FlowerForm(FlaskForm):
    sep_len = TextField('ENTER')

    submit = SubmitField('Analyze')



@app.route('/', methods=['GET', 'POST'])
def index():

    # Create instance of the form.
    form = FlowerForm()
    # If the form is valid on submission (we'll talk about validation next)
    if form.validate_on_submit():
        # Grab the data from the breed on the form.

        session['sep_len'] = form.sep_len.data

        return redirect(url_for("prediction"))


    return render_template('home.html', form=form)


@app.route('/prediction')
def prediction():

   
    content= np.array([session['sep_len']])

    with open('word_to_index.json') as f:
	    word_to_index = json.load(f)

    indices = sentences_to_indices(content, word_to_index, 10)
    results = str(np.argmax(flower_model.predict(indices)))

    return render_template('prediction.html',results=results)


if __name__ == '__main__':
    app.run(debug=True)