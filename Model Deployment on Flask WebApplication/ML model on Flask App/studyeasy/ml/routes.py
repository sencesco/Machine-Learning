import pickle, os
import numpy as np
from studyeasy import app
from flask import render_template, redirect, url_for, Blueprint
from studyeasy.ml.forms import RainPridictionForm
from sklearn.preprocessing import MinMaxScaler

ml = Blueprint('ml', __name__)

@ml.route('/rain_prediction', methods=('GET', 'POST'))
def rain_prediction():
    form = RainPridictionForm()
    if form.validate_on_submit():
        
        rainfall = form.rainfall.data
        cloud3pm = form.cloud3pm.data
        cloud9am = form.cloud9am.data
        humidity3pm = form.humidity3pm.data
        rainToday = form.rainToday.data
        
        # For convert string to numeric to input to machine learning.
        if rainToday == 'Yes':
            rainToday = 1
        else:
            rainToday = 0
            
        # For convert to numpy array to input to machinf learning. 
        data = np.array([[rainfall, cloud3pm, cloud9am, humidity3pm, rainToday ]])
        
        # Acces machine learning model path.
        _path_model = os.path.join(app.root_path, 'static/ML/model_xgb')
        # Call saved ML model
        saved_model = pickle.load(open(_path_model, 'rb'))
        # Acces standardize object path
        _path_MMS = os.path.join(app.root_path, 'static/ML/MMS') # minmaxscalar object
        # Standardize the input data from web apllication.
        MMS  = pickle.load(open(_path_MMS, 'rb'))
        # transform input to ML.
        data = MMS.transform(data)
        # Predict a class label.
        result = saved_model.predict(data)

        if result[0] == 0:
            result = 'No rain tomorrow'
        else: 
            result = 'Possible rain '

        return render_template('ml/result.html', title='Predicted result', result=result)
    return render_template('ml/rain_prediction.html', title='Rain Prediction', form=form)
