import flask
# from joblib import load
# from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping

# Use joblib to load in the StandardScaler.
# sc = load('model/scaler.joblib') #- disabled due to heroku versioning issue

#Workaround to manually create scaler using saved scaling attributes
def scale_data(array, means = np.load('model/means.npy'), stds = np.load('model/vars.npy')**0.5):
    return (array-means)/stds

# Load train neural network.
json_file = open('model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
deploymodel = model_from_json(loaded_model_json)
deploymodel.load_weights('model/deploymodelweights.h5')
deploymodel.compile(loss='mse', optimizer = 'adam')
app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        g = int(flask.request.form['g'])
        ppg = float(flask.request.form['ppg'])
        rpg = float(flask.request.form['rpg'])
        apg = float(flask.request.form['apg'])
        spg = float(flask.request.form['spg'])
        bpg = float(flask.request.form['bpg'])
        tov = float(flask.request.form['tov']) * g
        ws = float(flask.request.form['ws'])
        vorp = float(flask.request.form['vorp'])
        pts = g * ppg
        trb = g * rpg
        ast = g * apg
        stl = g * spg
        blk = g * bpg
        ppgx2 = ppg * ppg
        ppg_pts = ppg * pts
        input_variables = pd.DataFrame([[g,ppg,ppgx2, ppg_pts, trb, rpg, ast, apg, stl, spg, blk, bpg, tov, ws, vorp]],
                                       columns=['g','ppg','ppg^2', 'ppg pts', 'trb', 'rpg', 'ast', 'apg', 'stl', 'spg', 'blk', 'bpg', 'tov', 'ws', 'vorp'],
                                       dtype=float)
        input_variables_sc = scale_data(input_variables)
        prediction = deploymodel.predict(input_variables_sc)[0]
        prediction = int(prediction)
        if prediction < 1000000:
            prediction = str(prediction)
            predlength = len(prediction)
            commainsert = predlength - 3
            prediction = str("$" + prediction[0:commainsert] + "," + prediction[-3:])
        elif prediction < 1000000000:
            prediction = str(prediction)
            predlength = len(prediction)
            commainsert = predlength - 3
            commainsert2 = predlength - 6
            prediction = str("$" + prediction[0:commainsert2] + "," + prediction[-6:-3] + "," + prediction[-3:])
        else:
            prediction = str("This player likely does not exist in the NBA.")
        return flask.render_template('main.html',
                                     original_input={'Games Played':g,
                                                     'Points Per Game':ppg,
                                                     'Rebounds Per Game':rpg,
                                                     'Assists Per Game':apg,
                                                     'Steals Per Game':spg,
                                                     'Blocks Per Game':bpg,
                                                     'Turnovers Per Game':round(tov/g,1),
                                                     'Winshares':ws,
                                                     'Value Over Replacement Player':vorp,
                                                     },
                                     result=prediction,
                                     )
