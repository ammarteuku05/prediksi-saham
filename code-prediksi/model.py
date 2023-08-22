import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import mysql.connector
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
from sklearn.model_selection import KFold
from keras.models import load_model
import matplotlib.pyplot as plt
import os
from math import sqrt
from sklearn.ensemble import GradientBoostingRegressor

db_config = {
    'host': '127.0.0.1',
    'user': 'user',
    'password': 'password',
    'database': 'db'
}

def create_connection():
    connection = mysql.connector.connect(**db_config)
    return connection

def loadData(kodeSaham):
    try:
        # Create a connection to the database
        connection = create_connection()

        # Create a cursor to interact with the database
        cursor = connection.cursor()

        # Execute a SQL query to fetch data from a table (replace 'your_table_name' with your actual table name)
        query = "SELECT STR_TO_DATE(Tanggal, '%Y%m%d') AS Tanggal, Open_Price, Tertinggi, Terendah, Penutupan, Volume FROM saham WHERE Kode_Saham='" + kodeSaham + "'" + "ORDER BY Tanggal ASC"
        cursor.execute(query)

        # Fetch all the data from the table
        data = cursor.fetchall()

        # Close the cursor and connection
        cursor.close()
        connection.close()

        # Render the data in an HTML template (assuming you have an 'index.html' template)
        return data

    except Exception as e:
        return "Error: " + str(e)

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	dff = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(dff.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(dff.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def rnn_gru(x, y, xvalid, yvalid):
    mc = ModelCheckpoint('best_model.h5', monitor='val_mae', mode='min', verbose=2, save_best_only=True) 

    n_splits = 5

    rnn_model = Sequential()
    rnn_model.add(LSTM(64))
    rnn_model.add(Dense(1))

    gru_model = Sequential()
    gru_model.add(GRU(64, input_shape=(x.shape[1], 1)))
    gru_model.add(Dense(1))


    # Initialize ensemble model
    ensemble_model = Sequential()

    # Add RNN (LSTM) model to the ensemble
    
    ensemble_model.add(rnn_model)

    # Add GRU model to the ensemble
    ensemble_model.add(gru_model)

    # Compile the ensemble model
    ensemble_model.compile(loss='mean_squared_error', optimizer='adam', metrics='mae')
        # Perform ensemble stacking with k-fold cross-validation
    kf = KFold(n_splits=n_splits)
    i = 0
    for train_index, val_index in kf.split(x):
        X_trains, X_val = x[train_index], x[val_index]
        y_trains, y_val = y[train_index], y[val_index]
        i =i+1
        # Train the ensemble model
        ensemble_model.fit(X_trains, 
                        y_trains, 
                        epochs=300, 
                        validation_data=(xvalid, yvalid), 
                        verbose=2,
                        batch_size = 252, # inisiasi batch
                        callbacks=[mc]
                        )
    
    return ensemble_model

def ensemble_predict(ensemble_model, xtest):
    #Load Best MODEL
    ensemble_model.save('best_model.h5')
    Best_Model = load_model('best_model.h5')
    #lakukan prediksi
    print(xtest.shape)
    prediksi = Best_Model.predict(xtest)
    prediksiReshape = prediksi.reshape((prediksi.shape[0]))
    return prediksi, prediksiReshape
    

def gradient_boosting(prediksi, Y_Test):
    gb_model = GradientBoostingRegressor(n_estimators=100,learning_rate=0.1, random_state=42)

    #  Fit the model on the training data
    gb_model.fit(prediksi, Y_Test)

    #  Make predictions on the test data
    metaPrediksi = gb_model.predict(prediksi)
    MetaPrediksi = prediksi.reshape((prediksi.shape[0]))
    return metaPrediksi, MetaPrediksi, gb_model


# Fungsi mencari nilai Max()
def myMax(x):
  panjang = len(x)
  Vmax = 0 
  for i in x:
    if i > Vmax:
      Vmax = i
  
  return Vmax

# Fungsi mencari nilai Min()
def myMin(x):
  panjang = len(x)
  Vmin = x[0] 
  for i in x:
    if i < Vmin:
      Vmin = i
  
  return Vmin

# Fungsi Denormalisasi
def denormalize(x, min, max):
    final_value = x*(max - min) + min
    return round(final_value)



# MAE
def MAE(y_prediksi, y_aktual):
  sum = 0
  for i in range(len(y_prediksi)):
    sum = sum + abs(y_aktual[i]-y_prediksi[i])
  return (sum/len(y_prediksi))

# RMSE  
def RMSE(y_prediksi, y_aktual):
  sum = 0
  for i in range(len(y_prediksi)):
    sum = sum + (y_aktual[i]-y_prediksi[i])**2
  return sqrt(sum/len(y_prediksi))

def DA(y_prediksi, y_aktual):
  sum = 0
  for i in range(len(y_prediksi)-1):
    value = (y_aktual[i+1]-y_aktual[i])*(y_prediksi[i+1]-y_prediksi[i])
    if value >= 0 :
      sum = sum + 1
  hasil = (sum/(len(y_prediksi)-1))*100
  return hasil

# MAPE
def MAPE(y_prediksi, y_aktual):
  sum = 0
  for i in range(len(y_prediksi)):
    sum = sum + (abs(y_aktual[i] - y_prediksi[i])/y_aktual[i])
  hasil = (sum/len(y_prediksi))*100
  return hasil

def tanggaldownload():
    dates =datetime.now().date() - timedelta(days=1)
    dateStr = dates.strftime('%Y-%m-%d')
    cleaned_date = dateStr.replace('-', '')
    return cleaned_date

