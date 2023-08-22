import re
from flask import Flask, render_template, request, send_file
from numpy.lib import index_tricks
import os
import io
import json
import matplotlib.pyplot as plt
import mplfinance as fplt
import plotly
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from werkzeug.utils import send_file
from model import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

data=loadData('PTBA')

def dt_frame():
    df = pd.DataFrame(data, columns=['Tanggal', 'Open_Price', 'Tertinggi', 'Terendah', 'Penutupan', 'Volume'])

    df['Open_Price'] = df['Open_Price'].astype(float)
    df['Volume'] = df['Volume'].astype(float)
    df['Penutupan'] = df['Penutupan'].astype(float)
    df['Tertinggi'] = df['Tertinggi'].astype(float)
    df['Terendah'] = df['Terendah'].astype(float)
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%Y/%m/%d')
    df = df.sort_values(by='Tanggal')
    df = df.set_index('Tanggal')

    return df 

df = dt_frame()

def remove_volume():
    df.drop(['Volume'], axis=1, inplace=True)
    return df

data_frame = remove_volume()

Label = 1
def sliding_windows():
    df_reframed = series_to_supervised(data_frame,26,Label) 
    return df_reframed

df_reframed = sliding_windows()

def labeling():
    Non_Label = 1
    if Label == 1: Non_Label = -4
    if Label == 7: Non_Label = -28
    if Label == 30: Non_Label = -120 
    df_reframed.drop(df_reframed.iloc[ : , Non_Label:-1 ], axis=1, inplace=True) #kolom selain label di time step t remove
    return df_reframed

df_label = labeling()

def data_normalize():
    scaler = MinMaxScaler()
    v = df_reframed.values
    df_normalized = scaler.fit_transform(v)
    return df_normalized

normalize = data_normalize()

def data_test_train():
    dataset = normalize
    X = dataset[: , :-1]  # Dataset
    Y = dataset[: , -1]   # Label
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=True)
    X_validation, X_Test, Y_validation, Y_Test = train_test_split(X_test, Y_test, test_size=0.66, shuffle=True)

    #reshape jadi 3D
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_Test = X_Test.reshape((X_Test.shape[0], 1, X_Test.shape[1]))
    X_validation = X_validation.reshape((X_validation.shape[0], 1, X_validation.shape[1]))

    data_train= np.squeeze(X_train)
    data_test= np.squeeze(X_Test)
    return data_train, Y_train, data_test, Y_Test, X_validation, Y_validation

def data_test_train_to_predict():
    dataset = normalize
    X = dataset[: , :-1]  # Dataset
    Y = dataset[: , -1]   # Label
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=True)
    X_validation, X_Test, Y_validation, Y_Test = train_test_split(X_test, Y_test, test_size=0.66, shuffle=True)

    #reshape jadi 3D
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_Test = X_Test.reshape((X_Test.shape[0], 1, X_Test.shape[1]))
    X_validation = X_validation.reshape((X_validation.shape[0], 1, X_validation.shape[1]))
    print(X_train.shape, Y_train.shape, X_validation.shape, Y_validation.shape, X_Test.shape, Y_Test.shape) 

    return X_train, Y_train, X_Test, Y_Test, X_validation, Y_validation

xTrain, yTrain, xTest, yTest, xvalid, yvalid  = data_test_train()

xtp, ytp, xtestpredict, ytestpredict, xvalidpredict, yvalidpredict  = data_test_train_to_predict()

xt=pd.DataFrame(xTrain)
xtest=pd.DataFrame(xTest)

yt =pd.DataFrame(yTrain, columns=['Penutupan'])
ytest =pd.DataFrame(yTest, columns=['Penutupan'])

base_model = rnn_gru(xtp, ytp, xvalidpredict, yvalidpredict)

base, rehapePred = ensemble_predict(base_model, xtestpredict)

prediksi = pd.DataFrame(base, columns=['Hasil'])

meta_model, MetaPrediksi, gb_model = gradient_boosting(base, ytestpredict)

metaprediksi = pd.DataFrame(meta_model, columns=['Hasil'])

# # Cari Nilai Max() & Min()
minn = myMin(df_label.iloc[:,-1])
maxi = myMax(df_label.iloc[:,-1])

tp=pd.DataFrame(meta_model,columns=["trainpred"])
t=pd.DataFrame(ytestpredict,columns=["ytrain"])

do=pd.concat([tp,t],axis=1)

tprnn=pd.DataFrame(rehapePred,columns=["trainpred"])
trnn=pd.DataFrame(ytestpredict,columns=["ytrain"])

dornn=pd.concat([tprnn,trnn],axis=1)

# DENORMALISASI
denormalize_Prediksi = []
for m in MetaPrediksi:
   denormalize_Prediksi.append(denormalize(m,minn,maxi))

denormalize_aktual = []
for m in ytestpredict:
  denormalize_aktual.append(denormalize(m,minn,maxi))

if Label == 30: 
    DF_aktual = pd.read_excel('data_testing_30.xlsx')
    DF_rnn_gru = pd.read_excel('data_testing_rnn-gru_30.xlsx')
if Label == 7: 
    DF_aktual = pd.read_excel('data_testing_7.xlsx')
    DF_rnn_gru = pd.read_excel('data_testing_rnn-gru_7.xlsx')
if Label == 1: 
    DF_aktual = pd.read_excel('data_testing_1.xlsx')
    DF_rnn_gru = pd.read_excel('data_testing_rnn-gru_1.xlsx')

DF_Testing = DF_aktual

DF_Testing_rnn = DF_rnn_gru
 
mae = MAE(DF_aktual['Prediksi'],DF_aktual['Aktual']) 
rmse = RMSE(DF_aktual['Prediksi'],DF_aktual['Aktual']) 
da = DA(DF_aktual['Prediksi'],DF_aktual['Aktual'])
mape = MAPE(DF_aktual['Prediksi'],DF_aktual['Aktual'])

mae_rnn = MAE(DF_rnn_gru['Prediksi'],DF_rnn_gru['Aktual']) 
rmse_rnn = RMSE(DF_rnn_gru['Prediksi'],DF_rnn_gru['Aktual']) 
da_rnn = DA(DF_rnn_gru['Prediksi'],DF_rnn_gru['Aktual'])
mape_rnn = MAPE(DF_rnn_gru['Prediksi'],DF_rnn_gru['Aktual'])

def dt_frame_predict():
    df = pd.DataFrame(data, columns=['Tanggal', 'Open_Price', 'Tertinggi', 'Terendah', 'Penutupan', 'Volume'])

    df['Open_Price'] = df['Open_Price'].astype(float)
    df['Volume'] = df['Volume'].astype(float)
    df['Penutupan'] = df['Penutupan'].astype(float)
    df['Tertinggi'] = df['Tertinggi'].astype(float)
    df['Terendah'] = df['Terendah'].astype(float)
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%Y/%m/%d')
    df = df.sort_values(by='Tanggal')
    df = df.set_index('Tanggal')

    return df 

data_frame = dt_frame_predict()
      
@app.route("/")
def main():
    #close=json.dumps(close),date=json.dumps(date)
    return render_template('app.html', df=df.to_html(classes='table table-bordered table-striped table-hover'))

@app.route("/process_testing")
def process_testing():
    fig=px.line(dornn,y=['ytrain','trainpred'])
    gj = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    figg=px.line(do,y=['ytrain','trainpred'])
    gJSON = json.dumps(figg, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('process_testing.html',gj=gj, mae_rnn=mae_rnn, rmse_rnn=rmse_rnn, da_rnn=da_rnn, mape_rnn=mape_rnn,gJSON=gJSON, mae=mae, rmse=rmse, da=da, mape=mape)

# @app.route("/process_testing_rnngru")
# def process_testing_rnngru():
#     figg=px.line(do,y=['ytrain','trainpred'])
#     gJSON = json.dumps(figg, cls=plotly.utils.PlotlyJSONEncoder)
#     return render_template('process_testing_rnngru.html',gJSON=gJSON, mae_rnn=mae_rnn, rmse_rnn=rmse_rnn, da_rnn=da_rnn, mape_rnn=mape_rnn)

@app.route('/chart')
def chart():
    # fplt.plot(saham,type='candle',style='yahoo',savefig='static/assets/img/plot.png')
    figure=go.Figure(
    data= [
        go.Candlestick(
            #x=data['Date'],
            low=df['Terendah'],
            high=df['Tertinggi'],
            close=df['Penutupan'],
            open=df['Open_Price']
            # increasing_line_color='green',
            # decreasing_line_color='red'
        )
    ]
    )
    figure.update_layout(
    title= 'PTBA',
    yaxis_title='PTBA Stock Price IDR',
    xaxis_title='Tanggal'
    )
    graphJSON = json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('chart.html',graphJSON=graphJSON)

@app.route("/labeling")
def labeling():
    return render_template('sliding_window.html',  df_label=df_label.to_html(classes='table table-bordered table-striped table-hover'))

@app.route("/training")
def training():
    return render_template('training.html',xt=xt.to_html(classes='table table-bordered table-striped table-hover'),yt=yt.to_html(classes='table table-bordered table-striped table-hover'))

@app.route("/testing")
def testing():
    return render_template('testing.html',xtest=xtest.to_html(classes='table table-bordered table-striped table-hover'),ytest=ytest.to_html(classes='table table-bordered table-striped table-hover'))

@app.route("/base_model")
def base_model():
    return render_template('base_model.html',  prediksi=prediksi.to_html(classes='table table-bordered table-striped table-hover'))

@app.route("/denormalisasi")
def denormalisasi():
    return render_template('denormalisasi.html',  DF_Testing_rnn=DF_Testing_rnn.to_html(classes='table table-bordered table-striped table-hover'),DF_Testing=DF_Testing.to_html(classes='table table-bordered table-striped table-hover'))

@app.route("/meta_model")
def meta_model():
    return render_template('meta_model.html',  metaprediksi=metaprediksi.to_html(classes='table table-bordered table-striped table-hover'))
@app.route("/predict", methods=['GET'])
def predict():
    dateStr = tanggaldownload()
    data = pd.read_excel(r'/Users/teukumuhammadammar/Downloads/Ringkasan Saham-' + ''.join({dateStr}) + ''.join('.xlsx'))
    open=float(data['Open Price'])
    high=float(data['Tertinggi'])
    low=float(data['Terendah'])                                                        
    volume=float(data['Volume'])
    q=[open, high, low, volume]
    print("q = ", q)
    print("data =", q)
    q=pd.DataFrame(q).T
    q.columns=['Open_Price', 'Tertinggi', 'Terendah', 'Volume']
    d=(q-data_frame.min())/(data_frame.max()-data_frame.min())
    print("d = ",d)
    a=np.array(d.drop(columns=['Penutupan']))

    arr = a.flatten()

    # Reshape the array to have a single feature
    reshaped_array = arr.reshape(-1, 1)
    print("a  = ", a)
    test = gb_model.predict(reshaped_array)
    hasil= test*(data_frame['Penutupan'].max()-data_frame['Penutupan'].min())+data_frame['Penutupan'].min()
    mean = np.mean(hasil)
   
    hasil=int(mean.round())
    
    return render_template('predict.html',hasil=hasil, open=open, high=high, low=low, volume=volume)
if __name__ == "__main__":
    app.run(port=3000,debug=True)