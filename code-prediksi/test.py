from model import *


test = tanggaldownload()

print(test)


# @app.route("/predict", methods=['GET','POST'])
# def predict():
#     if request.method =='POST':
#         open=float(request.form['Open_Price'])
#         high=float(request.form['Tertinggi'])
#         low=float(request.form['Terendah'])                                                        
#         volume=float(request.form['Volume'])
#         q=[open, high, low, volume]
#         q=pd.DataFrame(q).T
#         q.columns=['Open_Price', 'Tertinggi', 'Terendah', 'Volume']
#         d=(q-data_frame.min())/(data_frame.max()-data_frame.min())
#         a=np.array(d.drop(columns=['Penutupan']))
    
#         arr = a.flatten()
    
#         # Reshape the array to have a single feature
#         reshaped_array = arr.reshape(-1, 1)
#         print("a  = ", a)
#         test = gb_model.predict(reshaped_array)
#         hasil= test*(data_frame['Penutupan'].max()-data_frame['Penutupan'].min())+data_frame['Penutupan'].min()
#         mean = np.mean(hasil)
        
#         if volume==0 or high==0 or open==0 or low==0:
#             hasil='Pasar Saham Libur'
#         elif volume<0 or high<0 or open<0 or low<0:
#             hasil='Harga tidak mungkin bernilai negatif'
#         elif high<low:
#             hasil='High harus lebih besar dari low'
#         elif open<low:
#             hasil='Open harus lebih besar atau sama dengan low'
#         elif open>high:
#             hasil='Open harus lebih kecil atau sama dengan high'
#         else:
#             hasil=int(mean.round())
        
#         return render_template('predict.html',hasil=hasil)
#     return render_template('predict.html')
