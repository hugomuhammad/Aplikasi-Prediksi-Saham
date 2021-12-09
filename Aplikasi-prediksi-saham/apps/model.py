import streamlit as st
import tensorflow as tf
from apps import data as data
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential, save_model, load_model

def app():

    st.write('## Prediksi Harga Saham')

    #dataframe diambil dari data modul data.py
    dataframe = data.dataframe

    df = dataframe.copy() 

    st.write('Data saham periode ' + df.Date[0] + ' sampai ' + df.Date[len(df.Date)-1])  

    #membuat line chart
    st.line_chart(df[['Open', 'Close']], use_container_width=True)


    sc = StandardScaler()

    
    #fungsi hasil prediksi model
    def get_result(model, input, reference):
        obj = sc.fit(reference.reshape(-1,1))
        model = model
        prediction = model.predict(input)
        prediction = obj.inverse_transform(prediction)
        return prediction

        
    #Proses

    st.write(' ### Input harga open, high, low saham hari ini')
    
    #data untuk referensi inverse transform normalisasi
    X = df[['Open','High','Low']].values
    y = df['Close'].values
    
    #membuat model
    #file_model = build_model(X_train, y_train)
    model = data.model

    #standard scaler
    sc = StandardScaler()

    #Input data untuk prediksi
    data_open = st.number_input('Masukkan harga open hari ini', min_value=0, max_value=None, value=0)
    data_high = st.number_input('Masukkan harga high hari ini', min_value=0, max_value=None, value=0)
    data_low = st.number_input('Masukkan harga low hari ini', min_value=0, max_value=None, value=0)


    #menyesuaikan format data untuk diinput 
    data_input = np.array([[[data_open]],
                            [[data_high]],
                            [[data_low]]])

    #menormalisasi data
    c = X[int(len(X)*0.8):].tolist()
    input = [data_open, data_high, data_low]
    c.append(input)

    c = np.array(c)
    new_model = sc.fit_transform(c)
    data_input = new_model[-1]
    
    #melakukan prediksi
    if data_low != 0:
        hasil_prediksi = get_result(model, data_input.reshape(1,3,1), y[int(len(y)*0.8):])
    else:
        pass
    

    st.write(" ### Hasil Prediksi")

    #menampilkan hasil prediksi
    if data_low != 0:
        if hasil_prediksi[0][0] > y[len(y)-1]:


            st.write('Harga closed saham hari ini adalah Rp. ' + str(hasil_prediksi[0][0]))

            a_number = abs((hasil_prediksi[0][0] - y[len(y)-1])/y[len(y)-1])
            percentage = "{:.2%}".format(a_number)

            st.write('Harga closed saham hari ini naik sebesar ' + percentage)

            new_title = '<p style="font-family:sans-serif; color:Green; font-size: 20px;"> Naik </p>'
            st.write(new_title, unsafe_allow_html=True)


        else:
            
            st.write('Harga closed saham hari ini adalah Rp. ' + str(hasil_prediksi[0][0]))

            a_number = abs((hasil_prediksi[0][0] - y[len(y)-1])/y[len(y)-1])
            percentage = "{:.2%}".format(a_number)

            st.write('Harga closed saham hari ini turun sebesar ' + percentage )

            new_title = '<p style="font-family:sans-serif; color:Red; font-size: 20px;"> Turun </p>'
            st.write(new_title, unsafe_allow_html=True)
    
    else:
        st.write('Tidak ada data untuk prediksi')

