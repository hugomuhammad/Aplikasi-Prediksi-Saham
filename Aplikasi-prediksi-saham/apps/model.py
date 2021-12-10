import streamlit as st
from apps import data as data
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

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
    def get_result(model, input):
        model = model
        prediction = model.predict(input)
        return prediction

        
    #Proses
    st.write(' ### Input harga open, high, low saham hari ini')
    
    #data untuk referensi inverse transform normalisasi
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
    data_input = np.array([[data_open, data_high, data_low]])
    
    #melakukan prediksi
    if data_low != 0:
        hasil_prediksi = get_result(model, data_input)
    else:
        pass
    

    st.write(" ### Hasil Prediksi")

    #menampilkan hasil prediksi
    if data_low != 0:
        if hasil_prediksi[0] > y[len(y)-1]:

            price = hasil_prediksi[0]
            price_format = '{:,.2f}'.format(price)

            st.write('Harga closed saham hari ini adalah Rp. ' + price_format)

            a_number = abs((hasil_prediksi[0] - y[len(y)-1])/y[len(y)-1])
            percentage = "{:.2%}".format(a_number)

            st.write('Harga closed saham hari ini naik sebesar ' + percentage + ' dari harga saham terakhir')

            new_title = '<p style="font-family:sans-serif; color:Green; font-size: 20px;"> Naik </p>'
            st.write(new_title, unsafe_allow_html=True)


        else:

            price = hasil_prediksi[0]
            price_format = '{:,.2f}'.format(price)
            
            st.write('Harga closed saham hari ini adalah Rp. ' + price_format)

            a_number = abs((hasil_prediksi[0] - y[len(y)-1])/y[len(y)-1])
            percentage = "{:.2%}".format(a_number)

            st.write('Harga closed saham hari ini turun sebesar ' + percentage + ' dari harga saham terakhir')

            new_title = '<p style="font-family:sans-serif; color:Red; font-size: 20px;"> Turun </p>'
            st.write(new_title, unsafe_allow_html=True)
    
    else:
        st.write('Tidak ada data untuk prediksi')



