import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model, load_model
from sklearn import datasets

def app():
    st.write('## Data Page')
    
    st.write('Upload data saham')

    sc = StandardScaler()

    #fungsi praproses data
    def preprocess(dataset):

        dataset = dataset.dropna()
        dataset = dataset.reset_index(drop=True)

        return dataset

    #fungsi split data
    def split(X, y):

        X_train = X[:int(len(X)*0.8)]
        X_test = X[int(len(X)*0.2):]
        y_train = y[:int(len(y)*0.8)]
        y_test = y[int(len(y)*0.2):]

        return X_train, X_test, y_train, y_test

    #fungsi seleksi fitur
    def select_feature_target(dataset):
        
        data = preprocess(dataset)

        sc = StandardScaler()

        #menggunakan fitur dan target
        X = data[['Open','High','Low']].values
        y = data['Close'].values

        #split data
        X_train, X_test, y_train, y_test = split(X, y)

        #normalisasi
        X_train = sc.fit_transform(X_train)
        X_test  = sc.fit_transform(X_test)
        y_train = sc.fit_transform(y_train.reshape(-1,1))

        #reshape data agar bisa diinput ke model
        X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
        X_test  = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

        return  X_train, X_test, y_train, y_test

    #fungsi membbuat model
    def build_model(X_train, y_train):

        lstm_model= tf.keras.Sequential()
        lstm_model.add(tf.keras.layers.LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
        lstm_model.add(tf.keras.layers.LSTM(units=50))
        lstm_model.add(tf.keras.layers.Dense(1))

        lstm_model.compile(loss='mean_squared_error',optimizer='adam')
        lstm_model.fit(X_train, y_train, epochs=200 , batch_size=32,verbose=2)

        #filepath = './saved_model'
        #save_model(lstm_model, filepath)
        return lstm_model 

    global dataframe
    global model

    df = False
    dataset_name = st.file_uploader("Choose a file")
    if dataset_name is not None:
        dataframe = pd.read_csv(dataset_name)
        X = dataframe[['Open','High','Low']].values
        y = dataframe['Close'].values
        X_train, X_test, y_train, y_test = select_feature_target(dataframe)
        df = True      

    st.write("Tabel data saham")
    if df == True:
        st.write(dataframe)
    
    
    #membuat model
    train = st.button("Buat Model")
    st.write(train)
    if train == True:
        model = build_model(X_train, y_train)
    
    
