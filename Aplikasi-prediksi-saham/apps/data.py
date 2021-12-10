import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
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


        return  X_train, X_test, y_train, y_test

    #fungsi membbuat model
    def build_model(X_train, y_train):
    
        #menggunakan model linear regression
        model = LinearRegression()
        model.fit(X_train, y_train)

        return model

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
    
    
