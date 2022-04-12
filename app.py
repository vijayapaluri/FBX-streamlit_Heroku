# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 23:12:04 2022

@author: eagle
"""
import plotly_express as px
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.formula.api as smf
import statsmodels.api as sm
from pandas.plotting import register_matplotlib_converters
from tabulate import tabulate
register_matplotlib_converters()

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from Daniele_update import *
#china_Amer_East_Co_week,china_Amer_East_Co_month,china_Amer_East_Co_6months,china_North_eu_week,china_North_Eu_month

def plot(x):
        fig,ax=plt.subplots()
        ax.plot(x)
        st.pyplot(fig)
        
def bar_plot(x,y,x_title,y_title):
        fig,ax=plt.subplots()
        ax=sns.lineplot(x,y,hue=x)
        ax.set_xlabel(x_title)
        ax.set_ylabel(y_title)
        st.pyplot(fig)

st.title("FBX logistics")
st.sidebar.date_input("Start Date")
st.sidebar.date_input("End Date")
regions=st.sidebar.selectbox("Regions", ["CAEC","CNE","Global","AECNEU","AWCC","NEUC","NEUAEC"])
duration=st.sidebar.selectbox("Days", ["7 Days","30 Days","180 Days"])
forecast=st.sidebar.button("Forecast")
if forecast:
    if regions=="CAEC":
        st.write("China America East Coast")
        df=pd.read_csv("C:/Users/vijjaya/CEA-NAW.csv", parse_dates = ['Date'], index_col = ['Date'])
        st.dataframe(df.describe())
        st.line_chart(df)
        plot(df)
        if duration=="7 Days":
            st.write("week")
            y_test,y_pred=china_Amer_East_Co_week(df)
            st.header("Arima")
            fig=plt.figure(figsize=(12,6))
            plt.plot(y_test)
            plt.plot(y_pred)
            plt.xlabel("series")
            plt.ylabel("values")
            plt.legend(['series','values'])
            st.pyplot(fig)
            st.dataframe(pd.DataFrame({"Pred":y_pred}))
            
        elif duration=="30 Days":
            st.write("Month")
            y_test,y_pred,y_pred2=china_Amer_East_Co_month(df)
            st.header("Exponential smoothing")
            fig=plt.figure(figsize=(12,6))
            plt.plot(y_test)
            plt.plot(y_pred)
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend(['series','values'])
            st.pyplot(fig)
            
            fig=plt.figure(figsize=(12,6))
            plt.plot(y_test)
            plt.plot(y_pred2)
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend(['series','values'])
            st.pyplot(fig)
            st.dataframe(pd.DataFrame({"Pred":y_pred}))
            
        elif duration=="180 Days":
            st.write("6 Months")
            y_test,y_pred,y_pred2=china_Amer_East_Co_6months(df)
            st.header("Exponential smoothing")
            fig=plt.figure(figsize=(12,6))
            plt.plot(y_test)
            plt.plot(y_pred)
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend(['series','values'])
            st.pyplot(fig)
            
            fig=plt.figure(figsize=(12,6))
            plt.plot(y_test)
            plt.plot(y_pred2)
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend(['series','values'])
            st.pyplot(fig)
            st.dataframe(pd.DataFrame({"Pred":y_pred}))
    elif regions=="CNE":
        st.write("China North Europe")
        df=pd.read_csv("C:/Users/vijjaya/China-East-Asia-North-Europe.csv", parse_dates = ['Date'], index_col = ['Date'])
        st.dataframe(df.describe())
        st.line_chart(df)
        if duration=="7 Days":
            st.write("week")
            y_test,y_pred,y_pred2=china_North_eu_week(df)
            st.header("Exponential Smoothing")
            fig=plt.figure(figsize=(12,6))
            plt.plot(y_test)
            plt.plot(y_pred)
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend(['series','values'])
            st.pyplot(fig)
            
            
            fig=plt.figure(figsize=(12,6))
            plt.plot(y_test)
            plt.plot(y_pred2)
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend(['series','values'])
            st.pyplot(fig)
            st.dataframe(pd.DataFrame({"Pred":y_pred}))
            
        elif duration=="30 Days":
            st.write("Month")
            y_test,y_pred,y_pred2=china_North_Eu_month(df)
            st.header("Exponential Smoothing")
            fig=plt.figure(figsize=(12,6))
            plt.plot(y_test)
            plt.plot(y_pred)
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend(['series','values'])
            st.pyplot(fig)
            
            fig=plt.figure(figsize=(12,6))
            plt.plot(y_test)
            plt.plot(y_pred2)
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend(['series','values'])
            st.pyplot(fig)
            st.dataframe(pd.DataFrame({"Pred":y_pred}))
            
        elif duration=="180 Days":
            st.write("6 Months")
            y_test,y_pred,y_pred2=china_North_Eu_6months(df)
            st.header("Exponential Smoothing")
            fig=plt.figure(figsize=(12,6))
            plt.plot(y_test)
            plt.plot(y_pred)
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend(['series','values'])
            st.pyplot(fig)
            
            fig=plt.figure(figsize=(12,6))
            plt.plot(y_test)
            plt.plot(y_pred2)
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend(['series','values'])
            st.pyplot(fig)
            st.dataframe(pd.DataFrame({"Pred":y_pred}))
            
    elif regions=="Global":
        st.write("Global")
        df=pd.read_csv("C:/Users/vijjaya/Global - daily.csv", parse_dates = ['Date'], index_col = ['Date']) 
        st.dataframe(df.describe())
        plot_ly=px.line(df)
        st.plotly_chart(plot_ly)
        if duration=="7 Days":
            st.write("week")
            st.header("ARIMA")
            y_test,y_pred=global_week(df)
            fig=plt.figure(figsize=(12,6))
            plt.plot(y_test)
            plt.plot(y_pred)
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend(['series','values'])
            st.pyplot(fig)
            st.dataframe(pd.DataFrame({"Pred":y_pred}))
            
            
        elif duration=="30 Days":
            st.write("Month")
            st.header("ARIMA")
            y_test,y_pred=global_month(df)
            fig=plt.figure(figsize=(12,6))
            plt.plot(y_test)
            plt.plot(y_pred)
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend(['series','values'])
            st.pyplot(fig)
            st.dataframe(y_pred)
            
        elif duration=="180 Days":
            st.write("6 Months")
            st.header("ARIMA")
            y_test,y_pred=global_6m(df)
            fig=plt.figure(figsize=(12,6))
            plt.plot(y_test)
            plt.plot(y_pred)
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend(['series','values'])
            st.pyplot(fig)
            st.dataframe(pd.DataFrame({"Pred":y_pred}))
            
    elif regions=="AECNEU":
        st.write("America East Coast North Europe")
        df=pd.read_csv('C:/Users/vijjaya/Norrth-America-East-Coast-North-Europe.csv', parse_dates = ['Date'], index_col = ['Date']) 
        st.dataframe(df.describe())
        st.line_chart(df)
        if duration=="7 Days":
            st.write("week")
            st.header("ExponentialSmoothing")
            y_test,y_pred=Aecneu_w(df)
            fig=plt.figure(figsize=(12,6))
            plt.plot(y_test)
            plt.plot(y_pred)
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend(['series','values'])
            st.pyplot(fig)
            st.dataframe(pd.DataFrame({"Pred":y_pred}))
            
            
        elif duration=="30 Days":
            st.write("Month")
            st.header("ExponentialSmoothing")
            y_test,y_pred=Aecneu_m(df)
            fig=plt.figure(figsize=(12,6))
            plt.plot(y_test)
            plt.plot(y_pred)
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend(['series','values'])
            st.pyplot(fig)
            st.dataframe(pd.DataFrame({"Pred":y_pred}))
            
            
        elif duration=="180 Days":
            st.write("6 Months")
            st.header("ARIMA")
            y_test,y_pred=Aecneu_6m(df)
            fig=plt.figure(figsize=(12,6))
            plt.plot(y_test)
            plt.plot(y_pred)
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend(['series','values'])
            st.pyplot(fig)
            st.dataframe(pd.DataFrame({"Pred":y_pred}))
            
            
            
            
    elif regions=="AWCC":
        st.write("America West Coast China")
        df=pd.read_csv('C:/Users/vijjaya/North-America-West-Coast-China-East-Asia.csv',  parse_dates = ['Date'], index_col = ['Date']) 
        st.dataframe(df.describe())
        st.line_chart(df)
        if duration=="7 Days":
            st.write("week")
            st.header("LSTM")
            y_test,y_pred=Awcc_w(df)
            fig=plt.figure(figsize=(12,6))
            plt.plot(y_test)
            plt.plot(y_pred)
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend(['series','values'])
            st.pyplot(fig)
            st.dataframe(y_pred)
            
        elif duration=="30 Days":
            st.write("Month")
            st.header("ExponentialSmoothing")
            y_test,y_pred=Awcc_m(df)
            fig=plt.figure(figsize=(12,6))
            plt.plot(y_test)
            plt.plot(y_pred)
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend(['series','values'])
            st.dataframe(pd.DataFrame({"Pred":y_pred}))
            
            st.pyplot(fig)
        elif duration=="180 Days":
            st.write("6 Months")
            st.header("ARIMA")
            y_test,y_pred=Awcc_6m(df)
            fig=plt.figure(figsize=(12,6))
            plt.plot(y_test)
            plt.plot(y_pred)
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend(['series','values'])
            st.pyplot(fig)
            st.dataframe(pd.DataFrame({"Pred":y_pred}))
            
    elif regions=="NEUC":
        st.write("North EU China")
        df=pd.read_csv('C:/Users/vijjaya/North-Europe-China-East-Asia.csv', parse_dates = ['Date'], index_col = ['Date']) 
        st.dataframe(df.describe())
        st.line_chart(df)
        if duration=="7 Days":
            st.write("week")
            st.header("ARIMA")
            y_test,y_pred=Neuc_w(df)
            fig=plt.figure(figsize=(12,6))
            plt.plot(y_test)
            plt.plot(y_pred)
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend(['series','values'])
            st.pyplot(fig)
            st.dataframe(pd.DataFrame({"Pred":y_pred}))
            
        elif duration=="30 Days":
            st.write("Month")
            st.header("ExponentialSmoothing")
            y_test,y_pred=Neuc_m(df)
            fig=plt.figure(figsize=(12,6))
            plt.plot(y_test)
            plt.plot(y_pred)
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend(['series','values'])
            st.pyplot(fig)
            st.dataframe(pd.DataFrame({"Pred":y_pred}))
            
        elif duration=="180 Days":
            st.write("6 Months")
            st.header("ARIMA")
            y_test,y_pred=Neuc_6m(df)
            fig=plt.figure(figsize=(12,6))
            plt.plot(y_test)
            plt.plot(y_pred)
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend(['series','values'])
            st.pyplot(fig)
            st.dataframe(pd.DataFrame({"Pred":y_pred}))
            
            
    elif regions=="NEUAEC":
        st.write("North EU America East Coast")
        df=pd.read_csv('C:/Users/vijjaya/North-Europe-North-America-East-Coast.csv', parse_dates = ['Date'], index_col = ['Date']) 
        st.dataframe(df.describe())
        st.line_chart(df)
        if duration=="7 Days":
            st.write("7 Days")
            st.header("ARIMA")
            y_test,y_pred=Neaec_w(df)
            fig=plt.figure(figsize=(12,6))
            plt.plot(y_test)
            plt.plot(y_pred)
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend(['series','values'])
            st.pyplot(fig)
            st.dataframe(pd.DataFrame({"Pred":y_pred}))
            
        elif duration=="30 Days":
            st.write("30 Days")
            st.header("ARIMA")
            y_test,y_pred=Neaec_m(df)
            fig=plt.figure(figsize=(12,6))
            plt.plot(y_test)
            plt.plot(y_pred)
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend(['series','values'])
            st.pyplot(fig)
            st.dataframe(pd.DataFrame({"Pred":y_pred}))
            
        elif duration=="180 Days":
            st.write("180 Days")
            st.header("ARIMA")
            y_test,y_pred=Neaec_6m(df)
            fig=plt.figure(figsize=(12,6))
            plt.plot(y_test)
            plt.plot(y_pred)
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend(['series','values'])
            st.pyplot(fig)
            st.dataframe(pd.DataFrame({"Pred":y_pred}))