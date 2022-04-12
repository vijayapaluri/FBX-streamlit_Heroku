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


def china_Amer_East_Co_week(df):
    China_Amer_East_Co_Train_w    = df[0:536]
    China_Amer_East_Co_Test_w=df[536:543]
    China_Amer_East_Co_Train_w    = np.log(China_Amer_East_Co_Train_w)
    China_Amer_East_Co_Test_w    = np.log(China_Amer_East_Co_Test_w)
    China_Amer_East_Co_Train_w = China_Amer_East_Co_Train_w.reset_index()
    #China_Amer_East_Co_Test_w_test=China_Amer_East_Co_Test_w.reset_index()
    y_CAEC_w = China_Amer_East_Co_Train_w[' Price ']
    x_CAEC_w = range(0, 536)
    China_Amer_East_Co_Train_w.columns.values
    CAEC_lm_w = smf.ols('y_CAEC_w ~ x_CAEC_w', data = China_Amer_East_Co_Train_w)
    CAEC_lm_w = CAEC_lm_w.fit()
    CAEC_lm_w.summary()
    CAEC_stationary_w = y_CAEC_w - CAEC_lm_w.fittedvalues
    CAEC_ar1_w = ARIMA(y_CAEC_w, order = (1, 0, 0))
    CAEC_ar1_w = CAEC_ar1_w.fit()
    CAEC_ar1_pred_w = CAEC_ar1_w.predict(start = 536, end = 543)
    CAEC_ar1_2_w = SARIMAX(y_CAEC_w, order = (1, 0, 0))
    CAEC_ar1_2_w = CAEC_ar1_2_w.fit()
    y_pred_CAEC_w = CAEC_ar1_2_w.get_forecast(len(China_Amer_East_Co_Test_w.index)+1)
    y_pred_df_CAEC_w = y_pred_CAEC_w.conf_int(alpha = 0.05) 
    y_pred_df_CAEC_w["Predictions"] = CAEC_ar1_2_w.predict(start = y_pred_df_CAEC_w.index[0], end = y_pred_df_CAEC_w.index[-1])
    CAEC_ar_pred_w = pd.DataFrame(CAEC_ar1_pred_w)
    China_Amer_East_Co_Test_w['pred_1'] = CAEC_ar_pred_w.values[1:8]
    rmspe1_CAEC_1_w = np.sqrt(np.sum(China_Amer_East_Co_Test_w.iloc[:, 0].subtract(China_Amer_East_Co_Test_w.iloc[:, 1])**2)/7)

    y_pred_df_CAEC_w.columns.values
    pred_2_w = pd.DataFrame(y_pred_df_CAEC_w['Predictions'])
    China_Amer_East_Co_Test_w['pred_2_w'] = pred_2_w.values[1:8]
    rmspe1_CAEC_2_w = np.sqrt(np.sum(China_Amer_East_Co_Test_w.iloc[:, 0].subtract(China_Amer_East_Co_Test_w.iloc[:, 2])**2)/7)

    CAEC_ar1_w2 = ARIMA(y_CAEC_w, order = (1, 0, 1))
    CAEC_ar1_w2 = CAEC_ar1_w2.fit()

    
    CAEC_ar1_pred_w2 = CAEC_ar1_w2.predict(start = 536, end = 543)

    CAEC_ar1_2_w2 = SARIMAX(y_CAEC_w, order = (1, 1, 0)) #(1 0 1) des not become stationary
    CAEC_ar1_2_w2 = CAEC_ar1_2_w2.fit()
    y_pred_CAEC_w2 = CAEC_ar1_2_w2.get_forecast(len(China_Amer_East_Co_Test_w.index)+1)
    y_pred_df_CAEC_w2 = y_pred_CAEC_w2.conf_int(alpha = 0.05) 
    y_pred_df_CAEC_w2["Predictions"] = CAEC_ar1_2_w2.predict(start = y_pred_df_CAEC_w2.index[0], end = y_pred_df_CAEC_w2.index[-1])

   
    CAEC_ar_pred_w2 = pd.DataFrame(CAEC_ar1_pred_w2)
    China_Amer_East_Co_Test_w['pred_1'] = CAEC_ar_pred_w2.values[1:8]
    rmspe2_CAEC_1_w = np.sqrt(np.sum(China_Amer_East_Co_Test_w.iloc[:, 0].subtract(China_Amer_East_Co_Test_w.iloc[:, 1])**2)/7)

    y_pred_df_CAEC_w2.columns.values
    pred_2_w2 = pd.DataFrame(y_pred_df_CAEC_w2['Predictions'])
    China_Amer_East_Co_Test_w['pred_2_w'] = pred_2_w2.values[1:8]
    rmspe2_CAEC_2_w = np.sqrt(np.sum(China_Amer_East_Co_Test_w.iloc[:, 0].subtract(China_Amer_East_Co_Test_w.iloc[:, 2])**2)/7)
    return y_CAEC_w,CAEC_ar1_pred_w

def china_Amer_East_Co_month(df):
#------------------------------------------------------------#
    #30 Days
    China_Amer_East_Co_Train_m    = df[0:513]
    China_Amer_East_Co_Test_m     = df[513:543]
    China_Amer_East_Co_Train_m    = np.log(China_Amer_East_Co_Train_m)
    China_Amer_East_Co_Test_m    = np.log(China_Amer_East_Co_Test_m)
    CAEC_model_m = ExponentialSmoothing(China_Amer_East_Co_Train_m, trend='add', seasonal=None)
    CAEC_model2_m = ExponentialSmoothing(China_Amer_East_Co_Train_m, trend='add', seasonal=None, damped=True)
    
    CAEC_fit1_m = CAEC_model_m.fit()
    CAEC_fit2_m = CAEC_model2_m.fit()
    CAEC_pred1_m = CAEC_fit1_m.forecast(30)
    CAEC_pred2_m = CAEC_fit2_m.forecast(30)
    
    #CAEC_model_test=ExponentialSmoothing(China_Amer_East_Co_Test_m,trend="add",seasonal=None)
    
    
    rmspe_CAEC_Exp1_m = np.sqrt(np.sum(China_Amer_East_Co_Test_m.iloc[:, 0].subtract(CAEC_pred1_m)**2)/29)
    rmspe_CAEC_Exp2_m = np.sqrt(np.sum(China_Amer_East_Co_Test_m.iloc[:, 0].subtract(CAEC_pred2_m)**2)/29)
    return China_Amer_East_Co_Test_m,CAEC_pred1_m,CAEC_pred2_m
    
def china_Amer_East_Co_6months(df):
    #180 Days
    China_Amer_East_Co_Train_6m    = df[0:363]
    China_Amer_East_Co_Test_6m    = df[363:543]
    China_Amer_East_Co_Train_6m    = np.log(China_Amer_East_Co_Train_6m)
    China_Amer_East_Co_Test_6m    = np.log(China_Amer_East_Co_Test_6m)
    CAEC_model_6m = ExponentialSmoothing(China_Amer_East_Co_Train_6m, trend='add', seasonal=None)
    CAEC_model2_6m = ExponentialSmoothing(China_Amer_East_Co_Train_6m, trend='add', seasonal=None, damped=True)
    
    CAEC_fit1_6m = CAEC_model_6m.fit()
    CAEC_fit2_6m = CAEC_model2_6m.fit()
    CAEC_pred1_6m = CAEC_fit1_6m.forecast(180)
    CAEC_pred2_6m = CAEC_fit2_6m.forecast(180)
    rmspe_CAEC_Exp1_6m = np.sqrt(np.sum(China_Amer_East_Co_Test_6m.iloc[:, 0].subtract(CAEC_pred1_6m)**2)/180)
    rmspe_CAEC_Exp2_6m = np.sqrt(np.sum(China_Amer_East_Co_Test_6m.iloc[:, 0].subtract(CAEC_pred2_6m)**2)/180)
    return China_Amer_East_Co_Test_6m,CAEC_pred1_6m,CAEC_pred2_6m
    


def china_North_eu_week(df):
    China_North_EU_Train_w        = df[0:536]
    China_North_EU_Test_w        = df[536:543]
    China_North_EU_Train_w        = np.log(China_North_EU_Train_w)
    China_North_EU_Test_w        =  np.log(China_North_EU_Train_w)
    # 7 Days
    CNEU_model_w = ExponentialSmoothing(China_North_EU_Train_w , trend='add', seasonal=None)
    CNEU_model2_w = ExponentialSmoothing(China_North_EU_Train_w, trend='add', seasonal=None, damped=True)
    CNEU_fit1_w = CNEU_model_w.fit()
    CNEU_fit2_w = CNEU_model2_w.fit()
    CNEU_pred1_w = CNEU_fit1_w.forecast(7)
    CNEU_pred2_w = CNEU_fit2_w.forecast(7)
    rmspe_CNEU_Exp1_w = np.sqrt(np.sum(China_North_EU_Test_w.iloc[:, 0].subtract(CNEU_pred1_w)**2)/7)
    rmspe_CNEU_Exp2_w = np.sqrt(np.sum(China_North_EU_Test_w.iloc[:, 0].subtract(CNEU_pred2_w)**2)/7)
    return China_North_EU_Test_w,CNEU_pred1_w,CNEU_pred2_w

def china_North_Eu_month(df):
    China_North_EU_Train_m        = df[0:513]
    China_North_EU_Test_m        = df[513:543]
    China_North_EU_Train_m        = np.log(China_North_EU_Train_m)
    China_North_EU_Test_m        = np.log(China_North_EU_Test_m)
    CNEU_model_m = ExponentialSmoothing(China_North_EU_Train_m, trend='add', seasonal=None)
    CNEU_model2_m = ExponentialSmoothing(China_North_EU_Train_m, trend='add', seasonal=None, damped=True)

    CNEU_fit1_m = CNEU_model_m.fit()
    CNEU_fit2_m = CNEU_model2_m.fit()
    CNEU_pred1_m = CNEU_fit1_m.forecast(30)
    CNEU_pred2_m = CNEU_fit2_m.forecast(30)

    rmspe_CNEU_Exp1_m = np.sqrt(np.sum(China_North_EU_Test_m.iloc[:, 0].subtract(CNEU_pred1_m)**2)/29)
    rmspe_CNEU_Exp2_m = np.sqrt(np.sum(China_North_EU_Test_m.iloc[:, 0].subtract(CNEU_pred2_m)**2)/29)
    return China_North_EU_Test_m,CNEU_pred1_m,CNEU_pred2_m
def china_North_Eu_6months(df):
    China_North_EU_Train_6m        = df[0:363]
    China_North_EU_Train_6m        = np.log(China_North_EU_Train_6m)
    China_North_EU_Test_6m        = df[363:543]
    China_North_EU_Test_6m        = np.log(China_North_EU_Test_6m)

    CNEU_model_6m = ExponentialSmoothing(China_North_EU_Train_6m, trend='add', seasonal=None)
    CNEU_model2_6m = ExponentialSmoothing(China_North_EU_Train_6m, trend='add', seasonal=None, damped=True)

    CNEU_fit1_6m = CNEU_model_6m.fit()
    CNEU_fit2_6m = CNEU_model2_6m.fit()
    CNEU_pred1_6m = CNEU_fit1_6m.forecast(180)
    CNEU_pred2_6m = CNEU_fit2_6m.forecast(180)

    rmspe_CNEU_Exp1_6m = np.sqrt(np.sum(China_North_EU_Test_6m.iloc[:, 0].subtract(CNEU_pred1_6m)**2)/180)
    rmspe_CNEU_Exp2_6m = np.sqrt(np.sum(China_North_EU_Test_6m.iloc[:, 0].subtract(CNEU_pred2_6m)**2)/180)
    return China_North_EU_Test_6m,CNEU_pred1_6m,CNEU_pred2_6m
    
def global_week(df):
    Global_Train_w                = df[0:536]
    Global_Test_w                = df[536:543]
    Global_Train_w                = np.log(Global_Train_w)
    Global_Test_w                = np.log(Global_Test_w)

    

    Global_Train_w = Global_Train_w.reset_index()
    y_Global_w = Global_Train_w[' Price ']
    x_Global_w = range(0, 536)
    Global_lm_w = smf.ols('y_Global_w ~ x_Global_w', data = Global_Train_w)
    Global_lm_w = Global_lm_w.fit()
    Global_lm_w.summary()

   

    Global_stationary_w = y_Global_w - Global_lm_w.fittedvalues
   

    Global_ar1_w = ARIMA(y_Global_w, order = (1,0,0))
    Global_ar1_w = Global_ar1_w.fit()

    
    Global_ar1_pred_w = Global_ar1_w.predict(start = 536, end = 543)

    

    Global_ar1_2_w = SARIMAX(y_Global_w, order = (1,0,0))
    Global_ar1_2_w = Global_ar1_2_w.fit()
    y_pred_Global_w = Global_ar1_2_w.get_forecast(len(Global_Test_w.index)+1)
    y_pred_df_Global_w = y_pred_Global_w.conf_int(alpha = 0.05)
    y_pred_df_Global_w['Predictions'] = Global_ar1_2_w.predict(start = y_pred_df_Global_w.index[0], end = y_pred_df_Global_w.index[-1])

    
    Global_ar_pred_w = pd.DataFrame(Global_ar1_pred_w)
    Global_Test_w['pred_1'] = Global_ar_pred_w.values[1:8]
    rmspe1_Global_1_w = np.sqrt(np.sum(Global_Test_w.iloc[:, 0].subtract(Global_Test_w.iloc[:, 1])**2)/7)

    pred_2_w = pd.DataFrame(y_pred_df_Global_w['Predictions'])
    Global_Test_w['pred_2_w'] = pred_2_w.values[1: 8]
    rmspe1_Global_2_w = np.sqrt(np.sum(Global_Test_w.iloc[:, 0].subtract(Global_Test_w.iloc[:, 2])**2)/7)

    ### ARIMA model (1 0 1)
    Global_ar1_w2 = ARIMA(y_Global_w, order = (1, 0, 1))
    Global_ar1_w2 = Global_ar1_w2.fit()
    

    Global_ar1_pred_w2 = Global_ar1_w2.predict(start = 536, end = 543)

    
    Global_ar1_2_w2 = SARIMAX(y_Global_w, order = (1, 1, 0))
    Global_ar1_2_w2 = Global_ar1_2_w2.fit()
    y_pred_Global_w2 = Global_ar1_2_w2.get_forecast(len(Global_Test_w.index)+1)
    y_pred_df_Global_w2 = y_pred_Global_w2.conf_int(alpha = 0.05) 
    y_pred_df_Global_w2["Predictions"] = Global_ar1_2_w2.predict(start = y_pred_df_Global_w2.index[0], end = y_pred_df_Global_w2.index[-1])

    
    Global_ar_pred_w2 = pd.DataFrame(Global_ar1_pred_w2)
    Global_Test_w['pred_1'] = Global_ar_pred_w2.values[1:8]
    rmspe2_Global_1_w = np.sqrt(np.sum(Global_Test_w.iloc[:, 0].subtract(Global_Test_w.iloc[:, 1])**2)/7)

    y_pred_df_Global_w2.columns.values
    pred_2_w2 = pd.DataFrame(y_pred_df_Global_w2['Predictions'])
    Global_Test_w['pred_2_w'] = pred_2_w2.values[1:8]
    rmspe2_Global_2_w = np.sqrt(np.sum(Global_Test_w.iloc[:, 0].subtract(Global_Test_w.iloc[:, 2])**2)/7)
    return y_Global_w,Global_ar1_pred_w2

def global_month(df):
    Global_Train_m                = df[0:513]
    Global_Test_m                = df[513:543]
    Global_Train_m                = np.log(Global_Train_m)
    Global_Test_m                = np.log(Global_Test_m)
    

    Global_Train_m = Global_Train_m.reset_index()
    y_Global_m = Global_Train_m[' Price ']
    x_Global_m = range(0, 513)
    Global_lm_m = smf.ols('y_Global_m ~ x_Global_m', data = Global_Train_m)
    Global_lm_m = Global_lm_m.fit()
    Global_lm_m.summary()
    

    Global_stationary_m = y_Global_m - Global_lm_m.fittedvalues
   

    Global_ar_m = ARIMA(y_Global_m, order = (1, 0, 0))
    Global_ar_m = Global_ar_m.fit()

    

    Global_ar_pred_m = Global_ar_m.predict(start = 513, end = 543)

   

    Global_ar_2_m = SARIMAX(y_Global_m, order = (1, 0, 0))

    Global_ar_2_m = Global_ar_2_m.fit()
    y_pred_Global_m = Global_ar_2_m.get_forecast(len(Global_Test_m.index)+1)
    y_pred_df_Global_m = y_pred_Global_m.conf_int(alpha = 0.05) 
    y_pred_df_Global_m["Predictions"] = Global_ar_2_m.predict(start = y_pred_df_Global_m.index[0], end = y_pred_df_Global_m.index[-1])

    
    Global_ar_pred_m = pd.DataFrame(Global_ar_pred_m)
    Global_Test_m['pred_1'] = Global_ar_pred_m.values[1:32]
    rmspe_Global_1_m = np.sqrt(np.sum(Global_Test_m.iloc[:, 0].subtract(Global_Test_m.iloc[:, 1])**2)/30)

    pred_2_m = pd.DataFrame(y_pred_df_Global_m['Predictions'])
    Global_Test_m['pred_2_m'] = pred_2_m.values[1:31]
    rmspe__Global_2_m = np.sqrt(np.sum(Global_Test_m.iloc[:, 0].subtract(Global_Test_m.iloc[:, 2])**2)/30)
    return y_Global_m,Global_ar_pred_m



def global_6m(df):
    Global_Train_6m                = df[0:363]
    Global_Test_6m                = df[363:543]
    Global_Train_6m                = np.log(Global_Train_6m)
    Global_Test_6m                = np.log(Global_Test_6m)
    
    Global_Train_6m = Global_Train_6m.reset_index()
    y_Global_6m = Global_Train_6m[' Price ']
    Global_ar_6m2 = ARIMA(y_Global_6m, order = (2, 0, 0))#(1 0 1) it doesn't become statioanry
    Global_ar_6m2= Global_ar_6m2.fit()

    
    Global_ar_pred_6m2 = Global_ar_6m2.predict(start = 363, end = 543)

    
    Global_ar_2_6m2 = SARIMAX(y_Global_6m, order = (2, 0, 0))
    Global_ar_2_6m2 = Global_ar_2_6m2.fit()
    y_pred_Global_6m2 = Global_ar_2_6m2.get_forecast(len(Global_Test_6m.index)+1)
    y_pred_df_Global_6m2 = y_pred_Global_6m2.conf_int(alpha = 0.05) 
    y_pred_df_Global_6m2["Predictions"] = Global_ar_2_6m2.predict(start = y_pred_df_Global_6m2.index[0], end = y_pred_df_Global_6m2.index[-1])

    
    Global_ar_pred6_m2 = pd.DataFrame(Global_ar_pred_6m2)
    Global_Test_6m['pred_1'] = Global_ar_pred_6m2.values[1:181]
    rmspe__Global_1_6m2 = np.sqrt(np.sum(Global_Test_6m.iloc[:, 0].subtract(Global_Test_6m.iloc[:, 1])**2)/180)

    y_pred_df_Global_6m2.columns.values
    pred_2_6m2 = pd.DataFrame(y_pred_df_Global_6m2['Predictions'])
    Global_Test_6m['pred_2_6m'] = pred_2_6m2.values[1:181]
    rmspe_Global_2_6m2 = np.sqrt(np.sum(Global_Test_6m.iloc[:, 0].subtract(Global_Test_6m.iloc[:, 2])**2)/180)
    return y_Global_6m,Global_ar_pred_6m2


def Aecneu_w(df):
    Amer_East_Co_North_EU_Train_w = df[0:536]
    Amer_East_Co_North_EU_Test_w = df[536:543]     
    Amer_East_Co_North_EU_Train_w = np.log(Amer_East_Co_North_EU_Train_w)
    Amer_East_Co_North_EU_Test_w = np.log(Amer_East_Co_North_EU_Test_w)        
    AECNEU_model_w = ExponentialSmoothing(Amer_East_Co_North_EU_Train_w , trend='add', seasonal=None)
    AECNEU_model2_w = ExponentialSmoothing(Amer_East_Co_North_EU_Train_w, trend='add', seasonal=None, damped=True)

    AECNEU_fit1_w = AECNEU_model_w.fit()
    AECNEU_fit2_w = AECNEU_model2_w.fit()
    AECNEU_pred1_w = AECNEU_fit1_w.forecast(7)
    AECNEU_pred2_w = AECNEU_fit2_w.forecast(7)
    rmspe_AECNEU_Exp1_w = np.sqrt(np.sum(Amer_East_Co_North_EU_Test_w.iloc[:, 0].subtract(AECNEU_pred1_w)**2)/7)
    rmspe_AECNEU_Exp2_w = np.sqrt(np.sum(Amer_East_Co_North_EU_Test_w.iloc[:, 0].subtract(AECNEU_pred2_w)**2)/7)
    return Amer_East_Co_North_EU_Test_w,AECNEU_pred1_w

def Aecneu_m(df):
    Amer_East_Co_North_EU_Train_m = df[0:513]
    Amer_East_Co_North_EU_Test_m = df[513:543]
    Amer_East_Co_North_EU_Train_m = np.log(Amer_East_Co_North_EU_Train_m)
    Amer_East_Co_North_EU_Test_m = np.log(Amer_East_Co_North_EU_Test_m)
    AECNEU_model_m = ExponentialSmoothing(Amer_East_Co_North_EU_Train_m, trend='add', seasonal=None)
    AECNEU_model2_m = ExponentialSmoothing(Amer_East_Co_North_EU_Train_m, trend='add', seasonal=None, damped=True)

    AECNEU_fit1_m = AECNEU_model_m.fit()
    AECNEU_fit2_m = AECNEU_model2_m.fit()
    AECNEU_pred1_m = AECNEU_fit1_m.forecast(30)
    AECNEU_pred2_m = AECNEU_fit2_m.forecast(30)

    rmspe_AECNEU_Exp1_m = np.sqrt(np.sum(Amer_East_Co_North_EU_Test_m.iloc[:, 0].subtract(AECNEU_pred1_m)**2)/29)
    rmspe_AECNEU_Exp2_m = np.sqrt(np.sum(Amer_East_Co_North_EU_Test_m.iloc[:, 0].subtract(AECNEU_pred2_m)**2)/29)
    return Amer_East_Co_North_EU_Test_m,AECNEU_pred1_m


def Aecneu_6m(df):
    #Arima model (1 0 0) 180 days Amer_East_Co_North_EU
    Amer_East_Co_North_EU_Train_6m = df[0:363]
    Amer_East_Co_North_EU_Test_6m = df[363:543]
    Amer_East_Co_North_EU_Train_6m = np.log(Amer_East_Co_North_EU_Train_6m)
    Amer_East_Co_North_EU_Test_6m = np.log(Amer_East_Co_North_EU_Test_6m)
    

    Amer_East_Co_North_EU_Train_6m = Amer_East_Co_North_EU_Train_6m.reset_index()
    y_AECNEU_6m = Amer_East_Co_North_EU_Train_6m[' Price ']
    x_AECNEU_6m = range(0, 363)
    AECNEU_lm_6m = smf.ols('y_AECNEU_6m ~ x_AECNEU_6m', data = Amer_East_Co_North_EU_Train_6m)
    AECNEU_lm_6m = AECNEU_lm_6m.fit()
    AECNEU_lm_6m.summary()

    

    AECNEU_stationary_6m = y_AECNEU_6m - AECNEU_lm_6m.fittedvalues
   

    AECNEU_ar_6m = ARIMA(y_AECNEU_6m, order = (1, 0, 0))
    AECNEU_ar_6m = AECNEU_ar_6m.fit()

    

    AECNEU_ar_pred_6m = AECNEU_ar_6m.predict(start = 363, end = 543)

    
    AECNEU_ar_2_6m = SARIMAX(y_AECNEU_6m, order = (1, 0, 0))
    AECNEU_ar_2_6m = AECNEU_ar_2_6m.fit()
    y_pred_AECNEU_6m = AECNEU_ar_2_6m.get_forecast(len(Amer_East_Co_North_EU_Test_6m.index)+1)
    y_pred_df_AECNEU_6m = y_pred_AECNEU_6m.conf_int(alpha = 0.05) 
    y_pred_df_AECNEU_6m["Predictions"] = AECNEU_ar_2_6m.predict(start = y_pred_df_AECNEU_6m.index[0], end = y_pred_df_AECNEU_6m.index[-1])

    

    AECNEU_ar_pred6_m = pd.DataFrame(AECNEU_ar_pred_6m)
    Amer_East_Co_North_EU_Test_6m['pred_1'] = AECNEU_ar_pred_6m.values[1:181]
    rmspe__AECNEU_1_6m = np.sqrt(np.sum(Amer_East_Co_North_EU_Test_6m.iloc[:, 0].subtract(Amer_East_Co_North_EU_Test_6m.iloc[:, 1])**2)/180)

    y_pred_df_AECNEU_6m.columns.values
    pred_2_6m = pd.DataFrame(y_pred_df_AECNEU_6m['Predictions'])
    Amer_East_Co_North_EU_Test_6m['pred_2_6m'] = pred_2_6m.values[1:181]
    rmspe_AECNEU_2_6m = np.sqrt(np.sum(Amer_East_Co_North_EU_Test_6m.iloc[:, 0].subtract(Amer_East_Co_North_EU_Test_6m.iloc[:, 2])**2)/180)

    AECNEU_ar_6m2 = ARIMA(y_AECNEU_6m, order = (1, 0, 1))#(1 0 1) it doesn't become statioanry
    AECNEU_ar_6m2= AECNEU_ar_6m2.fit()

    

    AECNEU_ar_pred_6m2 = AECNEU_ar_6m2.predict(start = 363, end = 543)

   

    AECNEU_ar_2_6m2 = SARIMAX(y_AECNEU_6m, order = (1, 0, 1))
    AECNEU_ar_2_6m2 = AECNEU_ar_2_6m2.fit()
    y_pred_AECNEU_6m2 = AECNEU_ar_2_6m2.get_forecast(len(Amer_East_Co_North_EU_Test_6m.index)+1)
    y_pred_df_AECNEU_6m2 = y_pred_AECNEU_6m2.conf_int(alpha = 0.05) 
    y_pred_df_AECNEU_6m2["Predictions"] = AECNEU_ar_2_6m2.predict(start = y_pred_df_AECNEU_6m2.index[0], end = y_pred_df_AECNEU_6m2.index[-1])

   
    AECNEU_ar_pred6_m2 = pd.DataFrame(AECNEU_ar_pred_6m2)
    Amer_East_Co_North_EU_Test_6m['pred_1'] = AECNEU_ar_pred_6m2.values[1:181]
    rmspe__AECNEU_1_6m2 = np.sqrt(np.sum(Amer_East_Co_North_EU_Test_6m.iloc[:, 0].subtract(Amer_East_Co_North_EU_Test_6m.iloc[:, 1])**2)/180)

    y_pred_df_AECNEU_6m2.columns.values
    pred_2_6m2 = pd.DataFrame(y_pred_df_AECNEU_6m2['Predictions'])
    Amer_East_Co_North_EU_Test_6m['pred_2_6m'] = pred_2_6m2.values[1:181]
    rmspe_AECNEU_2_6m2 = np.sqrt(np.sum(Amer_East_Co_North_EU_Test_6m.iloc[:, 0].subtract(Amer_East_Co_North_EU_Test_6m.iloc[:, 2])**2)/180)
    return y_AECNEU_6m,AECNEU_ar_pred_6m



def Awcc_w(df):
    Amer_West_Co_China_Train_w    = df[0:536]
    Amer_West_Co_China_Test_w    = df[536:543]
    Amer_West_Co_China_Train_w    = np.log(Amer_West_Co_China_Train_w)
    Amer_West_Co_China_Test_w    = np.log(Amer_West_Co_China_Test_w)
    scaler  = StandardScaler()
    lstm_AWCC_w =  scaler.fit_transform(Amer_West_Co_China_Train_w)
    X_train = []
    y_train = []
    for i in range(7, len(Amer_West_Co_China_Train_w)-7):
        X_train.append(lstm_AWCC_w[i-7:i, 0])
        y_train.append(lstm_AWCC_w[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    model = Sequential()
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))  
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1))
    
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    model.fit(X_train, y_train, epochs = 100, batch_size = 32)
    
    dataset_train_AWCC_w = Amer_West_Co_China_Train_w.iloc[:529]
    dataset_test_AWCC_w = Amer_West_Co_China_Train_w.iloc[529:]
    dataset_total_AWCC_w = pd.concat((dataset_train_AWCC_w, dataset_test_AWCC_w), axis = 0)
    inputs_AWCC_w = dataset_total_AWCC_w[len(dataset_total_AWCC_w) - len(dataset_test_AWCC_w) - 7:].values
    inputs_AWCC_w = inputs_AWCC_w.reshape(-1,1)
    inputs_AWCC_w = scaler.transform(inputs_AWCC_w)
    X_test_AWCC_w = []
    for i in range(7, 14):
        X_test_AWCC_w.append(inputs_AWCC_w[i-7:i, 0])
    X_test_AWCC_w = np.array(X_test_AWCC_w)
    X_test_AWCC_w = np.reshape(X_test_AWCC_w, (X_test_AWCC_w.shape[0], X_test_AWCC_w.shape[1], 1))
    
    
    pred_AWCC_w = model.predict(X_test_AWCC_w)
    pred_AWCC_w = scaler.inverse_transform(pred_AWCC_w)
    
    rmspe_AWCC_lstm_w = np.sqrt(mean_squared_error(Amer_West_Co_China_Test_w[' Price '], pred_AWCC_w))
    rmspe_AWCC_lstm_w
    return y_train,pred_AWCC_w

def Awcc_m(df):
    Amer_West_Co_China_Train_m    = df[0:513]
    Amer_West_Co_China_Test_m    = df[513:543]
    Amer_West_Co_China_Train_m    = np.log(Amer_West_Co_China_Train_m)
    Amer_West_Co_China_Test_m    = np.log(Amer_West_Co_China_Test_m)       
    AWCC_model_m = ExponentialSmoothing(Amer_West_Co_China_Train_m, trend='add', seasonal=None)
    AWCC_model2_m = ExponentialSmoothing(Amer_West_Co_China_Train_m, trend='add', seasonal=None, damped=True)

    AWCC_fit1_m = AWCC_model_m.fit()
    AWCC_fit2_m = AWCC_model2_m.fit()
    AWCC_pred1_m = AWCC_fit1_m.forecast(30)
    AWCC_pred2_m = AWCC_fit2_m.forecast(30) 

    rmspe_AWCC_Exp1_m = np.sqrt(np.sum(Amer_West_Co_China_Test_m.iloc[:, 0].subtract(AWCC_pred1_m)**2)/29)
    rmspe_AWCC_Exp2_m = np.sqrt(np.sum(Amer_West_Co_China_Test_m.iloc[:, 0].subtract(AWCC_pred2_m)**2)/29)
    return Amer_West_Co_China_Test_m,AWCC_pred1_m

def Awcc_6m(df):
    #Arima model (2 0 0) 180 days Amer_West_Co_China
    Amer_West_Co_China_Train_6m    = df[0:363]
    Amer_West_Co_China_Test_6m    = df[363:543]
    Amer_West_Co_China_Train_6m    = np.log(Amer_West_Co_China_Train_6m)
    Amer_West_Co_China_Test_6m    = np.log(Amer_West_Co_China_Test_6m)
    
    Amer_West_Co_China_Train_6m = Amer_West_Co_China_Train_6m.reset_index()
    y_AWCC_6m = Amer_West_Co_China_Train_6m[' Price ']
    x_AWCC_6m = range(0, 363)
    AWCC_lm_6m = smf.ols('y_AWCC_6m ~ x_AWCC_6m', data = Amer_West_Co_China_Train_6m)
    AWCC_lm_6m = AWCC_lm_6m.fit()
    AWCC_lm_6m.summary()


    
    AWCC_stationary_6m = y_AWCC_6m - AWCC_lm_6m.fittedvalues
    plt.plot(AWCC_stationary_6m)
    plot_acf(AWCC_stationary_6m)
    plot_pacf(AWCC_stationary_6m)

    AWCC_ar_6m = ARIMA(y_AWCC_6m, order = (2, 0, 0))
    AWCC_ar_6m = AWCC_ar_6m.fit()

    
    AWCC_ar_pred_6m = AWCC_ar_6m.predict(start = 363, end = 543)

    

    AWCC_ar_2_6m = SARIMAX(y_AWCC_6m, order = (2, 0, 0))
    AWCC_ar_2_6m = AWCC_ar_2_6m.fit()
    y_pred_AWCC_6m = AWCC_ar_2_6m.get_forecast(len(Amer_West_Co_China_Test_6m.index)+1)
    y_pred_df_AWCC_6m = y_pred_AWCC_6m.conf_int(alpha = 0.05) 
    y_pred_df_AWCC_6m["Predictions"] = AWCC_ar_2_6m.predict(start = y_pred_df_AWCC_6m.index[0], end = y_pred_df_AWCC_6m.index[-1])

    

    AWCC_ar_pred6_m = pd.DataFrame(AWCC_ar_pred_6m)
    Amer_West_Co_China_Test_6m['pred_1'] = AWCC_ar_pred_6m.values[1:181]
    rmspe__AWCC_1_6m = np.sqrt(np.sum(Amer_West_Co_China_Test_6m.iloc[:, 0].subtract(Amer_West_Co_China_Test_6m.iloc[:, 1])**2)/180)

    y_pred_df_AWCC_6m.columns.values
    pred_2_6m = pd.DataFrame(y_pred_df_AWCC_6m['Predictions'])
    Amer_West_Co_China_Test_6m['pred_2_6m'] = pred_2_6m.values[1:181]
    rmspe_AWCC_2_6m = np.sqrt(np.sum(Amer_West_Co_China_Test_6m.iloc[:, 0].subtract(Amer_West_Co_China_Test_6m.iloc[:, 2])**2)/180)
    return y_AWCC_6m,AWCC_ar_pred_6m

def Neuc_w(df):
    North_EU_China_Train_w        = df[0:536]
    North_EU_China_Test_w        = df[536:543]
    North_EU_China_Train_w        = np.log(North_EU_China_Train_w)
    North_EU_China_Test_w        = np.log(North_EU_China_Test_w)
    

    North_EU_China_Train_w = North_EU_China_Train_w.reset_index()
    y_NEUC_w = North_EU_China_Train_w[' Price ']
    x_NEUC_w = range(0, 536)
    NEUC_lm_w = smf.ols('y_NEUC_w ~ x_NEUC_w', data = North_EU_China_Train_w)
    NEUC_lm_w = NEUC_lm_w.fit()
    NEUC_lm_w.summary()
    

    NEUC_stationary_w = y_NEUC_w - NEUC_lm_w.fittedvalues
    
    NEUC_ar1_w = ARIMA(y_NEUC_w, order = (5,0,0))
    NEUC_ar1_w = NEUC_ar1_w.fit()

    
    NEUC_ar1_pred_w = NEUC_ar1_w.predict(start = 536, end = 543)

    

    NEUC_ar1_2_w = SARIMAX(y_NEUC_w, order = (5,0,0))
    NEUC_ar1_2_w = NEUC_ar1_2_w.fit()
    y_pred_NEUC_w = NEUC_ar1_2_w.get_forecast(len(North_EU_China_Train_w.index)+1)
    y_pred_df_NEUC_w = y_pred_NEUC_w.conf_int(alpha = 0.05)
    y_pred_df_NEUC_w['Predictions'] = NEUC_ar1_2_w.predict(start = y_pred_df_NEUC_w.index[0], end = y_pred_df_NEUC_w.index[-1])

   
    NEUC_ar_pred_w = pd.DataFrame(NEUC_ar1_pred_w)
    North_EU_China_Test_w['pred_1'] = NEUC_ar_pred_w.values[1:8]
    rmspe1_NEUC_1_w = np.sqrt(np.sum(North_EU_China_Test_w.iloc[:, 0].subtract(North_EU_China_Test_w.iloc[:, 1])**2)/7)

    pred_2_w = pd.DataFrame(y_pred_df_NEUC_w['Predictions'])
    North_EU_China_Test_w['pred_2_w'] = pred_2_w.values[1: 8]
    rmspe1_NEUC_2_w = np.sqrt(np.sum(North_EU_China_Test_w.iloc[:, 0].subtract(North_EU_China_Test_w.iloc[:, 2])**2)/7)
    return y_NEUC_w,NEUC_ar1_pred_w

def Neuc_m(df):
    North_EU_China_Train_m        = df[0:513]
    North_EU_China_Test_m        = df[513:543]
    North_EU_China_Train_m        = np.log(North_EU_China_Train_m)
    North_EU_China_Test_m        = np.log(North_EU_China_Test_m)
    NEUC_model_m = ExponentialSmoothing(North_EU_China_Train_m, trend='add', seasonal=None)
    NEUC_model2_m = ExponentialSmoothing(North_EU_China_Train_m, trend='add', seasonal=None, damped=True)

    NEUC_fit1_m = NEUC_model_m.fit()
    NEUC_fit2_m = NEUC_model2_m.fit()
    NEUC_pred1_m = NEUC_fit1_m.forecast(30)
    NEUC_pred2_m = NEUC_fit2_m.forecast(30)

    rmspe_NEUC_Exp1_m = np.sqrt(np.sum(North_EU_China_Test_m.iloc[:, 0].subtract(NEUC_pred1_m)**2)/29)
    rmspe_NEUC_Exp2_m = np.sqrt(np.sum(North_EU_China_Test_m.iloc[:, 0].subtract(NEUC_pred2_m)**2)/29)
    return North_EU_China_Test_m,NEUC_pred1_m

def Neuc_6m(df):
    North_EU_China_Train_6m        = df[0:363]
    North_EU_China_Test_6m        = df[363:543]
    North_EU_China_Train_6m        = np.log(North_EU_China_Train_6m)
    North_EU_China_Test_6m        = np.log(North_EU_China_Test_6m)
    

    North_EU_China_Train_6m = North_EU_China_Train_6m.reset_index()
    y_NEUC_6m = North_EU_China_Train_6m[' Price ']
    x_NEUC_6m = range(0, 363)
    NEUC_lm_6m = smf.ols('y_NEUC_6m ~ x_NEUC_6m', data = North_EU_China_Train_6m)
    NEUC_lm_6m = NEUC_lm_6m.fit()
    NEUC_lm_6m.summary()

   

    NEUC_stationary_6m = y_NEUC_6m - NEUC_lm_6m.fittedvalues
    
    NEUC_ar_6m = ARIMA(y_NEUC_6m, order = (5, 0, 0))
    NEUC_ar_6m = NEUC_ar_6m.fit()

    
    NEUC_ar_pred_6m = NEUC_ar_6m.predict(start = 363, end = 543)

    
    NEUC_ar_2_6m = SARIMAX(y_NEUC_6m, order = (5, 0, 0))
    NEUC_ar_2_6m = NEUC_ar_2_6m.fit()
    y_pred_NEUC_6m = NEUC_ar_2_6m.get_forecast(len(North_EU_China_Test_6m.index)+1)
    y_pred_df_NEUC_6m = y_pred_NEUC_6m.conf_int(alpha = 0.05) 
    y_pred_df_NEUC_6m["Predictions"] = NEUC_ar_2_6m.predict(start = y_pred_df_NEUC_6m.index[0], end = y_pred_df_NEUC_6m.index[-1])

   
    NEUC_ar_pred6_m = pd.DataFrame(NEUC_ar_pred_6m)
    North_EU_China_Test_6m['pred_1'] = NEUC_ar_pred_6m.values[1:181]
    rmspe__NEUC_1_6m = np.sqrt(np.sum(North_EU_China_Test_6m.iloc[:, 0].subtract(North_EU_China_Test_6m.iloc[:, 1])**2)/180)

    y_pred_df_NEUC_6m.columns.values
    pred_2_6m = pd.DataFrame(y_pred_df_NEUC_6m['Predictions'])
    North_EU_China_Test_6m['pred_2_6m'] = pred_2_6m.values[1:181]
    rmspe_NEUC_2_6m = np.sqrt(np.sum(North_EU_China_Test_6m.iloc[:, 0].subtract(North_EU_China_Test_6m.iloc[:, 2])**2)/180)
    return y_NEUC_6m,NEUC_ar_pred_6m


def Neaec_w(df):
    North_EU_Amer_East_Co_Train_w = df[0:536]
    North_EU_Amer_East_Co_Test_w = df[536:543]
    North_EU_Amer_East_Co_Train_w = np.log(North_EU_Amer_East_Co_Train_w)
    North_EU_Amer_East_Co_Test_w = np.log(North_EU_Amer_East_Co_Test_w)

    North_EU_Amer_East_Co = North_EU_Amer_East_Co_Train_w.reset_index()
    y_NEUAEC_w = North_EU_Amer_East_Co_Train_w[' Price ']
    x_NEUAEC_w = range(0, 536)
    NEUAEC_lm_w = smf.ols('y_NEUAEC_w ~ x_NEUAEC_w', data = North_EU_Amer_East_Co_Train_w)
    NEUAEC_lm_w = NEUAEC_lm_w.fit()
    NEUAEC_lm_w.summary()


    NEUAEC_stationary_w = y_NEUAEC_w - NEUAEC_lm_w.fittedvalues
    

    NEUAEC_ar1_w = ARIMA(y_NEUAEC_w, order = (1,0,0))
    NEUAEC_ar1_w = NEUAEC_ar1_w.fit()


    NEUAEC_ar1_pred_w = NEUAEC_ar1_w.predict(start = 536, end = 543)


    NEUAEC_ar1_2_w = SARIMAX(y_NEUAEC_w, order = (5,0,0))
    NEUAEC_ar1_2_w = NEUAEC_ar1_2_w.fit()
    y_pred_NEUAEC_w = NEUAEC_ar1_2_w.get_forecast(len(North_EU_Amer_East_Co_Train_w.index)+1)
    y_pred_df_NEUAEC_w = y_pred_NEUAEC_w.conf_int(alpha = 0.05)
    y_pred_df_NEUAEC_w['Predictions'] = NEUAEC_ar1_2_w.predict(start = y_pred_df_NEUAEC_w.index[0], end = y_pred_df_NEUAEC_w.index[-1])


    NEUAEC_ar_pred_w = pd.DataFrame(NEUAEC_ar1_pred_w)
    North_EU_Amer_East_Co_Test_w['pred_1'] = NEUAEC_ar_pred_w.values[1:8]
    rmspe1_NEUAEC_1_w = np.sqrt(np.sum(North_EU_Amer_East_Co_Test_w.iloc[:, 0].subtract(North_EU_Amer_East_Co_Test_w.iloc[:, 1])**2)/7)

    pred_2_w = pd.DataFrame(y_pred_df_NEUAEC_w['Predictions'])
    North_EU_Amer_East_Co_Test_w['pred_2_w'] = pred_2_w.values[1: 8]
    rmspe1_NEUAEC_2_w = np.sqrt(np.sum(North_EU_Amer_East_Co_Test_w.iloc[:, 0].subtract(North_EU_Amer_East_Co_Test_w.iloc[:, 2])**2)/7)

   
    return y_NEUAEC_w,NEUAEC_ar1_pred_w

def Neaec_m(df):
    North_EU_Amer_East_Co_Train_m = df[0:513]
    North_EU_Amer_East_Co_Test_m = df[513:543]
    North_EU_Amer_East_Co_Train_m = np.log(North_EU_Amer_East_Co_Train_m)
    North_EU_Amer_East_Co_Test_m = np.log(North_EU_Amer_East_Co_Test_m)
    North_EU_Amer_East_Co_Train_m = North_EU_Amer_East_Co_Train_m.reset_index()
    y_NEUAEC_m = North_EU_Amer_East_Co_Train_m[' Price ']
    x_NEUAEC_m = range(0, 513)
    NEUAEC_lm_m = smf.ols('y_NEUAEC_m ~ x_NEUAEC_m', data = North_EU_Amer_East_Co_Train_m)
    NEUAEC_lm_m = NEUAEC_lm_m.fit()
    NEUAEC_lm_m.summary()
    
    
    
    NEUAEC_ar_m2 = ARIMA(y_NEUAEC_m, order = (4, 0, 4))
    NEUAEC_ar_m2 = NEUAEC_ar_m2.fit()
    
    
    NEUAEC_ar_pred_m2 = NEUAEC_ar_m2.predict(start = 513, end = 543)
    
    
    NEUAEC_ar_2_m2 = SARIMAX(y_NEUAEC_m, order = (1, 0, 1))
    NEUAEC_ar_2_m2 = NEUAEC_ar_2_m2.fit()
    y_pred_NEUAEC_m2 = NEUAEC_ar_2_m2.get_forecast(len(North_EU_Amer_East_Co_Test_m.index)+1)
    y_pred_df_NEUAEC_m2 = y_pred_NEUAEC_m2.conf_int(alpha = 0.05) 
    y_pred_df_NEUAEC_m2["Predictions"] = NEUAEC_ar_2_m2.predict(start = y_pred_df_NEUAEC_m2.index[0], end = y_pred_df_NEUAEC_m2.index[-1])
    
    
    NEUAEC_ar_pred_m_2 = pd.DataFrame(NEUAEC_ar_pred_m2)
    North_EU_Amer_East_Co_Test_m['pred_1'] = NEUAEC_ar_pred_m_2.values[1:31]
    rmspe_NEUAEC_1_m2 = np.sqrt(np.sum(North_EU_Amer_East_Co_Test_m.iloc[:, 0].subtract(North_EU_Amer_East_Co_Test_m.iloc[:, 1])**2)/30)
    
    y_pred_df_NEUAEC_m2.columns.values
    pred_2_m2 = pd.DataFrame(y_pred_df_NEUAEC_m2['Predictions'])
    North_EU_Amer_East_Co_Test_m['pred_2_m'] = pred_2_m2.values[1:31]
    rmspe__NEUAEC_2_m2 = np.sqrt(np.sum(North_EU_Amer_East_Co_Test_m.iloc[:, 0].subtract(North_EU_Amer_East_Co_Test_m.iloc[:, 2])**2)/30)
    return y_NEUAEC_m,NEUAEC_ar_pred_m2


def Neaec_6m(df):
    North_EU_Amer_East_Co_Train_6m = df[0:363]
    North_EU_Amer_East_Co_Test_6m = df[363:543]
    North_EU_Amer_East_Co_Train_6m = np.log(North_EU_Amer_East_Co_Train_6m)
    North_EU_Amer_East_Co_Test_6m = np.log(North_EU_Amer_East_Co_Test_6m)



    North_EU_Amer_East_Co_Train_6m = North_EU_Amer_East_Co_Train_6m.reset_index()
    y_NEUAEC_6m = North_EU_Amer_East_Co_Train_6m[' Price ']
    x_NEUAEC_6m = range(0, 363)
    NEUAEC_lm_6m = smf.ols('y_NEUAEC_6m ~ x_NEUAEC_6m', data = North_EU_Amer_East_Co_Train_6m)
    NEUAEC_lm_6m = NEUAEC_lm_6m.fit()
    NEUAEC_lm_6m.summary()

    

    NEUAEC_stationary_6m = y_NEUAEC_6m - NEUAEC_lm_6m.fittedvalues
    

    NEUAEC_ar_6m = ARIMA(y_NEUAEC_6m, order = (1, 0, 0))
    NEUAEC_ar_6m = NEUAEC_ar_6m.fit()

    

    NEUAEC_ar_pred_6m = NEUAEC_ar_6m.predict(start = 363, end = 543)

    

    NEUAEC_ar_2_6m = SARIMAX(y_NEUAEC_6m, order = (1, 0, 0))
    NEUAEC_ar_2_6m = NEUAEC_ar_2_6m.fit()
    y_pred_NEUAEC_6m = NEUAEC_ar_2_6m.get_forecast(len(North_EU_Amer_East_Co_Test_6m.index)+1)
    y_pred_df_NEUAEC_6m = y_pred_NEUAEC_6m.conf_int(alpha = 0.05) 
    y_pred_df_NEUAEC_6m["Predictions"] = NEUAEC_ar_2_6m.predict(start = y_pred_df_NEUAEC_6m.index[0], end = y_pred_df_NEUAEC_6m.index[-1])

   
    NEUAEC_ar_pred6_m = pd.DataFrame(NEUAEC_ar_pred_6m)
    North_EU_Amer_East_Co_Test_6m['pred_1'] = NEUAEC_ar_pred_6m.values[1:181]
    rmspe__NEUAEC_1_6m = np.sqrt(np.sum(North_EU_Amer_East_Co_Test_6m.iloc[:, 0].subtract(North_EU_Amer_East_Co_Test_6m.iloc[:, 1])**2)/180)

    y_pred_df_NEUAEC_6m.columns.values
    pred_2_6m = pd.DataFrame(y_pred_df_NEUAEC_6m['Predictions'])
    North_EU_Amer_East_Co_Test_6m['pred_2_6m'] = pred_2_6m.values[1:181]
    rmspe_NEUAEC_2_6m = np.sqrt(np.sum(North_EU_Amer_East_Co_Test_6m.iloc[:, 0].subtract(North_EU_Amer_East_Co_Test_6m.iloc[:, 2])**2)/180)
    return y_NEUAEC_6m,NEUAEC_ar_pred_6m

   

