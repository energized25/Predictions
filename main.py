#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  3 16:47:38 2025

@author: avh
"""

# !! Take care of data inconsitencies - e.g. JAO CH AT 2024 "Auction canceled", Time Shift??
#maybe calculate without transformations? 
# delete values and, if demanded, create dedicated function for the extra hour

#how to deal with zero quantities jao? (no prices)

# Write function to evaluate 10:15 auctions

#still problems with CH DE -- DE no data before 2018 - problem solved, will leave out missing data

#build wind filter and combined filter - ok

#write a filter to exclude certain days oder years eg 2022

#PV / Solar values for other countries - austria is avaiable early on newmarkettransparency
#diversify variables (separate: on and offshore wind. Function to read in downloaded files)
#other variable ideas: ticket prices (bus, rail, air), large events, stock prices

#exaa de , at, nl 10:15 auctions!!
#balancing power prices


# general to dos:
    
    #integrate more data sources (PV, Wind other countries, Power stations availability, Temperature, Holidays, ...)
    #Build decision making and bidding module JAO
    #Build JAO Read in module and exchange bid creation module (s)
    #build scheduling modules
    #optional: build result tracking and record module
    #build second model with continous data

from read_functions import read_JAO_prices, imp_EPEX_Prices, imp_trade_prices, read_Wind_PV_data, read_market_data, read_X #, net_prc, read_generation_forecast, read_Wind_PV_data
from parameters import get_startdate, get_enddate
from functions import merge_dfs, merge_dfs_cont, read_forecasts_hourly, get_PV_hourly, get_WINDtotal_hourly, W_PV_cont
import pandas as pd
import numpy as np


# Define involved countries
Country_1='CH' #out country
Country_2='DE' #in country


if Country_2 != "IT":
    Df_Jao_in = read_JAO_prices(get_startdate(), get_enddate() , Country_2, Country_1)

Df_Jao_out = read_JAO_prices(get_startdate(), get_enddate() , Country_1, Country_2)

Df_PRC_1 = imp_trade_prices(get_startdate(), get_enddate(), Country_1)

Df_PRC_2 = imp_trade_prices(get_startdate(), get_enddate(), Country_2)


"""
# %%

# This cell creates data in  24 hours view - nice to get feeling for data but not well suited for data analysis (unless working with excel...)


Df_PRC_Res = merge_dfs(Df_PRC_1, Df_PRC_2, Df_Jao_out)

#next steps: aus Df_Res, Wochentagen, Supplydaten den Triggergrund-DF erstellen
#bestehende Methode auch für andere Grenzen und Richtungen erweitern
# Jao downloads / alternative Möglichkeit?
#PV_WIND_FC = read_forecasts_hourly()



PV_h = get_PV_hourly()
WI_h = get_WINDtotal_hourly()

#Erstellt einen Dataframe mit drei Blöcken: Nettopreise, PV-Daten, Wind-Daten
Df_merged = Df_PRC_Res.merge(PV_h, on="Date", how="inner")\
                          .merge(WI_h, on="Date", how="inner")


Df_merged['Weekday'] = pd.to_datetime(Df_merged.index).day_name()
Df_merged.index = pd.to_datetime(Df_merged.index)

from functions import filter_dataframe

# Explanation filter_dataframe function
# Filters data frame, for day, week or season. 
    #day = {'A': 5, 'B': 10, 'C': 15, 'D': 20, 'E': 30, 'F': 50, 'G': 70}
    #week = {  # Wochentage 'A': Monday, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': Sunday}
    #season = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 10, 'G': None} current month, number of years back
    #PV compares mean of rows 13_y bis 15_y to: {'A': 0, 'B': 10000, 'C': 20000, 'D': 30000, 'E': 40000, 'F': 50000, 'G': 60000} 'till + 10000
    #Wind compares mean of rows to {'A': 0, 'B': 10000, 'C': 20000, 'D': 30000, 'E': 40000, 'F': 50000, 'G': 60000}  'till + 10000
#final = filter_dataframe(Df_merged, 'Wind', 'G')


final = filter_dataframe(Df_merged, 'Wind', 'C')

fin_day = filter_dataframe(final, 'week', 'G')

"""

# %%

# This cell creates data in continous format (no 24 hours view)

Df_PRC_cont=merge_dfs_cont(Df_PRC_1, Df_PRC_2, Df_Jao_out)

Df_PRC_cont=W_PV_cont(Df_PRC_cont)



# %%



#ML based prediction model for net prices - OLD but working

Df_PRC_cont=Df_PRC_cont.drop(columns=["Date","Hour"])

#Df_PRC_cont["DateTime"]=Df_PRC_cont.index
#Df_PRC_cont["DateTime"]=pd.to_datetime(Df_PRC_cont["DateTime"])

#seasons
Df_PRC_cont["Weekday"] = Df_PRC_cont.index.dayofweek  # 0 = Montag, 6 = Sonntag
Df_PRC_cont["Month"] = Df_PRC_cont.index.month
Df_PRC_cont["Hour"] = Df_PRC_cont.index.hour+1
Df_PRC_cont["Season"] = Df_PRC_cont.index.month.map({12: "Winter", 1: "Winter", 2: "Winter",
                                3: "Spring", 4: "Spring", 5: "Spring",
                                6: "Summer", 7: "Summer", 8: "Summer",
                                9: "Autumn", 10: "Autumn", 11: "Autumn"})

#holidays

from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar() #integrate sth more suitable
holidays = cal.holidays(start=Df_PRC_cont.index.min(), end=Df_PRC_cont.index.max())
hours = range(24)  # Stunden von 0 bis 23
holiday_hours = [pd.Timestamp(f"{date} {hour}:00:00") for date in holidays for hour in range(24)]

Df_PRC_cont["Holiday"] = Df_PRC_cont.index.isin(holiday_hours).astype(int)


#normalize variables
copy=Df_PRC_cont

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Df_PRC_cont[["PV", "Wind"]] = scaler.fit_transform(Df_PRC_cont[["PV", "Wind"]])

#dummies

Df_PRC_cont = pd.get_dummies(Df_PRC_cont, columns=["Season", "Weekday"])

Df_PRC_cont=Df_PRC_cont.rename(columns={"Weekday_0":"Monday","Weekday_1":"Tuesday","Weekday_2":"Wednesday","Weekday_3":"Thursday","Weekday_4":"Friday","Weekday_5":"Saturday","Weekday_6":"Sunday"})

#Train-Test-Split
from sklearn.model_selection import train_test_split
X = Df_PRC_cont.drop(columns=["Net Price"])
y = Df_PRC_cont["Net Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Modelltraining
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#predict


y_pred = model.predict(X_test)

#error anaylsis

from sklearn.metrics import mean_absolute_error
print("MAE:", mean_absolute_error(y_test, y_pred))



#build X

X_Data=read_X()
X_Data=X_Data.drop(columns=["Wind on","Wind off"])
X_Data = X_Data.loc[:, ~X_Data.columns.str.contains('^Unnamed')]

#scale X

new_data_scaled = scaler.transform(X_Data[["PV","Wind"]])
X_Data_scaled=X_Data
X_Data_scaled["PV"]=new_data_scaled[:, 0]
X_Data_scaled["Wind"]=new_data_scaled[:, 1]

X_Data_filtered = X_Data_scaled.iloc[range(0, 96, 4)]
X_Data_filtered.insert(3, "Hour", range(1, len(X_Data_filtered) + 1))  # Fügt "Hour" als 4. Spalte hinzu

#predict with real data

y_pred = model.predict(X_Data_filtered)


"""
# %%


#ML based prediction model for net prices - LSTM Model - does not work

Df_PRC_cont=Df_PRC_cont.drop(columns=["Date"])

#Df_PRC_cont["DateTime"]=Df_PRC_cont.index
#Df_PRC_cont["DateTime"]=pd.to_datetime(Df_PRC_cont["DateTime"])

#seasons
Df_PRC_cont["Weekday"] = Df_PRC_cont.index.dayofweek  # 0 = Montag, 6 = Sonntag
Df_PRC_cont["Month"] = Df_PRC_cont.index.month
Df_PRC_cont["Hour"] = Df_PRC_cont.index.hour

#Df_PRC_cont["Season"] = Df_PRC_cont["Month"].map({12: "Winter", 1: "Winter", 2: "Winter",
#                                3: "Spring", 4: "Spring", 5: "Spring",
#                                6: "Summer", 7: "Summer", 8: "Summer",
#                                9: "Autumn", 10: "Autumn", 11: "Autumn"})

#holidays

from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()
holidays = cal.holidays(start=Df_PRC_cont.index.min(), end=Df_PRC_cont.index.max())
hours = range(24)  # Stunden von 0 bis 23
holiday_hours = [pd.Timestamp(f"{date} {hour}:00:00") for date in holidays for hour in range(24)]

Df_PRC_cont["Holiday"] = Df_PRC_cont.index.isin(holiday_hours).astype(int)


#normalize variables
copy=Df_PRC_cont

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
Df_PRC_cont[["Net Price", "PV", "Wind"]] = scaler.fit_transform(Df_PRC_cont[["Net Price", "PV", "Wind"]])

#sequence creation

def create_sequences(data, target_col, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i : i + seq_length].values)  # Sequenz aus vorherigen Zeitpunkten
        #y.append(data.iloc[i + seq_length][target_col]) # Zielwert danach
        if i + seq_length < len(data):  # Prüfen, ob Zielindex gültig ist
            y.append(data.iloc[i + seq_length][target_col])  
    return np.array(X), np.array(y)

seq_length = 24  # last 24h 
X, y = create_sequences(Df_PRC_cont.drop(columns=["Net Price"]), "Net Price", seq_length)


split = int(0.8 * len(X))  # 80% Training, 20% Test
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


#build LSTM Model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, X.shape[2])),
    LSTM(50, return_sequences=False),
    Dense(1)  # Eine Ausgabe für die Preisprognose
])

model.compile(optimizer="adam", loss="mse")


#start training
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

#predict
y_pred = model.predict(X_test)


#rescale

y_pred_rescaled = scaler.inverse_transform(y_pred)

"""