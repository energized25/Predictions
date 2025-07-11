#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 25 13:49:36 2025

@author: avh
"""
 
from read_functions import read_JAO_prices, imp_EPEX_Prices, imp_trade_prices, read_Wind_PV_data, read_market_data, read_X #, net_prc, read_generation_forecast, read_Wind_PV_data
from parameters import get_startdate, get_enddate
from functions import merge_exaa, merge_dfs, merge_dfs_cont, read_forecasts_hourly, get_PV_hourly, get_WINDtotal_hourly, W_PV_cont, MLM
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# Define country
Country='DE'

Df_PRC_2 = imp_trade_prices(get_startdate(), get_enddate(), Country,True) #True reads EXAA 10:15 data (Sequence 2)

Df_PRC_1 = imp_trade_prices(get_startdate(), get_enddate(), Country) # reads Sequence 1 (12:00 auction)

#As Seq 2 is traded before seq 1, pos values mean seq 1 > seq 2

Df_PRC_cont=merge_exaa(Df_PRC_1, Df_PRC_2)
#Date, Hour, Net Price, PV, Wind

Df_PRC_cont=W_PV_cont(Df_PRC_cont)


#y_pred = MLM (Df_PRC_cont)


# %%

#Write a module to simulate the decisions and profits for a longer period to estimate gains EXAA DE , AT
#include budget needed for trades

#Define a series of days
#Build X for each day based on generation data
#Each time filter Df_PRC_cont until day before and predict
#Write a continous df with long / short signal for every hour in range
#store gains in an array and sum up
#for de and for AT
#use for optimization of model parameters



#Why not separate Wind on / off
# from new transparency platform: separate files for wind, solar

#- start, end date
# create X_Data for every day
# create y_pred for every day, extract trading decision
# read out Df_PRC_cont , multiply with trading decision and save into a Dataframe
# analyze and optimize total result of data frame


#improve, if 24 separate models for each hour? 

start_date=pd.Timestamp("2025-04-01")
end_date=pd.Timestamp("2025-04-30")
Q=10 #traded quantity per delivery hour

def simu (start_date, end_date, Df_PRC_cont):
    
    
    #ML based prediction model for net prices

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
    #copy=Df_PRC_cont
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Df_PRC_cont[["PV", "Wind"]] = scaler.fit_transform(Df_PRC_cont[["PV", "Wind"]])
    
    #dummies
    
    Df_PRC_cont = pd.get_dummies(Df_PRC_cont, columns=["Season", "Weekday"])
    
    Df_PRC_cont=Df_PRC_cont.rename(columns={"Weekday_0":"Monday","Weekday_1":"Tuesday","Weekday_2":"Wednesday","Weekday_3":"Thursday","Weekday_4":"Friday","Weekday_5":"Saturday","Weekday_6":"Sunday"})
    
    #Train-Test-Split
    from sklearn.model_selection import train_test_split
    #X = Df_PRC_cont.drop(columns=["Net Price"])
    #y = Df_PRC_cont["Net Price"]
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    Df_PRC_cont_filtered = Df_PRC_cont[Df_PRC_cont.index <= "2025-01-01"].copy()

    X = Df_PRC_cont_filtered.drop(columns=["Net Price"])
    y = Df_PRC_cont_filtered["Net Price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    #Model training
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    

    current_date = start_date
    # TradeD = None
    # while current_date <= end_date:
    #     new_col_name = f"{current_date}"
    #     print(current_date.strftime("%Y-%m-%d"))  # Ausgabe des Datums
    #     # create X_Data
    #     #X_Data=read_X()
    #     X_Data = Df_PRC_cont[Df_PRC_cont.index.date == current_date.date()]
    #     X_Data=X_Data.drop(columns=["Net Price"])
    #     X_Data = X_Data.loc[:, ~X_Data.columns.str.contains('^Unnamed')]
    #     #X skalieren
    #     #new_data_scaled = scaler.transform(X_Data[["PV","Wind"]])
    #     #X_Data_scaled=X_Data
    #     #X_Data_scaled["PV"]=new_data_scaled[:, 0]
    #     #X_Data_scaled["Wind"]=new_data_scaled[:, 1]
    #     #X_Data_filtered = X_Data_scaled.iloc[range(0, 96, 4)]
    #     #X_Data_filtered.insert(3, "Hour", range(1, len(X_Data_filtered) + 1))  # Fügt "Hour" als 4. Spalte hinzu
    #     # create y_pred
    #     y_pred = model.predict(X_Data)
    #     new_column = np.where(y_pred > 0, 1, np.where(y_pred < 0, -1, 0))
    #     if TradeD is None:
    #         TradeD = pd.DataFrame(new_column, columns=[new_col_name])  # Erstinitialisierung
    #     else:
    #         TradeD[new_col_name] = new_column  # Neue Spalte hinzufügen
  
    #     current_date += timedelta(days=1)

    TradeD = pd.DataFrame(columns=["Signal"])
    while current_date <= end_date:
        print(current_date.strftime("%Y-%m-%d"))  # Ausgabe des Datums
        # create X_Data
        #X_Data=read_X()
        X_Data = Df_PRC_cont[Df_PRC_cont.index.date == current_date.date()]
        X_Data=X_Data.drop(columns=["Net Price"])
        X_Data = X_Data.loc[:, ~X_Data.columns.str.contains('^Unnamed')]
        #X skalieren
        #new_data_scaled = scaler.transform(X_Data[["PV","Wind"]])
        #X_Data_scaled=X_Data
        #X_Data_scaled["PV"]=new_data_scaled[:, 0]
        #X_Data_scaled["Wind"]=new_data_scaled[:, 1]
        #X_Data_filtered = X_Data_scaled.iloc[range(0, 96, 4)]
        #X_Data_filtered.insert(3, "Hour", range(1, len(X_Data_filtered) + 1))  # add "Hour" as 4. Culumn
        # create y_pred
        y_pred = model.predict(X_Data)
        signals = np.where(y_pred > 0, 1, np.where(y_pred < 0, -1, 0))
        timestamps = [current_date + timedelta(hours=i) for i in range(24)]
        new_data = pd.DataFrame({"Timestamp": timestamps, "Signal": signals.flatten()})
        TradeD = pd.concat([TradeD, new_data], ignore_index=True)


  
        current_date += timedelta(days=1)



    return TradeD


TradeD = simu(start_date, end_date, Df_PRC_cont)
TradeD.set_index("Timestamp", inplace=True)

TradeD["Net Price"] = TradeD.index.map(Df_PRC_cont["Net Price"])
TradeD["Result"] = TradeD["Net Price"] * TradeD["Signal"]*Q
sum_result = TradeD["Result"].sum()
print(sum_result)

#IST = Df_PRC_cont[Df_PRC_cont.index.date == start_date.date()]
