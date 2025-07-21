#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 11:28:57 2025

@author: sarahvonhardenberg
"""

from read_functions import imp_bal_prices, imp_trade_prices_quarter, sumup, sim_model
from parameters import get_startdate, get_enddate
from functions import merge_rebap, add_W_PV

import pandas as pd

Country ="DE"

#import Balancing Prices
reBAP_GER = imp_bal_prices(Country)

#import SDAC Prices
Df_PRC_1 = imp_trade_prices_quarter(get_startdate(), get_enddate(), Country)

Df_rebap_net=merge_rebap(Df_PRC_1, reBAP_GER)

Df_res=add_W_PV(Df_rebap_net)

Df_PRC_cont = Df_res.iloc[:, 2:6]

start_date=pd.Timestamp("2025-04-01")
end_date=pd.Timestamp("2025-05-25")
Q=10 #traded quantity per 15 minutes block

Df_PRC_cont = Df_PRC_cont.rename(columns={
    "Generation - Solar [MW] Day Ahead/ BZN|DE-LU": "PV",
    "Generation - Wind Offshore [MW] Day Ahead/ BZN|DE-LU": "Wind OFFSH",
    "Generation - Wind Onshore [MW] Day Ahead/ BZN|DE-LU": "Wind ONSH"
})

model=sim_model(start_date, end_date, Df_PRC_cont)

TradeD=sumup(model,start_date,end_date,Df_PRC_cont)
TradeD.set_index("Timestamp", inplace=True)

Df_PRC_cont=Df_PRC_cont.groupby(Df_PRC_cont.index).first()

TradeD["Net Price"] = TradeD.index.map(Df_PRC_cont["Net Price"])
TradeD["Result"] = TradeD["Net Price"] * TradeD["Signal"]*Q
sum_result = TradeD["Result"].sum()
print(sum_result)


""" Backlog - needed features
We have a function creating an ML model and siumlating trading decisions day by day based on PV and Wind forecast data.
We NEED a function to import a specific forecast for the next day and to export the trading decision
Should the model be trained every run, or exported and saved for a couple of times?
Check how far public API can be used at Entsoe
Need a good data provider for forecasts day ahead (e.g. Energy Quantified - no account as of today)

Whole Scheduleing issue / clearing issue is unclear
--> AT --> web tool
--> DE --> also Web tool available? 

We have no price predictions, only trading decisions (how to set limits? unlimited?)




"""





"""

#Plot

import pandas as pd
import matplotlib.pyplot as plt

# Beispiel: DataFrame mit datetime im Index und Preisen
# df = pd.read_csv('deine_datei.csv', parse_dates=['Datum'], index_col='Datum')

# Plot erstellen
plt.figure(figsize=(12, 6))  # Größe des Plots
plt.plot(Df_rebap_net.index, Df_rebap_net['Net Price'], color='royalblue', linewidth=2, marker='o')

# Titel und Achsenbeschriftungen
plt.title('Net Price über Zeit', fontsize=16)
plt.xlabel('Datum', fontsize=12)
plt.ylabel('Net Price (€)', fontsize=12)

# Schönes Datumsformat auf der x-Achse
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)

# Optional: Hintergrundfarbe
plt.gca().set_facecolor('#f9f9f9')

# Anzeigen
plt.tight_layout()
plt.show()

"""


"""

# OLD CODE - Simu in 1 module (not seperated in 2 steps)
def simu (start_date, end_date, Df_PRC_cont):
    
    
    #ML based prediction model for net prices

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
    Df_PRC_cont[["PV", "Wind OFFSH", "Wind ONSH"]] = scaler.fit_transform(Df_PRC_cont[["PV", "Wind OFFSH", "Wind ONSH"]])
    
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

    TradeD = pd.DataFrame(columns=["Signal"])
    while current_date <= end_date:
        print(current_date.strftime("%Y-%m-%d"))  # Ausgabe des Datums

        X_Data = Df_PRC_cont[Df_PRC_cont.index.date == current_date.date()]
        X_Data=X_Data.drop(columns=["Net Price"])
        X_Data = X_Data.loc[:, ~X_Data.columns.str.contains('^Unnamed')]

        y_pred = model.predict(X_Data)
        signals = np.where(y_pred > 0, 1, np.where(y_pred < 0, -1, 0))
        timestamps = [current_date + timedelta(minutes=i * 15) for i in range(96)]
        new_data = pd.DataFrame({"Timestamp": timestamps, "Signal": signals.flatten()})
        TradeD = pd.concat([TradeD, new_data], ignore_index=True)

        current_date += timedelta(days=1)

    return TradeD


TradeD = simu(start_date, end_date, Df_PRC_cont)
TradeD.set_index("Timestamp", inplace=True)

Df_PRC_cont=Df_PRC_cont.groupby(Df_PRC_cont.index).first()

TradeD["Net Price"] = TradeD.index.map(Df_PRC_cont["Net Price"])
TradeD["Result"] = TradeD["Net Price"] * TradeD["Signal"]*Q
sum_result = TradeD["Result"].sum()
print(sum_result)

"""