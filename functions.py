#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  3 23:10:12 2025

@author: sarahvonhardenberg
"""

from read_functions import read_Wind_PV_data, read_X
import pandas as pd
import pytz
from datetime import datetime, timedelta
from parameters import get_season_filter_settings

def merge_dfs(Df_PRC_1,Df_PRC_2,Df_Jao_out):

    import pandas as pd
    
    # Stelle sicher, dass die Spalte in datetime konvertiert ist
    Df_PRC_1["MTU (CET/CEST)"] = pd.to_datetime(Df_PRC_1["MTU (CET/CEST)"])
    Df_PRC_2["MTU (CET/CEST)"] = pd.to_datetime(Df_PRC_2["MTU (CET/CEST)"])
    
    #Df_JAO_out["MTU (CET/CEST)"] = pd.to_datetime(Df_JAO_out["MTU (CET/CEST)"])
    # Erzeuge eine neue Spalte für das Datum (ohne Uhrzeit)
    Df_PRC_1["Date"] = Df_PRC_1["MTU (CET/CEST)"].dt.date
    Df_PRC_1["Hour"] = Df_PRC_1["MTU (CET/CEST)"].dt.hour+1
    
    Df_PRC_2["Date"] = Df_PRC_2["MTU (CET/CEST)"].dt.date
    Df_PRC_2["Hour"] = Df_PRC_2["MTU (CET/CEST)"].dt.hour+1
    
    #Df_PRC_2["TS_CET"] = Df_PRC_2["MTU (CET/CEST)"].apply(convert_to_cet) NOT WORKING 100%
    
   # Df_PRC_1 = remove_duplicate_3(Df_PRC_1) - works
   # Df_PRC_2 = remove_duplicate_3(Df_PRC_2) - works

 
    
    Df_Jao_out["Date"] = pd.to_datetime(Df_Jao_out["Market period start"], format='%d/%m/%Y').dt.strftime('%Y-%m-%d') 
    Df_Jao_out["Hour"] = pd.to_datetime(Df_Jao_out["TimeTable"].str[:5], format='%H:%M').dt.hour+1
    
    Df_Jao_out=remove_duplicate_3(Df_Jao_out)
    Df_PRC_1=remove_duplicate_3(Df_PRC_1)
    Df_PRC_2=remove_duplicate_3(Df_PRC_2)

    #Filter out "Auction canceled" events
    Df_Jao_out = Df_Jao_out[Df_Jao_out["Additional information"] != "Auction cancelled"]
   # Df_Jao_out = remove_duplicate_3(Df_Jao_out) GEHT NICHT - WARUM?

    
    
    # Wandle den DataFrame um: Eine Zeile pro Tag, 24 Spalten für die Stunden
    Df_PRC_1_trans = Df_PRC_1.pivot(index="Date", columns="Hour", values="Day-ahead Price (EUR/MWh)")
    Df_PRC_2_trans = Df_PRC_2.pivot(index="Date", columns="Hour", values="Day-ahead Price (EUR/MWh)")
    Df_Jao_out_trans = Df_Jao_out.pivot(index="Date", columns="Hour", values="Price (€/MWH)")
    
    
    #Df_Jao_out.index = Df_Jao_out.index.astype(str).str[:10]
    #Df_Jao_out.rename_axis("Date", inplace=True)
    
    Df_Jao_out_trans=Df_Jao_out_trans.astype(float)
    
    Df_Jao_out_trans.index = Df_Jao_out_trans.index.astype(str)
    Df_PRC_1_trans.index = Df_PRC_1_trans.index.astype(str)
    Df_PRC_2_trans.index = Df_PRC_2_trans.index.astype(str)
    
    # Zunächst alle drei DataFrames zusammenführen, um nur gemeinsame Einträge zu behalten
    Df_Result = Df_PRC_2_trans.merge(Df_PRC_1_trans, on="Date", how="inner")\
                              .merge(Df_Jao_out_trans, on="Date", how="inner")
                              
                              # Berechnung: Jede Spalte - die 24. Spalte danach - die nächste 24. Spalte danach
    Df_Result_fin=pd.DataFrame(columns=list(range(1, 25)))
    for i in range(0, 24):
        Df_Result_fin.iloc[:,i] = Df_Result.iloc[:,i] - Df_Result.iloc[:,i+24] - Df_Result.iloc[:,i+48]

    return Df_Result_fin




import pandas as pd

def remove_duplicate_3(df):
    """
    Entfernt die zweite Instanz von aufeinanderfolgenden "3"-Stunden im Oktober.
    
    :param df: Pandas DataFrame mit Spalten "Date" (YYYY-MM-DD) und "Hour" (1-24)
    :return: Bereinigter DataFrame
    """
    #this function is to solve the time shift problem in october
    
    # Konvertiere "Date" in datetime-Format
    #df["Date"] = pd.to_datetime(df["Date"])

    # Finde doppelte 3er-Stunden in Oktober
    #indices_to_drop = []
    #for i in range(len(df) - 1):
    #    if df.loc[i, "Hour"] == 3 and df.loc[i + 1, "Hour"] == 3:
    #        if df.loc[i, "Date"].month == 10:
    #            indices_to_drop.append(i + 1)  # Zweite Instanz merken
    
    indices_to_drop = df.index[(df['Hour'] == 3) & (df['Hour'].shift() == 3)]

    # Entferne die zweite Instanz der doppelten Stunde "3" im Oktober
    df = df.drop(indices_to_drop)
    
    return df

def read_forecasts_hourly():

    # Read Generation forecast
    PV_Wind_FC = read_Wind_PV_data()



    # Kopiere den ursprünglichen DataFrame
    PV_Wind_FC_copy = PV_Wind_FC.copy()

    # Extrahiere den Startzeitstempel aus der ersten Spalte
    PV_Wind_FC_copy['timestamp_start'] = PV_Wind_FC_copy.iloc[:, 0].str.split(" - ").str[0]
    PV_Wind_FC_copy['timestamp_start'] = pd.to_datetime(PV_Wind_FC_copy['timestamp_start'], format='%d.%m.%Y %H:%M')

    # Lösche die ursprüngliche Zeitstempelspalte (erste Spalte)
    PV_Wind_FC_copy.drop(columns=[PV_Wind_FC_copy.columns[0]], inplace=True)

    # Behalte nur die erste Zeile jeder Stunde, indem die Index-Duplikate entfernt werden
    PV_Wind_FC_copy = PV_Wind_FC_copy.set_index('timestamp_start')
    PV_Wind_FC_hourly = PV_Wind_FC_copy[~PV_Wind_FC_copy.index.floor('h').duplicated(keep='first')]
    return PV_Wind_FC_hourly


def PV_WI_structure(PV_WIND_FC):

    PV_WI_copy = PV_WIND_FC
    PV_WI_copy.index = pd.to_datetime(PV_WI_copy.index)
    PV_WI_copy["Date"] = PV_WI_copy.index
    PV_WI_copy["Hour"] = PV_WI_copy.index
    PV_WI_copy["Date"] = PV_WI_copy["Date"].dt.date
    PV_WI_copy["Hour"] = PV_WI_copy["Hour"].dt.hour+1
    return PV_WI_copy


def get_PV_hourly():
   PV_WIND_FC = read_forecasts_hourly()
   PV_WI_copy = PV_WI_structure(PV_WIND_FC)
   PV_hourly = PV_WI_copy.pivot(index="Date", columns="Hour", values="Generation - Solar [MW] Day Ahead/ BZN|DE-LU")
   PV_hourly.index = PV_hourly.index.astype(str)   
   return PV_hourly

def get_WINDtotal_hourly():
    PV_WIND_FC = read_forecasts_hourly()
    PV_WI_copy = PV_WI_structure(PV_WIND_FC)
    
    WI_ons_hourly = PV_WI_copy.pivot(index="Date", columns="Hour", values="Generation - Wind Onshore [MW] Day Ahead/ BZN|DE-LU")
    WI_ofs_hourly = PV_WI_copy.pivot(index="Date", columns="Hour", values="Generation - Wind Offshore [MW] Day Ahead/ BZN|DE-LU")
    WI_total_hourly = WI_ons_hourly+WI_ofs_hourly
    WI_total_hourly.index = WI_total_hourly.index.astype(str)
    return WI_total_hourly

#WI_ons_hourly = PV_WI_copy.pivot(index="Date", columns="Hour", values="Generation - Wind Onshore [MW] Day Ahead/ BZN|DE-LU")
#WI_ofs_hourly = PV_WI_copy.pivot(index="Date", columns="Hour", values="Generation - Wind Offshore [MW] Day Ahead/ BZN|DE-LU")
#WI_total_hourly = WI_ons_hourly+WI_ofs_hourly


def filter_dataframe(net_prices, typee, count):
    # Mapping für Anzahl der Einträge und Wochentage
    count_map_day = {'A': 5, 'B': 10, 'C': 15, 'D': 20, 'E': 30, 'F': 50, 'G': 70}
    count_map_week = {  # Wochentage
        'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6
    }
    year_map_season = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 10, 'G': None}
    count_map_PV = {'A': 0, 'B': 10000, 'C': 20000, 'D': 30000, 'E': 40000, 'F': 50000, 'G': 60000}
    count_map_Wind = {'A': 0, 'B': 10000, 'C': 20000, 'D': 30000, 'E': 40000, 'F': 50000, 'G': 60000}
    if typee not in ["day", "week", "season", "PV", "Wind"]:
        return None
    net_prices = net_prices[~net_prices.index.isin(["Mean per hour", "Variance"])]
    # Wochentag-Spalte hinzufügen
   # net_prices['Weekday'] = net_prices.index.day_name()
    
    if typee == "day":
        # Anzahl der letzten Einträge für den Tag
        num_entries = count_map_day.get(count, 0)
        result = net_prices.tail(num_entries)
    
    elif typee == "week":
        # Filtern nach Wochentag und neuesten 25 Einträgen
        weekday = count_map_week.get(count, None)
        if weekday is not None:
            net_prices.index = pd.to_datetime(net_prices.index)
            filtered = net_prices[net_prices.index.dayofweek == weekday]
            #filtered = net_prices[net_prices ["Weekday"] == weekday]
            result = filtered.tail(25)
        else:
            result = None
    
    elif typee == "season":
        # Filtern nach Monaten. Es wird nach dem aktuellen Monat gefiltert, das kann man sicher schlauer machen.
        
        ran=get_season_filter_settings()
        today = datetime.now()
        current_month = today.month
        years_back = year_map_season.get(count, None)
        result = filter_dates_per_year(net_prices, ran)
        if years_back is not None:
            start_date = today - timedelta(days=years_back*365)
            #filtered = net_prices[
            #    (net_prices.index.month == current_month) & 
            #    (net_prices.index >= start_date)
            #]
            filtered = result[
                (result.index >= start_date)
            ]
            #result = filtered
            
        #else:
            #result = net_prices[net_prices.index.month == current_month]
    
    elif typee == "Wind":
        if count_map_Wind is not None:
            num_cols = 3 * 24
            net_prices.iloc[:, :num_cols] = net_prices.iloc[:, :num_cols].apply(pd.to_numeric, errors="coerce")
            net_prices = net_prices.dropna()
            x = float(count_map_Wind.get(count, None))
            net_prices["Avg"]=net_prices.iloc[:,48:72].mean(axis=1)
            result = net_prices[(net_prices["Avg"] >= x) & (net_prices["Avg"] <= x + 10000)]
        else:
            result = None          
    
    elif typee == "PV":
        # Filtern nach PV und neuesten 25 Einträgen
        if count_map_PV is not None:
            num_cols = 3 * 24
            net_prices.iloc[:, :num_cols] = net_prices.iloc[:, :num_cols].apply(pd.to_numeric, errors="coerce")
            net_prices = net_prices.dropna()
            x = float(count_map_PV.get(count, None))
            net_prices["Avg"]=net_prices[["13_y", "14_y", "15_y"]].mean(axis=1)
            result = net_prices[(net_prices["Avg"] >= x) & (net_prices["Avg"] <= x + 10000)]
        else:
            result = None    
    
    
# Anzahl der relevanten Spalten
    num_cols = 24  # Anpassung nötig, falls abc weniger als 24 Spalten hat

# Durchschnitt über alle Zeilen berechnen (nur für die ersten 24 Spalten)
    mean_values = result.iloc[:, :num_cols].mean()
    var_values = result.iloc[:, :num_cols].var()

# Neue Zeile hinzufügen
    result.loc["Mean per hour"] = mean_values
    result.loc["Variance"] = var_values

    return result


def convert_to_cet(timestamp_str):
    tz_europe = pytz.timezone("Europe/Zurich")  # Ursprüngliche Zeitzone mit Sommerzeit
    tz_cet = pytz.timezone("CET")  # Zielzeitzone ohne Sommerzeit
    
    dt = pd.Timestamp(timestamp_str)
    
    # Zeitzone setzen (ambiguous=True für doppelte Stunden im Herbst)
    dt_localized = dt.tz_localize(tz_europe, ambiguous="NaT")  # ambiguous="NaT" vermeidet Fehler
    dt_cet = dt_localized.tz_convert(tz_cet)
    
    return dt_cet



def filter_dates_per_year(df, ran):
    # Morgen bestimmen
    tomorrow = datetime.today().date() + timedelta(days=1)

    # Extrahiere den Tag und Monat für den Vergleich über alle Jahre
    start_range = (tomorrow - timedelta(days=ran)).strftime("%m-%d")
    end_range = (tomorrow + timedelta(days=ran)).strftime("%m-%d")

    # Verwende `apply()` für den Vergleich
    filtered_df = df[df.index.to_series().apply(lambda x: start_range <= x.strftime("%m-%d") <= end_range)]

    return filtered_df


# %%

def merge_dfs_cont(Df_PRC_1,Df_PRC_2,Df_Jao_out):
    
    # ensure datetime format
    Df_PRC_1["MTU (CET/CEST)"] = pd.to_datetime(Df_PRC_1["MTU (CET/CEST)"])
    Df_PRC_2["MTU (CET/CEST)"] = pd.to_datetime(Df_PRC_2["MTU (CET/CEST)"])
    
    Df_PRC_1["Date"] = Df_PRC_1["MTU (CET/CEST)"].dt.date
    Df_PRC_1["Hour"] = Df_PRC_1["MTU (CET/CEST)"].dt.hour+1
    
    Df_PRC_2["Date"] = Df_PRC_2["MTU (CET/CEST)"].dt.date
    Df_PRC_2["Hour"] = Df_PRC_2["MTU (CET/CEST)"].dt.hour+1
    
    Df_Jao_out["Date"] = pd.to_datetime(Df_Jao_out["Market period start"], format='%d/%m/%Y').dt.strftime('%Y-%m-%d') 
    Df_Jao_out["Hour"] = pd.to_datetime(Df_Jao_out["TimeTable"].str[:5], format='%H:%M').dt.hour+1
    
    Df_Jao_out["MTU (CET/CEST)"] = pd.to_datetime(Df_Jao_out["Market period start"], format="%d/%m/%Y").dt.strftime("%Y-%m-%d") + " " + Df_Jao_out["TimeTable"].str[:5] + ":00"
    Df_Jao_out["MTU (CET/CEST)"] = pd.to_datetime(Df_Jao_out["MTU (CET/CEST)"])
    
    #remove time shift duplicate hour 3
    Df_Jao_out=remove_duplicate_3(Df_Jao_out)
    Df_PRC_1=remove_duplicate_3(Df_PRC_1)
    Df_PRC_2=remove_duplicate_3(Df_PRC_2)

    #Filter out "Auction canceled" events
    Df_Jao_out = Df_Jao_out[Df_Jao_out["Additional information"] != "Auction cancelled"]

    keep=["Date","Hour","Day-ahead Price (EUR/MWh)", "MTU (CET/CEST)"]
    Df_PRC_1_new = Df_PRC_1[keep]
    Df_PRC_2_new = Df_PRC_2[keep]
    keep=["Date","Hour","Price (€/MWH)","MTU (CET/CEST)"]
    Df_Jao_out_new = Df_Jao_out[keep]    
    
    Df_PRC_1_new= Df_PRC_1_new.rename(columns={"Day-ahead Price (EUR/MWh)": "Price C1"})
    Df_PRC_2_new= Df_PRC_2_new.rename(columns={"Day-ahead Price (EUR/MWh)": "Price C2"})
    Df_Jao_out_new= Df_Jao_out_new.rename(columns={"Price (€/MWH)": "Price JAO"})

    #Df_Result = pd.concat([Df_PRC_2_new.set_index("MTU (CET/CEST)"),
    #                   Df_PRC_1_new.set_index("MTU (CET/CEST)"),
    #                   Df_Jao_out_new.set_index("MTU (CET/CEST)")], axis=1).reset_index()
    # pd.concat also keeps incomplete rows


   
    Df_Result = Df_PRC_2_new.merge(Df_PRC_1_new, on="MTU (CET/CEST)", how="inner")\
                              .merge(Df_Jao_out_new, on="MTU (CET/CEST)", how="inner")
    
    Df_Result[["Price C1", "Price C2", "Price JAO"]] = Df_Result[["Price C1", "Price C2", "Price JAO"]].apply(pd.to_numeric, errors="coerce")

    
    Df_Result["Net Price"]=   Df_Result["Price C2"]-   Df_Result["Price C1"] -    Df_Result["Price JAO"]
    keep=["Date","Hour","Net Price","MTU (CET/CEST)"]
    Df_Result = Df_Result[keep]
    return Df_Result
    

def merge_exaa(Df_PRC_1,Df_PRC_2):
    
    # ensure datetime format
    Df_PRC_1["MTU (CET/CEST)"] = pd.to_datetime(Df_PRC_1["MTU (CET/CEST)"])
    Df_PRC_2["MTU (CET/CEST)"] = pd.to_datetime(Df_PRC_2["MTU (CET/CEST)"])
    
    Df_PRC_1["Date"] = Df_PRC_1["MTU (CET/CEST)"].dt.date
    Df_PRC_1["Hour"] = Df_PRC_1["MTU (CET/CEST)"].dt.hour+1
    
    Df_PRC_2["Date"] = Df_PRC_2["MTU (CET/CEST)"].dt.date
    Df_PRC_2["Hour"] = Df_PRC_2["MTU (CET/CEST)"].dt.hour+1
  
    #remove time shift duplicate hour 3
    Df_PRC_1=remove_duplicate_3(Df_PRC_1)
    Df_PRC_2=remove_duplicate_3(Df_PRC_2)

    keep=["Date","Hour","Day-ahead Price (EUR/MWh)", "MTU (CET/CEST)"]
    Df_PRC_1_new = Df_PRC_1[keep]
    Df_PRC_2_new = Df_PRC_2[keep]

    Df_PRC_1_new= Df_PRC_1_new.rename(columns={"Day-ahead Price (EUR/MWh)": "Price C1"})
    Df_PRC_2_new= Df_PRC_2_new.rename(columns={"Day-ahead Price (EUR/MWh)": "Price C2"})
 
    #Df_Result = pd.concat([Df_PRC_2_new.set_index("MTU (CET/CEST)"),
    #                   Df_PRC_1_new.set_index("MTU (CET/CEST)"),
    #                   Df_Jao_out_new.set_index("MTU (CET/CEST)")], axis=1).reset_index()
    # pd.concat also keeps incomplete rows


   
    Df_Result = Df_PRC_2_new.merge(Df_PRC_1_new, on="MTU (CET/CEST)", how="inner")
                       
    #Df_Result[["Price C1", "Price C2"]] = Df_Result[["Price C1", "Price C2"].apply(pd.to_numeric, errors="coerce")]

    
    Df_Result["Net Price"]=   Df_Result["Price C1"]-   Df_Result["Price C2"]
    Df_Result = Df_Result.rename(columns={'Date_x': 'Date', 'Hour_x': 'Hour'})

    
    keep=["Date","Hour","Net Price","MTU (CET/CEST)"]
    Df_Result = Df_Result[keep]
    return Df_Result


def W_PV_cont(Df_PRC_cont):

    PV_WIND_FC = read_forecasts_hourly()
    keep=["Generation - Solar [MW] Day Ahead/ BZN|DE-LU","Generation - Wind Offshore [MW] Day Ahead/ BZN|DE-LU","Generation - Wind Onshore [MW] Day Ahead/ BZN|DE-LU"]
    PV_WIND_FC = PV_WIND_FC[keep]
    PV_WIND_FC["Wind"] = PV_WIND_FC["Generation - Wind Offshore [MW] Day Ahead/ BZN|DE-LU"] + PV_WIND_FC["Generation - Wind Onshore [MW] Day Ahead/ BZN|DE-LU"]
    PV_WIND_FC["MTU (CET/CEST)"] = PV_WIND_FC.index
    
    Df_Result = Df_PRC_cont.merge(PV_WIND_FC, on="MTU (CET/CEST)", how="inner")
    Df_Result = Df_Result.set_index("MTU (CET/CEST)")
    Df_Result["PV"]=Df_Result["Generation - Solar [MW] Day Ahead/ BZN|DE-LU"]
    
    keep=["Date","Hour","Net Price","PV","Wind"]
    Df_Result=Df_Result[keep]
    return Df_Result


def MLM(Df_PRC_cont):

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
    
    
    #y_pred = model.predict(X_test)
    
    #error anaylsis
    
   # from sklearn.metrics import mean_absolute_error
    #print("MAE:", mean_absolute_error(y_test, y_pred))
    
    

    #X bauen
    
    X_Data=read_X()
    X_Data=X_Data.drop(columns=["Wind on","Wind off"])
    X_Data = X_Data.loc[:, ~X_Data.columns.str.contains('^Unnamed')]
    
    #X skalieren
    
    new_data_scaled = scaler.transform(X_Data[["PV","Wind"]])
    X_Data_scaled=X_Data
    X_Data_scaled["PV"]=new_data_scaled[:, 0]
    X_Data_scaled["Wind"]=new_data_scaled[:, 1]
    
    X_Data_filtered = X_Data_scaled.iloc[range(0, 96, 4)]
    X_Data_filtered.insert(3, "Hour", range(1, len(X_Data_filtered) + 1))  # Fügt "Hour" als 4. Spalte hinzu
    
    #predict with real data
    
    y_pred = model.predict(X_Data_filtered)
    
    return y_pred