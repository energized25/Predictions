#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  3 16:48:25 2025

@author: sarahvonhardenberg
"""

import pandas as pd
from datetime import datetime, timedelta
from parameters import get_startyear, get_endyear
import os
import sys

heute = datetime.now()
morgen = heute + timedelta(days=1)
dat_trad_day = morgen
#dat_trad_day = 01.06.2025 ?? Format?

# Define involved countries
#Country_1='SWI'
#Country_2='ITA'
#d_a=26
#m_a=4
#y_a=2025
#d_e=28
#m_e=4
#y_e=2025

#from datetime import datetime, timedelta
#start_date = datetime(y_a, m_a, d_a)  # Beispiel: 1. Januar 2023
#end_date = datetime(y_e, m_e, d_e)  # Beispiel: 10. Januar 2023

# Schleife durch den Bereich von A bis B
#current_date = start_date

def read_X():
    
    

    
    # Generiere den Dateinamen basierend auf dem aktuellen Datum

    dateiname = "DAT/SUP/InputX.csv"
    daten2 = pd.DataFrame()
    
    try:
        # Lese die CSV-Datei ein (ohne Überschriften)
        if not os.path.exists(dateiname):
            print("Failed reading "+dateiname+" - Exit")
            sys.exit(1)
        daten2 = pd.read_csv(dateiname)
        
        return daten2
    
    except FileNotFoundError:
                print(f"Die Datei {dateiname} wurde nicht gefunden.")
    except KeyError:
            print("Die angegebenen Spalten konnten nicht gefunden werden.")




def read_JAO_prices(start_date, end_date, Country_1, Country_2):
    import pandas as pd

    JAO_prices = pd.DataFrame()

    start_year = start_date.year
    end_year = end_date.year
    data_frames = []
    for year in range(start_year, end_year + 1):
        file_name = "JAO_marketdata_export_" + str(year)  # Der Ordnername entspricht der Jahreszahl
        file_path = "DAT/PRC/JAO/"+ Country_1+"-" +Country_2+"/"+ file_name +".csv"
        #print(f"Verarbeite Datum: {dat_trad_day.strftime('%d.%m.%Y')}")

       
        #data = imp_JAO_prices(dat_trad_day,Country_1,Country_2)
       
        #daten transponieren und in temp liste schreiben

        #daten_transponiert = daten2.T
        #result = daten_transponiert.iloc[2:3, :]
        #result.index = [current_date]
        #result.columns = [i for i in range(1, 25)]
 
        #JAO_prices = pd.concat([JAO_prices, result], ignore_index=False)
   

        # Gehe zum nächsten Tag
        #current_date += timedelta(days=1)
        
        df = pd.read_csv(file_path)  # CSV-Datei einlesen
        #df["Year"] = year  # Füge eine Spalte mit der Jahreszahl hinzu

            
        data_frames.append(df)


    if data_frames:
        final_df = pd.concat(data_frames, ignore_index=True)  # DataFrames vertikal zusammenfügen
    return final_df
        
        
        
        
        
        
        
        
        
        
        
        
    return JAO_prices

def read_JAO_prices_old(current_date, end_date, Country_1, Country_2):
    import pandas as pd
    daten2 = pd.DataFrame()
    JAO_prices = pd.DataFrame()
    daten_transponiert =pd.DataFrame()
    while current_date <= end_date:
        dat_trad_day = current_date
        #print(f"Verarbeite Datum: {dat_trad_day.strftime('%d.%m.%Y')}")

       
        daten2 = imp_JAO_prices(dat_trad_day,Country_1,Country_2)
       
        #daten transponieren und in temp liste schreiben

        daten_transponiert = daten2.T
        result = daten_transponiert.iloc[2:3, :]
        result.index = [current_date]
        result.columns = [i for i in range(1, 25)]
 
        JAO_prices = pd.concat([JAO_prices, result], ignore_index=False)
   

        # Gehe zum nächsten Tag
        current_date += timedelta(days=1)
        
    return JAO_prices

    
    
    
def imp_JAO_prices(dat_trad_day,Country_1,Country_2):

# Daten JAO CH-IT

    import pandas as pd
    from datetime import datetime
    import os
    import sys



    
    # Generiere den Dateinamen basierend auf dem aktuellen Datum
    heute = datetime.now()
    dateiname = "DAT/PRC/JAO/" + Country_1 +"-"+ Country_2 + "/JAO_marketdata_export_" + (dat_trad_day).strftime("%d-%m-%Y") + ".csv"
    daten2 = pd.DataFrame()
    
    try:
        # Lese die CSV-Datei ein (ohne Überschriften)
        if not os.path.exists(dateiname):
            print("Failed reading "+dateiname+" - Exit")
            sys.exit(1)
        daten2 = pd.read_csv(dateiname)

        # Wähle nur die Spalten 3, 7 und 10 aus (basierend auf nullbasierten Indizes: 2, 6, 9)
        daten2 = daten2.iloc[:, [2, 6, 9]]
        
        # Füge die gewünschten Überschriften hinzu
        # daten.columns = ["Datum", "Stunde", "Euro / MWh"]
        
        # Überprüfe die Struktur des DataFrames
        print(daten2.head())  # Zeigt die ersten Zeilen der Tabelle
        print("\nDataFrame-Informationen:")
        print(daten2.info())  # Zeigt Informationen über die Spalten und Datentypen
        return daten2
        # Daten sind nun bereit für die weitere Analyse
        # Beispiel: Durchschnittspreis berechnen
        #durchschnittspreis = daten2["Euro / MWh"].mean()
        # print(f"Durchschnittspreis pro MWh: {durchschnittspreis:.2f} Euro")
        
    except FileNotFoundError:
                print(f"Die Datei {dateiname} wurde nicht gefunden.")
    except KeyError:
            print("Die angegebenen Spalten konnten nicht gefunden werden.")



#Daten CH EPEX
def imp_EPEX_Prices(dat_trad_day,Country):

    import pandas as pd
    from datetime import datetime, timedelta


# Generiere den Dateinamen basierend auf dem aktuellen Datum
    heute = datetime.now()
    morgen = heute + timedelta(days=1)

    dateiname = "DAT/PRC/" + "EPEXCH_" + dat_trad_day.strftime("%Y%m%d") + ".csv"


    try:
    # Lese die CSV-Datei ein (ohne Überschriften)
        daten3 = pd.read_csv(dateiname)
        daten3 = daten3.iloc[:, [0, 3, 4]]
        return daten3
    except FileNotFoundError:
        print(f"Die Datei {dateiname} wurde nicht gefunden.")
    except KeyError:
        print("Die angegebenen Spalten konnten nicht gefunden werden.")




def imp_trade_prices(start_date, end_date, country, EXAA=False):
    # Extrahiere die Jahreszahlen aus den übergebenen Datumswerten
    import pandas as pd
    import os

    
    start_year = start_date.year
    end_year = end_date.year

    data_frames = []

    # Iteriere von der niedrigsten zur höchsten Jahreszahl
    for year in range(start_year, end_year + 1):
        folder_path = str(year)  # Der Ordnername entspricht der Jahreszahl
        file_path = "DAT/PRC/"+ country+ "/"+ folder_path

        if os.path.exists(file_path):        
            for file_name in os.listdir(file_path):
                if file_name.startswith("GUI") and file_name.endswith(".csv"):
                    file_path = os.path.join(file_path, file_name)
           
            try:
                df = pd.read_csv(file_path)  # CSV-Datei einlesen
                if df.empty:
                    print("The file is empty, no data to load.")
                df["Year"] = year  # Füge eine Spalte mit der Jahreszahl hinzu
                df["Country"] = country  # Füge eine Spalte mit dem Land hinzu
                
                if EXAA ==False:
                    if country in ["DE", "AT"]: #hier wäre besser eine Bedingung die prüft ob es sequence 1 und 2 gibt
                         df = df[(df['Sequence'] == 'Sequence 1')]
                else:
                    if EXAA == True:
                        df = df[(df['Sequence'] == 'Sequence 2')]
           
                if EXAA ==False:
                    df = transform_quarterly_to_hourly(df)
                else:
                    df = quarter_means(df)
 
                data_frames.append(df)
            except pd.errors.EmptyDataError:
                print("Error: No columns to parse from the file. The file might be empty or misformatted.") 
           
    
            #df = pd.read_csv(file_path)  # CSV-Datei einlesen
            
            
           
        else:
            print(f"Datei nicht gefunden: {file_path}")

    if data_frames:
        final_df = pd.concat(data_frames, ignore_index=True)  # DataFrames vertikal zusammenfügen
        return final_df
    else:
        print("Keine Daten gefunden.")
        return None

# Beispielaufruf
#df = load_csv_files("2022-01-01", "2024-12-31", "DE")

#if df is not None:
 #   print(df.head())  # Zeigt die ersten Zeilen des DataFrames


def transform_quarterly_to_hourly(df):
    """ Falls die Daten viertelstündlich sind, transformiere sie zu Stunden """
    time_column = df.columns[0]  # Angenommen, die erste Spalte enthält den Zeitstempel
    
    # Prüfe, ob es Viertelstunden-Intervalle gibt
    df["Start_Time"] = df[time_column].apply(lambda x: x.split(" - ")[0])
    df["Start_Time"] =  df["Start_Time"].apply(lambda x: x[:19])
    df["End_Time"] = df[time_column].apply(lambda x: x.split(" - ")[1])
    df["End_Time"] =  df["End_Time"].apply(lambda x: x[:19])
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], format="%d/%m/%Y %H:%M:%S", dayfirst=True)
    df["End_Time"] = pd.to_datetime(df["End_Time"], format="%d/%m/%Y %H:%M:%S", dayfirst=True)

    # Prüfe, ob die kleinste Differenz 15 Minuten beträgt (Viertelstundenformat)
    time_diffs = df["Start_Time"].diff().dropna().value_counts()
    df.reset_index(drop=True, inplace=True)
    
    if any(time_diffs.index == pd.Timedelta(minutes=15)):
        print("Viertelstundenformat erkannt – Transformation wird durchgeführt.")

        # Gruppiere jeweils 4 Zeilen zusammen (jede Stunde hat 4 Viertelstunden)
        df_grouped = df.groupby(df.index // 4).first()

        # Korrigiere den Zeitstempel
        df_grouped[time_column] = df_grouped["Start_Time"].dt.strftime("%Y-%m-%d %H:%M:%S")# + " - " + (df_grouped["Start_Time"] + pd.Timedelta(hours=1)).dt.strftime("%d/%m/%Y %H:%M:%S")
        
        # Entferne die Hilfsspalten
        df_grouped.drop(columns=["Start_Time", "End_Time"], inplace=True)

        return df_grouped
    
    # Entferne die Hilfsspalten
    df[time_column] = df["Start_Time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df.drop(columns=["Start_Time", "End_Time"], inplace=True)
    print("Stundenformat erkannt – keine Transformation notwendig.")
    return df

def quarter_means(df):
    """ Falls die Daten viertelstündlich sind, transformiere sie zu Stunden """
    time_column = df.columns[0]  # Angenommen, die erste Spalte enthält den Zeitstempel
    
    # Prüfe, ob es Viertelstunden-Intervalle gibt
    df["Start_Time"] = df[time_column].apply(lambda x: x.split(" - ")[0])
    df["Start_Time"] =  df["Start_Time"].apply(lambda x: x[:19])
    df["End_Time"] = df[time_column].apply(lambda x: x.split(" - ")[1])
    df["End_Time"] =  df["End_Time"].apply(lambda x: x[:19])
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], format="%d/%m/%Y %H:%M:%S", dayfirst=True)
    df["End_Time"] = pd.to_datetime(df["End_Time"], format="%d/%m/%Y %H:%M:%S", dayfirst=True)

    # Prüfe, ob die kleinste Differenz 15 Minuten beträgt (Viertelstundenformat)
    time_diffs = df["Start_Time"].diff().dropna().value_counts()
    df.reset_index(drop=True, inplace=True)
    
    if any(time_diffs.index == pd.Timedelta(minutes=15)):
        print("Viertelstundenformat erkannt – Transformation wird durchgeführt.")

        # Gruppiere jeweils 4 Zeilen zusammen (jede Stunde hat 4 Viertelstunden)

        df_mean = df.groupby(df.index // 4)["Day-ahead Price (EUR/MWh)"].mean().round(2)
        df_grouped = df.groupby(df.index // 4).first()
        df_grouped['Day-ahead Price (EUR/MWh)'] = df_mean                        
        # Korrigiere den Zeitstempel
        df_grouped[time_column] = df_grouped["Start_Time"].dt.strftime("%Y-%m-%d %H:%M:%S")# + " - " + (df_grouped["Start_Time"] + pd.Timedelta(hours=1)).dt.strftime("%d/%m/%Y %H:%M:%S")
        
        # Entferne die Hilfsspalten
        df_grouped.drop(columns=["Start_Time", "End_Time"], inplace=True)

        return df_grouped
    
    # Entferne die Hilfsspalten
    df[time_column] = df["Start_Time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df.drop(columns=["Start_Time", "End_Time"], inplace=True)
    print("Stundenformat erkannt – keine Transformation notwendig.")
    return df


def read_generation_forecast(year):
    """
    Liest die Datei "Generation Forecasts for Wind and Solar_YYYY01010000-ZZZZ01010000.csv" ein
    und gibt den eingelesenen Pandas DataFrame zurück.
    
    :param year: Das Jahr, für das die Datei eingelesen werden soll (YYYY)
    :return: Pandas DataFrame
    """
    # Berechne das nächste Jahr
    next_year = year + 1
    
    # Erstelle den Dateinamen
    file_name = f"DAT/SUP/GER/Generation Forecasts for Wind and Solar_{year}01010000-{next_year}01010000.csv"
    
    # Lese die CSV-Datei ein
    try:
        data_frame = pd.read_csv(file_name)
        #data_frame.replace("-", "nan", inplace=True)
        #data_frame.iloc[:, 1:10] = data_frame.iloc[:, 1:10].astype(float)
        data_frame.iloc[:, 1:10] = data_frame.iloc[:, 1:10].apply(pd.to_numeric, errors="coerce")
        return data_frame
    except FileNotFoundError:
        print(f"Die Datei {file_name} wurde nicht gefunden.")
        return None


# Übergeordnetes Programm
def read_Wind_PV_data():
    
    import pandas as pd
    
    # Start- und Endjahr definieren
    start_year = get_startyear()
    end_year = get_endyear()
    
    # Aggregierter DataFrame
    PV_Wind_FC = pd.DataFrame()
    
    # Loop durch die Jahre
    for year in range(start_year, end_year + 1):
        # Funktion aufrufen und DataFrame übernehmen
        yearly_data = read_generation_forecast(year)
        
        # Daten aggregieren, falls erfolgreich eingelesen
        if yearly_data is not None:
            PV_Wind_FC = pd.concat([PV_Wind_FC, yearly_data], ignore_index=True)
    
    # Überprüfen, ob die Daten erfolgreich aggregiert wurden
    if not PV_Wind_FC.empty:
        print("Daten erfolgreich aggregiert!")
        print(PV_Wind_FC.head())  # Zeige die ersten Zeilen des aggregierten DataFrames an
    return PV_Wind_FC
    #else:
    #    print("Es gab ein Problem beim Einlesen und Aggregieren der Daten.")
    
    
    import pandas as pd

def read_market_data(file_path="JAO_marketdata_export_08-05-2025.xlsx"):
    """
    Liest die Excel-Datei ein und gibt sie als Pandas DataFrame zurück.
    
    :param file_path: Pfad zur Excel-Datei (Standard ist der Dateiname)
    :return: Pandas DataFrame mit den eingelesenen Daten
    """
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print(f"Fehler beim Einlesen der Datei: {e}")
        return None

