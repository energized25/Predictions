#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  3 19:20:36 2025

@author: sarahvonhardenberg
"""


from datetime import datetime, timedelta

# define Start- and End date
# no data before 1/1/2016

d_a=1
m_a=1
y_a=2020
d_e=20
m_e=3
y_e=2025

#Season filter settings
    #range of days from current date +/-
ran=20
    
start_date = datetime(y_a, m_a, d_a)  # Beispiel: 1. Januar 2023
end_date = datetime(y_e, m_e, d_e)  # Beispiel: 10. Januar 2023

def get_startdate():
    return start_date

def get_startyear():
    return y_a

def get_enddate():
    return end_date

def get_endyear():
    return y_e

def get_season_filter_settings():
    return ran