# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 22:01:04 2023

@author: DIMITRIS
"""

# import librar# import libraries
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import math
from datetime import datetimeies
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import math
from datetime import datetime

# %%
#Import csv, read data, rename column names

df = pd.read_csv('PycharmProjects/Kalliopi-CIT/alldata.csv')
new_columns = {'Date':'Date',
             'Tmax (C)':'Tmax',
             'Precip. (mm/day)':'PR',
             'Wind Speed (m/s)':'WN',
             'Cloudiness':'CC',
             'RH min':'RHmin'}

df.rename(columns=new_columns,inplace=True)
df.head()

# %%
def EtToRank(temp):
    temperature_ranges = {
        (39, float('inf')): 0,
        (37, 39): 2,
        (35, 37): 4,
        (33, 35): 5,
        (31, 33): 6,
        (29, 31): 7,
        (27, 29): 8,
        (26, 27): 9,
        (23, 26): 10,
        (20, 23): 9,
        (18, 20): 7,
        (15, 18): 6,
        (11, 15): 5,
        (7, 11): 4,
        (0, 7): 3,
        (-6, -0): 2,
        (-float('inf'), -6): 1
    }

    for (lower, upper), tsn in temperature_ranges.items():
        if lower < temp <= upper :
            return tsn

    return None  # Return a default value if no range is matched


def CCToRank(coverage):
    Cloud_coverage_ranges = {
        (10,20): 10,
        (1, 10): 9,
        (20, 30): 9,
        (30, 40): 8,
        (40, 50): 7,
        (50, 60): 6,
        (60, 70): 5,
        (70, 80): 4,
        (80, 90): 3,
        (90, 100): 2,
    }

    for (lower, upper), cc in Cloud_coverage_ranges.items():
        if coverage < 1:
            return 8
        elif coverage >= 100:
            return 1
        elif lower <= coverage < upper :
            return cc

    return None  # Return a default value if no range is matched


def PRToRank(pr):
    Pr_coverage_ranges = {
        (0,3): 9,
        (3,6): 8,
        (6,9): 5,
        (9,12): 2,
        (12,25): 0,
        (25,float('inf')): -1
    }

    for (lower, upper), item in Pr_coverage_ranges.items():
        if pr == 0:
            return 10
        elif lower <= pr < upper:
            return item

    return None  # Return a default value if no range is matched



def WNToRank(windSpeed):
    Wn_coverage_ranges = {
        (0.277777778,2.5): 10,
        (2.5,5.27777778): 9,
        (5.27777778,8.05555556): 8,
        (8.05555556,10.8333333): 6,
        (10.8333333,13.6111111): 3,
        (13.6111111,19.4444445): 0,
        (19.4444445,float('inf')): -10
    }

    for (lower, upper), item in Wn_coverage_ranges.items():
        if windSpeed == 0:
            return 8
        elif lower <= windSpeed < upper:
            return item

    return None  # Return a default value if no range is matched

def TsToRank(ts):
    Ts_coverage_ranges = {
        (-float('inf'),21): -4,
        (21,26): -3,
        (26,29): -2,
        (29,31): -1,
        (31,32.5): 0,
        (32.5,33.5): 1,
        (33.5,34.5): 2,
        (34.5,35.5): 3,
        (35.5,float('inf')): 4
    
    }

    for (lower, upper), item in Ts_coverage_ranges.items():
        if ts == 0:
            return 8
        elif lower <= ts < upper:
            return item

    return None  # Return a default value if no range is matched
# %%



oldMonth = (datetime.strptime(df['Date'][0], "%Y-%m-%d")).month
oldIndex =  0

HCI_Month_avg = []

for i,value in df['Tmax'].iteritems(): 
    newMonth = (datetime.strptime(df['Tmax'][i], "%Y-%m-%d")).month

    
    if not oldMonth == newMonth:
        HCI_Month_avg.append(df['Tmax'][old_index:i].mean())
        print(df['Tmax'][old_index:i])
        
        print(oldIndex,i)
        break
    
        #newMonth = (datetime.strptime(HCI['Date'][i], "%Y-%m-%d")).month
        #new_index = i
        old_Month = newMonth
        old_index = i
        
       
    