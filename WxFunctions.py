import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

def test_subroutine():
    return "WxFunctions module is working!"

def load_daily():
    """
    Loads and cleans daily weather data from Snow Weather_Daily.dat.
    Also loads necessary libraries.
    Returns a cleaned pandas DataFrame.
    """
    # import numpy as np
    # import pandas as pd
    # import seaborn as sns

    # # import plotly.express as px
    # import matplotlib.pyplot as plt
    # import plotly.graph_objects as go

    # from datetime import datetime

    # %matplotlib inline

    the_months = {1:'January',2:'February',3:'March',4:'April',
                5:'May',6:'June',7:'July',8:'August',9:'September',
                10:'October',11:'November',12:'December'}

    #####################################
    #####   Import and clean data   #####

    ###   Copy data to .csv file   ###
    import os
    if os.name == 'nt':  # Windows
        # print("Running on Windows")
        cmd = f'copy "data/Snow Weather_Daily.dat" "data/Snow_Daily.csv"'
    else:  # Unix/Linux
        # print("Running on Unix/Linux")
        cmd = f'cp ./data/Snow\ Weather_Daily.dat ./data/Snow_Daily.csv'

    os.system(cmd)

    from pathlib import Path
    data_file = Path('./data/Snow_Daily.csv')
    if not data_file.is_file():
        raise FileNotFoundError("The data file was not found.")
    wx_data = pd.read_csv(data_file, header=1)
    
    ###   Fix the timestamp   ###

    stamp = "TIMESTAMP" # Called TIMESTAMP if all-data file
    wx_data.drop([0,1], axis=0, inplace=True)
    wx_data.insert(0,'Date',pd.to_datetime(wx_data['AirTF_TMx']).dt.date)
    wx_data.drop('TIMESTAMP', axis=1, inplace=True)

    wx_data.insert(1,'Year', pd.to_datetime(wx_data['Date']).dt.year)
    wx_data.insert(2,'Month',
                pd.to_datetime(wx_data['Date']).dt.month.apply(lambda x:the_months[x]))
    wx_data.insert(3,'Day', pd.to_datetime(wx_data['Date']).dt.day)
    wx_data['DOY'] = pd.to_datetime(wx_data['Date']).dt.strftime('%m-%d')

    #wx_data.tail()

    ###   Convert data to float   ###
    wx_data['AirTF_Max'] = wx_data['AirTF_Max'].apply(lambda x: float(x))
    wx_data['AirTF_Min'] = wx_data['AirTF_Min'].apply(lambda x: float(x))
    wx_data['AirTF_Avg'] = wx_data['AirTF_Avg'].apply(lambda x: float(x))

    ###   Clean the AirTF Data   ###

    wx_data.loc[wx_data['AirTF_Max'] > 150, 'AirTF_Max'] = np.nan
    wx_data.loc[wx_data['AirTF_Avg'] > 150, 'AirTF_Avg'] = np.nan
    wx_data.loc[wx_data['AirTF_Min'] > 150, 'AirTF_Min'] = np.nan

    wx_data.loc[wx_data['AirTF_Max'] < -30, 'AirTF_Max'] = np.nan
    wx_data.loc[wx_data['AirTF_Avg'] < -30, 'AirTF_Avg'] = np.nan
    wx_data.loc[wx_data['AirTF_Min'] < -30, 'AirTF_Min'] = np.nan

    ###   Convert Precip amounts to Float ###
    wx_data['Rain_Tot'] = wx_data['Rain_Tot'].apply(float)
    wx_data['HeatedPrecip_Tot'] = wx_data['HeatedPrecip_Tot'].apply(float)
    return wx_data

#############################################################
#####   Function to find Wx Data from a Specific Date   #####

def WxExtremes(wx_data, start_date, end_date=datetime.now().date(), n=5):
    first_date = pd.Timestamp(start_date)
    last_date = pd.Timestamp(end_date)
    wx_range = wx_data[(wx_data['Date'] >= first_date.date()) &
                   (wx_data['Date'] <= last_date.date())]
    print(f"===== {start_date} through {end_date} =====")
    print("--- Average Temperatures ---")
    print(f"Average High: {wx_range['AirTF_Max'].mean()}")
    print(f"Average     : {wx_range['AirTF_Avg'].mean()}")
    print(f"Average Low : {wx_range['AirTF_Min'].mean()}")
    print(f"--- {n} highest temperatures ---")
    print(wx_range.sort_values('AirTF_Max', ascending=False).head(n)[['Date', 'AirTF_Max']])
    print(f"--- {n} lowest temperatures ---")
    print(wx_range.sort_values('AirTF_Min', ascending=True).head(n)[['Date', 'AirTF_Min']])
    print("--- Accumulated Precipitation ---")
    print("Total Liquid Precip: {0}".format(wx_range['Rain_Tot'].sum()))
    print("Total Heated Precip: {0}".format(wx_range['HeatedPrecip_Tot'].sum()))
    print(f"--- {n} days with highest precipitation ---")
    print(wx_range.sort_values('Rain_Tot', ascending=False).head(n)[['Date', 'Rain_Tot']])
    print(f"--- {n} days with highest heated precipitation ---")
    print(wx_range.sort_values('HeatedPrecip_Tot', ascending=False).head(n)[['Date', 'HeatedPrecip_Tot']])
    # sns.lineplot(data=wx_range, x='Date', y='AirTF_Max')#,'AirTF_Avg','AirTF_Min'])
    # plt.show()
    # fig = px.line(wx_range, x='Date', y=['AirTF_Max','AirTF_Avg','AirTF_Min'])
    # fig.show()
    # fig = px.line(wx_range, x='Date', y=['Rain_Tot','HeatedPrecip_Tot'])
    # fig.show()

def WxSummary(wx_data, start_date, end_date=datetime.now().date()):
    first_date = pd.Timestamp(start_date)
    last_date = pd.Timestamp(end_date)
    wx_range = wx_data[(wx_data['Date'] >= first_date.date()) &
                   (wx_data['Date'] <= last_date.date())]
    wx_summary = {
        "Max T"               : wx_range['AirTF_Max'].max(),
        "Max T Date"          : wx_range.sort_values('AirTF_Max', ascending=False).head(1)['Date'].values,
        "Average T"           : wx_range['AirTF_Avg'].mean(),
        "Min T"               : wx_range['AirTF_Min'].min(),
        "Min T Date"          : wx_range.sort_values('AirTF_Min', ascending=True).head(1)['Date'].values,
        "Total Precip"        : wx_range['Rain_Tot'].sum(),
        "Max Precip"          : wx_range['Rain_Tot'].max(),
        "Max Precip Date"     : wx_range.sort_values('Rain_Tot', ascending=False).head(1)['Date'].values,
        "Total Heated Precip" : wx_range['HeatedPrecip_Tot'].sum()
    }
    return wx_summary

def Days_Above(wx_data):
    days_above = pd.DataFrame(columns=range(2019,2026))
    for yr in range(2019,2026):
        for T in range(105,60,-5):
            days_above.loc[T,yr] = len(wx_data[(wx_data['Year'] == yr) & (wx_data['AirTF_Max'] >= T)])
    return days_above

def Days_Below(wx_data):
    days_below = pd.DataFrame(columns=range(2019,2026))
    for yr in range(2019,2026):
        for T in range(-10,51,5):
            days_below.loc[T,yr] = len(wx_data[(wx_data['Year'] == yr) & (wx_data['AirTF_Min'] <= T)])
    return days_below

def DailyNumbers(wx_data, date_for_nums):
    cond = (wx_data['Date'] == pd.Timestamp(date_for_nums).date())
    # return wx_data[cond]['AirTF_Max']
    return pd.Series({
        'Max T'         : wx_data[cond]['AirTF_Max'].max(),
        'Average T'     : wx_data[cond]['AirTF_Avg'].mean(),
        'Min T'         : wx_data[cond]['AirTF_Min'].min(),
        'Precip'        : wx_data[cond]['Rain_Tot'].sum(),
        'Heated Precip' : wx_data[cond]['HeatedPrecip_Tot'].sum()
    })

def WaterYears(wx_data, startdate='10-01', enddate='09-30'):
    table = pd.DataFrame()
    endyear = 0
    if int(enddate[:2]) < 10:
        endyear = 1
    for yr in range(2019,2026):
        print(str(yr)+'-' + startdate + ' through ' + str(yr+endyear)+'-' + enddate)
        table[str(yr)+'-'+str(yr+1)] = WxSummary(wx_data, str(yr)+'-10-01', str(yr+endyear)+'-' + enddate)
    return table

def CalendarYears(wx_data, startdate='01-01', enddate='12-31'):
    table = pd.DataFrame()
    for yr in range(2019,2027):
        table[yr] = WxSummary(wx_data, str(yr) + '-' + startdate, str(yr) + '-' + enddate)
    return table