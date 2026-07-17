import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import matplotlib.image as mpimg
import seaborn as sns

the_months = {1:'January',2:'February',3:'March',4:'April',
              5:'May',6:'June',7:'July',8:'August',9:'September',
              10:'October',11:'November',12:'December'}

###   Load the data   ###
wx_daily = pd.read_csv('https://raw.githubusercontent.com/drolsonmi/weather/refs/heads/main/data/Snow%20Weather_Daily.dat', header=1)
wx_5min = pd.read_csv('https://raw.githubusercontent.com/drolsonmi/weather/refs/heads/main/data/Snow%20Weather_FiveMin.dat', header=1)


###   Clean the data   ###
wx_5min = wx_5min.drop([0,1], axis=0)

###   Make Time and Date columns   ###
wx_5min.insert(0, 'Time', pd.to_datetime(wx_5min['TIMESTAMP']).dt.time)
wx_5min.insert(0, 'Date', pd.to_datetime(wx_5min['TIMESTAMP']).dt.date)

wx_5min.insert(2,'Year', pd.to_datetime(wx_5min['Date']).dt.year)
wx_5min.insert(3,'Month',
               pd.to_datetime(wx_5min['Date']).dt.month.apply(lambda x:the_months[x]))
wx_5min.insert(4,'Day', pd.to_datetime(wx_5min['Date']).dt.day)
wx_5min.insert(5,'Hour', pd.to_datetime(wx_5min['TIMESTAMP']).dt.hour)
wx_5min.insert(6,'Minute', pd.to_datetime(wx_5min['TIMESTAMP']).dt.minute)
wx_5min.insert(7,'Second', pd.to_datetime(wx_5min['TIMESTAMP']).dt.second)

###   Convert date and time to datetime format   ###
wx_5min['TIMESTAMP'] = pd.to_datetime(wx_5min['TIMESTAMP'], format='%Y-%m-%d %H:%M:%S')
wx_5min['Time'] = pd.to_datetime(wx_5min['Time'], format='%H:%M:%S')
wx_5min['Date'] = pd.to_datetime(wx_5min['Date'], format='%Y-%m-%d')

###   Convert data to float   ###
wx_5min['AirTF_Avg'] = wx_5min['AirTF_Avg'].astype(float)

###   Clean the AirTF Data   ###
wx_5min.loc[wx_5min['AirTF_Avg'] > 150, 'AirTF_Avg'] = np.nan
wx_5min.loc[wx_5min['AirTF_Avg'] < -30, 'AirTF_Avg'] = np.nan

###   Convert Precip amounts to Float ###
wx_5min['Rain_Tot'] = wx_5min['Rain_Tot'].apply(float)
wx_5min['HeatedPrecip_Tot'] = wx_5min['HeatedPrecip_Tot'].apply(float)




#######  Create the image  #######

fig = plt.figure(figsize=(12, 12))
fig.text(0.5, 0.98,
         f"Snow College Weather Station - {wx_5min['TIMESTAMP'].iloc[-1].strftime('%B %d, %Y - %I:%M %p')}                    ",
         ha='center',
         fontsize=13,
         fontweight='bold')


###   Snow Weather Logo   ###
img = mpimg.imread('./images/SnowWeatherLogo_Blue.png')
ax_logo = fig.add_axes((0.01, 0.9, 0.14, 0.14))
ax_logo.imshow(img)
ax_logo.axis('off') 

###   Temperature and Relative Humidity   ###

# Temperature
ax_temp = fig.add_axes((0.2, 0.85, 0.5, 0.1))
sns.lineplot(data=wx_5min.tail(288), x='TIMESTAMP', y='AirTF_Avg', ax=ax_temp, color='red')
ax_temp.xaxis.set_major_formatter(mdates.DateFormatter('%-I %p'))
ax_temp.set_xlabel('')
ax_temp.set_ylabel('Temperature (F)', color='red')
ax_temp.set_title('Temperature and Relative Humidity', fontsize=12)

# Relative Humidity
ax_rh = ax_temp.twinx()
sns.lineplot(data=wx_5min.tail(288), x='TIMESTAMP', y='RH_Avg', ax=ax_rh, color='green')
ax_rh.set_ylabel('Relative Humidity (%)', color='green')


###   Pressure   ###
ax_press = fig.add_axes((0.2, 0.7, 0.5, 0.1))
sns.lineplot(data=wx_5min.tail(288), x='TIMESTAMP', y='BP_inHg', ax=ax_press, color='blue')
ax_press.xaxis.set_major_formatter(mdates.DateFormatter('%-I %p'))
ax_press.set_xlabel('')
ax_press.set_ylabel('Pressure (inHg)', color='blue')
ax_press.set_title('Pressure', fontsize=12)

###   Precipitation   ###
# Liquid precipitation
ax_precip = fig.add_axes((0.2, 0.55, 0.5, 0.1))
sns.lineplot(data=wx_5min.tail(288), x='TIMESTAMP', y='RainRunTot', ax=ax_precip, color='purple')
ax_precip.xaxis.set_major_formatter(mdates.DateFormatter('%-I %p'))
ax_precip.set_xlabel('')
ax_precip.set_ylabel('Precipitation (in)', color='purple')
ax_precip.set_title('Precipitation', fontsize=12)

# Heated precipitation
ax_heated = ax_precip.twinx()

sns.lineplot(
    data=wx_5min.tail(288),
    x='TIMESTAMP',
    y='HeatedRunTot',
    ax=ax_heated,
    color='orange'
)

ax_heated.set_ylabel('Heated Precipitation (in)', color='orange')

# Match colors
ax_precip.tick_params(axis='y', colors='purple')
ax_heated.tick_params(axis='y', colors='orange')





sns.lineplot(data=wx_5min.tail(288), x='TIMESTAMP', y='HeatedRunTot', ax=ax_precip, color='orange')


###   Solar Radiation   ###


###   Current Values   ###


###   Current Wind Conditions   ###


###   Records for last 3 days   ###


###   Output   ###
fig.savefig('./images/weather_image.png', dpi=300)
plt.show()