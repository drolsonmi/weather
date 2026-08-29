import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import matplotlib.image as mpimg
import matplotlib.transforms as transforms
from matplotlib.patches import FancyBboxPatch, Circle
import seaborn as sns


the_months = {1:'January',2:'February',3:'March',4:'April',
              5:'May',6:'June',7:'July',8:'August',9:'September',
              10:'October',11:'November',12:'December'}


import platform

hour_flag = '%#I' if platform.system() == 'Windows' else '%-I'
date_flag = '%#d' if platform.system() == 'Windows' else '%-d'
time_fmt = f'{hour_flag} %p'



###   Meteogram x-position/width   ###
# Shifted right (was x=0.2, w=0.5) to leave room for the Current Conditions
# and Wind panels stacked in the left margin.
GRAPH_X, GRAPH_W = 0.30, 0.48

###   Panel Dimensions   ###

panel_x, panel_w = 0.01, 0.22
panel_y, panel_h = 0.60, 0.29
panel2_h = 0.22

row1_center = 0.58
row2_center = 0.15

compass_cx_frac = 0.30
compass_size_fig = 0.085



###   Copy Five-Minute Data   ###
# import os
# os.system('cd C:\Users\GramSC\Documents\weather')
# cmd = f'copy "C:\Campbellsci\LoggerNet\Snow Weather_FiveMin.dat" "./data/Snow Weather_FiveMin.dat"'
# os.system(cmd)

###   Load the data   ###
# wx_daily = pd.read_csv('https://raw.githubusercontent.com/drolsonmi/weather/refs/heads/main/data/Snow%20Weather_Daily.dat', header=1)
wx_5min = pd.read_csv('https://raw.githubusercontent.com/drolsonmi/weather/refs/heads/main/data/Snow%20Weather_FiveMin.dat', header=1)
# wx_5min = pd.read_csv('C:\Campbellsci\LoggerNet\Snow Weather_FiveMin.dat')
# wx_5min = pd.read_csv(r'C:\Users\GramSC\Documents\weather\data\Snow Weather_FiveMin.dat', header=1)
# wx_5min = pd.read_csv('./data/Snow Weather_FiveMin.dat')

###   Change TIMESTAMP to datetime format   ###
wx_5min['TIMESTAMP'] = pd.to_datetime(wx_5min['TIMESTAMP'], errors='coerce')
wx_5min = wx_5min.dropna(subset=['TIMESTAMP'])

###   Clean the data   ###
# wx_5min = wx_5min.drop([0,1], axis=0)

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

###   Load 15-minute data   ###
wx_15min = pd.read_csv(
    'https://raw.githubusercontent.com/drolsonmi/weather/refs/heads/main/data/Snow%20Weather_FifteenMin.dat',
    skiprows=[0, 2, 3],  # keep row 1 (variable names) as header
    header=0
)
wx_15min['TIMESTAMP'] = pd.to_datetime(wx_15min['TIMESTAMP'])

#######  Data Subset  #######
def obtain_subset(X, endtime="now", starttime=0):
    return X.tail(288)

wx = obtain_subset(wx_5min)

#######  Create the image  #######
sns.set_style("darkgrid")

title_timestamp = wx['TIMESTAMP'].iloc[-1].strftime(f'%B {date_flag}, %Y - {hour_flag}:%M %p')

fig = plt.figure(figsize=(12, 12))
# fig.text(0.5, 0.98,
fig.text(GRAPH_X + GRAPH_W/2 + 0.05, 0.98,
         f"Snow College Weather Station - {title_timestamp}                    ",
         ha='center',
         fontsize=14,
         fontweight='bold')

xmin = wx['TIMESTAMP'].min()
xmax = wx['TIMESTAMP'].max()

###   Snow Weather Logo   ###
img = mpimg.imread('./images/SnowWeatherLogo_Blue.png')
# ax_logo = fig.add_axes((0.01, 0.9, 0.14, 0.14))
ax_logo = fig.add_axes((panel_x + 0.02, panel_y + panel_h + 0.005, panel_w - 0.04, 0.14))
ax_logo.imshow(img)
ax_logo.axis('off') 

# Common data used by the side panels
today_mask = wx_5min['Date'] == pd.Timestamp.today().strftime("%Y-%m-%d")
latest = wx.tail(1)



###   Temperature and Relative Humidity   ###

# Temperature
ax_temp = fig.add_axes((GRAPH_X, 0.85, GRAPH_W, 0.1))
sns.lineplot(data=wx, x='TIMESTAMP', y='AirTF_Avg', ax=ax_temp, color='red')
ax_temp.xaxis.set_major_formatter(mdates.DateFormatter(time_fmt))
ax_temp.set_xlabel('')
ax_temp.set_xlim(xmin, xmax)
ax_temp.set_ylabel('Temperature (F)', color='red')
ax_temp.set_title('Temperature and Relative Humidity', fontsize=12)

# Relative Humidity
ax_rh = ax_temp.twinx()
sns.lineplot(data=wx, x='TIMESTAMP', y='RH_Avg', ax=ax_rh, color='green')
ax_rh.set_ylabel('Relative Humidity (%)', color='green')
# ax_temp.grid(False, which='major', axis='x')
ax_rh.grid(False, which='major', axis='x')
ax_rh.grid(False, which='major', axis='y')
ax_rh.set_xlim(xmin, xmax)

###   Pressure   ###
ax_press = fig.add_axes((GRAPH_X, 0.7, GRAPH_W, 0.1))
sns.lineplot(data=wx, x='TIMESTAMP', y='BP_inHg', ax=ax_press, color='purple')
ax_press.xaxis.set_major_formatter(mdates.DateFormatter(time_fmt))
ax_press.set_xlabel('')
ax_press.set_ylabel('Pressure (inHg)', color='purple')
ax_press.set_title('Atmospheric Pressure (MSLP)', fontsize=12)
ax_press.set_xlim(xmin, xmax)


###   Precipitation   ###
# Liquid precipitation
ax_precip = fig.add_axes((GRAPH_X, 0.55, GRAPH_W, 0.1))
sns.lineplot(
    data=wx,
    x='TIMESTAMP',
    y='RainRunTot',
    ax=ax_precip,
    color='blue'
)
ax_precip.xaxis.set_major_formatter(mdates.DateFormatter(time_fmt))
ax_precip.set_xlabel('')
ax_precip.set_ylabel('Precipitation (in)', color='blue')
ax_precip.set_title('Precipitation', fontsize=12)
ax_precip.set_xlim(xmin, xmax)

# Heated precipitation
ax_heated = ax_precip.twinx()
sns.lineplot(
    data=wx,
    x='TIMESTAMP',
    y='HeatedRunTot',
    ax=ax_heated,
    color='orange'
)
ax_heated.set_ylabel('Heated Precipitation (in)', color='orange')

# Set Heated and Precipitation y-axis limits to match Precipitation y-axis limits
ymax = max(wx['RainRunTot'].max(), wx['HeatedRunTot'].max())
ax_precip.set_xlim(xmin, xmax)
ax_precip.set_ylim(0, max(0.021, ymax+0.002))
ax_heated.set_xlim(xmin, xmax)
ax_heated.set_ylim(0, max(0.021, ymax+0.002))


###   Wind   ###

# Wind Speed
ax_wind = fig.add_axes((GRAPH_X, 0.38, GRAPH_W, 0.1))
sns.lineplot(data=wx, x='TIMESTAMP', y='AveWindSp', ax=ax_wind, color='blue')
sns.scatterplot(data=wx, x='TIMESTAMP', y='WindGust', ax=ax_wind, color='red', s=3)
ax_wind.xaxis.set_major_formatter(mdates.DateFormatter(time_fmt))
ax_wind.set_xlabel('')
ax_wind.set_xlim(xmin, xmax)
ax_wind.set_ylabel('Wind Speed (mph)', color='blue')
# ax_wind.set_title('Wind Speed and Direction', fontsize=12)


# Wind Direction

# Trim to the same time window as your existing plot
wx15_plot = wx_15min[(wx_15min['TIMESTAMP'] >= xmin) & (wx_15min['TIMESTAMP'] <= xmax)].copy()

# u, v components of the vector the wind is blowing TOWARD
wind_dir_rad = np.radians(wx15_plot['WindDir'])
speed_ratio = (wx15_plot['AveWindSp'] + wx15_plot['AveWindSp'].max()) / (2*wx15_plot['AveWindSp'].max())
u = -np.sin(wind_dir_rad) * speed_ratio
v = -np.cos(wind_dir_rad) * speed_ratio

ax_dir = fig.add_axes((GRAPH_X, 0.455, GRAPH_W, 0.05), sharex=ax_wind)
ax_dir.axis('off')
ax_dir.set_xlim(xmin, xmax)
ax_dir.set_title('Wind Speed and Direction', fontsize=12)

trans2 = transforms.blended_transform_factory(ax_dir.transData, ax_dir.transAxes)
ax_dir.quiver(wx15_plot['TIMESTAMP'], [0.5]*len(wx15_plot), u, v,
              transform=trans2, scale=25, width=0.003,
              headwidth=3, headlength=4, color='black', alpha=0.5)

###   Solar Radiation   ###
ax_solar = fig.add_axes((GRAPH_X, 0.23, GRAPH_W, 0.1))
sns.lineplot(data=wx, x='TIMESTAMP', y='SlrkW', ax=ax_solar, color='orange')
ax_solar.xaxis.set_major_formatter(mdates.DateFormatter(time_fmt))
ax_solar.set_xlabel('')
ax_solar.set_xlim(xmin, xmax)
ax_solar.set_ylabel(r'$\text{Power (}kW/m^2\text{)}$', color='orange')
ax_solar.set_title('Solar Irradiance', fontsize=12)




###   Current Conditions Panel   ###

ax_panel = fig.add_axes((panel_x, panel_y, panel_w, panel_h))
ax_panel.axis('off')
ax_panel.set_xlim(0, 1)
ax_panel.set_ylim(0, 1)

# Rounded background card
ax_panel.add_patch(FancyBboxPatch(
    (0.02, 0.02), 0.96, 0.96,
    boxstyle="round,pad=0.02,rounding_size=0.04",
    linewidth=1, edgecolor='#cccccc', facecolor='#f7f7f7',
    transform=ax_panel.transAxes, zorder=0
))

# Header (date/timestamp line removed)
ax_panel.text(0.5, 0.93, "CURRENT CONDITIONS", ha='center', va='center',
              fontsize=11, fontweight='bold', color='#333333',
              transform=ax_panel.transAxes)

def stat_card(ax, x, y, w, h, label, value, color,
              high=None, low=None, unit='', unit_below=None):
    """Draw one label / big-value / high-low stat block in axes-fraction coords."""
    # Label
    ax.text(x + w/2, y + h - 0.03, label, ha='center', va='top',
            fontsize=8, fontweight='bold', color='#777777', transform=ax.transAxes)
    # Big value -- moved down (closer to High/Low) to tighten the gap
    ax.text(x + w/2, y + h*0.44 + 0.03, value, ha='center', va='center',
            fontsize=19, fontweight='bold', color=color, transform=ax.transAxes)
    # Optional small unit directly below the value (e.g. "inHg")
    if unit_below:
        ax.text(x + w/2, y + h*0.44 - 0.04, unit_below, ha='center', va='top',
                fontsize=7, color=color, transform=ax.transAxes)
    # High / Low, stacked, pulled up closer to the value
    if high is not None and low is not None:
        ax.text(x + w*0.28, y + h*0.08 + 0.02, f"High\n{high:0.1f}{unit}",
                ha='center', va='bottom', fontsize=7, color='#555555',
                transform=ax.transAxes, linespacing=1.4)
        ax.text(x + w*0.72, y + h*0.08 + 0.02, f"Low\n{low:0.1f}{unit}",
                ha='center', va='bottom', fontsize=7, color='#555555',
                transform=ax.transAxes, linespacing=1.4)

stats = [
    dict(label="TEMPERATURE", col='AirTF_Avg', color='firebrick',
         value=lambda v: f"{v:0.1f}\u00b0F", unit='\u00b0', unit_below=None),
    dict(label="HUMIDITY",    col='RH_Avg',    color='seagreen',
         value=lambda v: f"{v:0.0f}%",         unit='%', unit_below=None),
    dict(label="DEW POINT",   col='TdC_Avg',   color='teal',
         value=lambda v: f"{v:0.1f}\u00b0F",   unit='\u00b0', unit_below=None),
    dict(label="PRESSURE",    col='BP_inHg',   color='indigo',
         value=lambda v: f"{v:0.2f}",          unit='inHg', unit_below=None),
]

# Grid geometry (2x2, rows 1-2)
col_w  = 0.44
gap_x  = 0.08   # space between columns
x_left  = 0.03
x_right = x_left + col_w + gap_x

row_h  = 0.28
y_row1 = 0.62
y_row2 = 0.32
y_row3 = 0.03

positions_rows12 = [(x_left, y_row1), (x_right, y_row1),
                     (x_left, y_row2), (x_right, y_row2)]

for (x, y), s in zip(positions_rows12, stats):
    now_val = latest[s['col']].values[0]
    hi = wx_5min.loc[today_mask, s['col']].max()
    lo = wx_5min.loc[today_mask, s['col']].min()
    stat_card(ax_panel, x, y, col_w, row_h,
              s['label'], s['value'](now_val), s['color'],
              high=hi, low=lo, unit=s['unit'], unit_below=s['unit_below'])

# --- Row 3: Precipitation (centered, full width -- solar moved off this panel) ---
def precip_card(ax, x, y, w, h, min_temp_today):
    """
    Shows Rain, Heated, or both, depending on today's minimum temperature:
      - min < 32F   -> heated precip only (liquid precip is unreliable/frozen)
      - min > 40F   -> regular precip only (no heater needed)
      - 32-40F      -> show both
    """
    rain_total = wx_5min.loc[today_mask, 'Rain_Tot'].sum()
    heated_total = wx_5min.loc[today_mask, 'HeatedPrecip_Tot'].sum()

    show_heated_only = min_temp_today < 32
    show_regular_only = min_temp_today > 40
    show_both = not show_heated_only and not show_regular_only

    ax.text(x + w/2, y + h - 0.03, "PRECIPITATION", ha='center', va='top',
            fontsize=8, fontweight='bold', color='#777777', transform=ax.transAxes)

    if show_both:
        # Two values stacked: regular on top, heated below
        ax.text(x + w/2, y + h*0.60, f"{rain_total:0.2f}\"", ha='center', va='center',
                fontsize=15, fontweight='bold', color='steelblue', transform=ax.transAxes)
        ax.text(x + w/2, y + h*0.28, f"{heated_total:0.2f}\"", ha='center', va='center',
                fontsize=13, fontweight='bold', color='darkorange', transform=ax.transAxes)
        ax.text(x + w/2, y + h*0.28 - 0.065, "Heated", ha='center', va='top',
                fontsize=6.5, color='darkorange', transform=ax.transAxes)
    elif show_heated_only:
        ax.text(x + w/2, y + h*0.5, f"{heated_total:0.2f}\"", ha='center', va='center',
                fontsize=17, fontweight='bold', color='darkorange', transform=ax.transAxes)
        ax.text(x + w/2, y + h*0.5 - 0.075, "Heated Precip", ha='center', va='top',
                fontsize=7, color='darkorange', transform=ax.transAxes)
    else:  # regular only
        ax.text(x + w/2, y + h*0.5, f"{rain_total:0.2f}\"", ha='center', va='center',
                fontsize=17, fontweight='bold', color='steelblue', transform=ax.transAxes)
        ax.text(x + w/2, y + h*0.5 - 0.075, "Precipitation", ha='center', va='top',
                fontsize=7, color='steelblue', transform=ax.transAxes)

min_temp_today = wx_5min.loc[today_mask, 'AirTF_Avg'].min()
full_w = (x_right + col_w) - x_left  # spans both columns, centered in the panel
precip_card(ax_panel, x_left, y_row3, full_w, row_h, min_temp_today)


###   Current Wind Panel   ###
# (unchanged)

panel2_y = panel_y - 0.02 - panel2_h  # small gap between panels
rose_y = panel2_y + 0.02  # small gap between rose and top of panel

ax_windpanel = fig.add_axes((panel_x, panel2_y, panel_w, panel2_h))
ax_windpanel.axis('off')
ax_windpanel.set_xlim(0, 1)
ax_windpanel.set_ylim(0, 1)

ax_windpanel.add_patch(FancyBboxPatch(
    (0.02, 0.02), 0.96, 0.96,
    boxstyle="round,pad=0.02,rounding_size=0.04",
    linewidth=1, edgecolor='#cccccc', facecolor='#f7f7f7',
    transform=ax_windpanel.transAxes, zorder=0
))

ax_windpanel.text(0.5, 0.90, "CURRENT WIND", ha='center', va='center',
                   fontsize=11, fontweight='bold', color='#333333',
                   transform=ax_windpanel.transAxes)



compass_cx_fig = panel_x + panel_w * compass_cx_frac
compass_cy_fig = panel2_y + panel2_h * row1_center + 0.01

ax_compass = fig.add_axes((
    compass_cx_fig - compass_size_fig/2,
    compass_cy_fig - compass_size_fig/2,
    compass_size_fig, compass_size_fig
))
ax_compass.set_aspect('equal')
ax_compass.axis('off')
ax_compass.set_xlim(-1.3, 1.3)
ax_compass.set_ylim(-1.3, 1.3)

ax_compass.add_patch(Circle((0, 0), 1.0, facecolor='white',
                             edgecolor='#999999', linewidth=1, zorder=1))

for ang, txt in [(0, 'N'), (90, 'E'), (180, 'S'), (270, 'W')]:
    rad = np.radians(ang)
    ax_compass.text(1.18*np.sin(rad), 1.18*np.cos(rad), txt,
                     ha='center', va='center', fontsize=7,
                     fontweight='bold', color='#666666')

wind_dir_now = latest['WindDir'].values[0]
dir_rad = np.radians(wind_dir_now)
needle_x = np.sin(dir_rad)
needle_y = np.cos(dir_rad)

ax_compass.annotate('', xy=(needle_x*0.92, needle_y*0.92), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='-|>', color='crimson',
                                     linewidth=2, mutation_scale=14),
                     zorder=3)
ax_compass.add_patch(Circle((0, 0), 0.06, facecolor='crimson',
                             edgecolor='none', zorder=4))

ax_windpanel.text(compass_cx_frac, row1_center - 0.24, f"{wind_dir_now:0.0f}\u00b0",
                   ha='center', va='center', fontsize=9,
                   fontweight='bold', color='#555555',
                   transform=ax_windpanel.transAxes)

wind_speed_now = latest['AveWindSp'].values[0]
wind_gust_now = latest['WindGust'].values[0]
stack_x = 0.72

ax_windpanel.text(stack_x, row1_center + 0.14, f"{wind_speed_now:0.1f}",
                   ha='center', va='center', fontsize=16, fontweight='bold',
                   color='steelblue', transform=ax_windpanel.transAxes)
ax_windpanel.text(stack_x, row1_center + 0.14 - 0.055, "mph wind",
                   ha='center', va='top', fontsize=7, color='#777777',
                   transform=ax_windpanel.transAxes)

ax_windpanel.text(stack_x, row1_center - 0.06, f"{wind_gust_now:0.1f}",
                   ha='center', va='center', fontsize=16, fontweight='bold',
                   color='firebrick', transform=ax_windpanel.transAxes)
ax_windpanel.text(stack_x, row1_center - 0.06 - 0.055, "mph gust",
                   ha='center', va='top', fontsize=7, color='#777777',
                   transform=ax_windpanel.transAxes)

max_wind_today = wx_5min.loc[today_mask, 'AveWindSp'].max()
max_gust_today = wx_5min.loc[today_mask, 'WindGust'].max()

ax_windpanel.text(0.28, row2_center + 0.05, f"{max_wind_today:0.1f} mph",
                   ha='center', va='center', fontsize=11, fontweight='bold',
                   color='steelblue', transform=ax_windpanel.transAxes)
ax_windpanel.text(0.28, row2_center - 0.04, "Max Wind",
                   ha='center', va='top', fontsize=6.5, color='#777777',
                   transform=ax_windpanel.transAxes)

ax_windpanel.text(0.72, row2_center + 0.05, f"{max_gust_today:0.1f} mph",
                   ha='center', va='center', fontsize=11, fontweight='bold',
                   color='firebrick', transform=ax_windpanel.transAxes)
ax_windpanel.text(0.72, row2_center - 0.04, "Max Gust",
                   ha='center', va='top', fontsize=6.5, color='#777777',
                   transform=ax_windpanel.transAxes)


###   Records for last 3 days   ###


###   Output   ###
fig.savefig('./images/weather_image.png', dpi=300)
# plt.show()