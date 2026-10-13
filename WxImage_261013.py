import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import matplotlib.image as mpimg
import matplotlib.transforms as transforms
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import seaborn as sns

# Online Data Sources - For testing and development
file_5min = 'https://snow.edu/community/weather/SnowWeather_5min.dat'
file_15min = 'https://snow.edu/community/weather/SnowWeather_15min.dat'
file_daily = 'https://snow.edu/community/weather/SnowWeather_Daily.dat'
file_now = 'https://snow.edu/community/weather/SnowWeather_Public.dat'
file_logo = './images/SnowWeatherLogo_Blue.png'
file_output = './images/weather_image.png'

# On Computer Data Sources - For running the actual program
# file_5min = r'C:/Campbellsci/LoggerNet/SnowWeather_5Min.dat'
# file_15min = r'C:/Campbellsci/LoggerNet/SnowWeather_15Min.dat'
# file_daily = r'C:/Campbellsci/LoggerNet/SnowWeather_Daily.dat'
# file_now = r'C:/Campbellsci/LoggerNet/SnowWeather_Public.dat'
# file_logo = r'C:/Users/weather/Weather/SnowWeatherLogo_Blue.png'
# file_output = r'C:/Users/weather/Weather/Weather1.png'

the_months = {1:'January',2:'February',3:'March',4:'April',
              5:'May',6:'June',7:'July',8:'August',9:'September',
              10:'October',11:'November',12:'December'}


import platform

hour_flag = '%#I' if platform.system() == 'Windows' else '%-I'
date_flag = '%#d' if platform.system() == 'Windows' else '%-d'
time_fmt = f'{hour_flag} %p'



###   Canvas and layout   ###
# All positions and sizes below are in INCHES, measured from the bottom-left
# corner of the image. fig_rect() converts them to the figure fractions that
# matplotlib wants, so changing FIG_H/FIG_W never resizes anything.
FIG_W, FIG_H = 12, 15

def fig_rect(x, y, w, h):
    """(x, y, width, height) in inches -> figure fractions."""
    return (x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H)

MARGIN = 0.12      # outer margin on the left/right
GAP = 0.24         # gap between neighboring panels

# --- Right column (full height): Precipitation Totals on top,
#     then the Rain Through table and the 7-Day Summary below it
RIGHT_W = 4.18
RIGHT_X = FIG_W - MARGIN - RIGHT_W
WEEK_TABLE_H = 2.22                            # 7-Day Summary height

# --- Top row: Current Conditions | Current Wind | Precipitation Totals
LOGO_X, LOGO_Y, LOGO_W, LOGO_H = 0.36, FIG_H - 1.26, 2.04, 1.68   # logo box (top-left)
TITLE_X = 7.08                                 # horizontal center of the title lines
TITLE_Y = LOGO_Y + LOGO_H/2                    # title block is centered on the logo
TOP_ROW_TOP = FIG_H - 1.32                     # top edge of the top-row panels
TOP_ROW_H = 3.20                               # height of all three top-row panels (was 3.48)
WIND_W = 2.80                                  # Current Wind width
COND_W = RIGHT_X - GAP - WIND_W - GAP - MARGIN # Current Conditions fills the rest
panel_x, panel_w = MARGIN, COND_W
panel_y, panel_h = TOP_ROW_TOP - TOP_ROW_H, TOP_ROW_H
panel2_x, panel2_w = MARGIN + COND_W + GAP, WIND_W
panel2_y, panel2_h = panel_y, TOP_ROW_H
RAIN_PANEL_H = TOP_ROW_H                       # Precipitation Totals matches the row height

# --- Left side, under Current Conditions + Current Wind: the graphs
#     (same sizes as before: 5.76" x 1.2")
GRAPH_W = 5.76
GRAPH_X = 0.95                                 # leaves room for the left-hand y-axis labels
GRAPH_H = 1.2
GRAPH_TOP = panel_y - 0.50                     # top edge of the Temperature graph

# Wind panel layout (fractions of the wind panel)
row1_center = 0.60
row2_center = 0.15

compass_cx_frac = 0.30
compass_size = 1.02   # inches



###   Load the data   ###
### Daily Data
wx_daily = pd.read_csv(file_daily, skiprows=[0, 2, 3], header=0)
###   5-minute data
wx_5min = pd.read_csv(file_5min, skiprows=[0, 2, 3], header=0) 
###   Load 15-minute data   ###
wx_15min = pd.read_csv(file_15min, skiprows=[0, 2, 3], header=0)
###   Most recent observation (single row) 
wx_now = pd.read_csv(file_now, skiprows=[0, 2, 3], header=0)

###   The Public table's field names differ from the 5-min table's        ###
###   (instantaneous readings, not averages/totals) -- rename to match.    ###
###   Note: the Public table has no wind-gust field at all.                ###
wx_now = wx_now.rename(columns={
    'AirTF': 'AirTF_Avg',
    'RH': 'RH_Avg',
    'TdC': 'TdC_Avg',
    'WS_mph': 'AveWindSp',
    'Rain': 'Rain_Tot',
    'HeatedPrecip': 'HeatedPrecip_Tot',
})

###   Change TIMESTAMP to datetime format   ###
wx_5min['TIMESTAMP'] = pd.to_datetime(wx_5min['TIMESTAMP'], errors='coerce')
wx_5min = wx_5min.dropna(subset=['TIMESTAMP'])
wx_15min['TIMESTAMP'] = pd.to_datetime(wx_15min['TIMESTAMP'])

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

###   Clean and format the "now" data (same treatment as wx_5min)   ###
wx_now['TIMESTAMP'] = pd.to_datetime(wx_now['TIMESTAMP'], errors='coerce')
wx_now = wx_now.dropna(subset=['TIMESTAMP'])
for _col in ['AirTF_Avg', 'RH_Avg', 'TdC_Avg', 'BP_inHg', 'WindDir', 'AveWindSp']:
    wx_now[_col] = wx_now[_col].astype(float)
wx_now.loc[wx_now['AirTF_Avg'] > 150, 'AirTF_Avg'] = np.nan
wx_now.loc[wx_now['AirTF_Avg'] < -30, 'AirTF_Avg'] = np.nan


#######  Data Subset  #######
def obtain_subset(X, endtime="now", starttime=0):
    return X.tail(288)

wx = obtain_subset(wx_5min)

#######  Create the image  #######
sns.set_style("darkgrid")

title_timestamp = wx_now['TIMESTAMP'].iloc[-1].strftime(f'%B {date_flag}, %Y - {hour_flag}:%M %p')

fig = plt.figure(figsize=(FIG_W, FIG_H))
# Two-line title: station name over the date, centered vertically on the logo
fig.text(TITLE_X / FIG_W, (TITLE_Y + 0.02) / FIG_H, "Snow College Weather Station",
         ha='center', va='bottom', fontsize=20, fontweight='bold')
fig.text(TITLE_X / FIG_W, (TITLE_Y - 0.25) / FIG_H, "Ephraim, Utah",
         ha='center', va='bottom', fontsize=16, fontweight='bold')
fig.text(TITLE_X / FIG_W, (TITLE_Y - 0.30) / FIG_H, title_timestamp,
         ha='center', va='top', fontsize=16, fontweight='bold', color='#444444')

xmin = wx['TIMESTAMP'].min()
xmax = wx['TIMESTAMP'].max()

###   Snow Weather Logo   ###
img = mpimg.imread(file_logo)
ax_logo = fig.add_axes(fig_rect(LOGO_X, LOGO_Y, LOGO_W, LOGO_H))  # top-left corner
ax_logo.imshow(img)
ax_logo.axis('off') 

# Common data used by the side panels
today_mask = wx_5min['Date'] == pd.Timestamp.today().strftime("%Y-%m-%d")  # still from wx_5min: needed for today's high/low/totals
latest = wx_now.tail(1)  # current conditions come from the "now" (Public) file

###   Season / calendar-year precip totals (for the Rain Panel)   ###
# RainRunTot resets at midnight, so each day's *last* value is that day's total.
# Summing those daily totals over a date range gives the cumulative precip for that range.
_daily_last_rain = wx_5min.sort_values('TIMESTAMP').groupby('Date')['RainRunTot'].last()

_today_norm = pd.Timestamp.today().normalize()
_oct1_this_year = pd.Timestamp(year=_today_norm.year, month=10, day=1)
water_year_start = _oct1_this_year if _today_norm >= _oct1_this_year \
    else pd.Timestamp(year=_today_norm.year - 1, month=10, day=1)
calendar_year_start = pd.Timestamp(year=_today_norm.year, month=1, day=1)

water_year_precip = _daily_last_rain[_daily_last_rain.index >= water_year_start].sum()
calendar_year_precip = _daily_last_rain[_daily_last_rain.index >= calendar_year_start].sum()

###   Year-by-year rain to date (for the table under the Rain Gauge Panel)   ###
# For each year on record, total rain from Oct 1 -> today's month/day and
# from Jan 1 -> today's month/day, using the same daily totals as the gauges.
RAIN_TABLE_MAX_YEARS = 6      # table is sized for this many years (+ an average row)
RAIN_TABLE_FIRST_YEAR = 2024  # earliest year shown (5-min rain data starts in 2024)

def _same_day_in_year(year, ref):
    """ref's month/day in another year (Feb 29 -> Feb 28 in non-leap years)."""
    day = min(ref.day, pd.Timestamp(year=year, month=ref.month, day=1).days_in_month)
    return pd.Timestamp(year=year, month=ref.month, day=day)

def _rain_between(start, end):
    """Rain total for start..end (inclusive); NaN if the record doesn't cover the start."""
    if _daily_last_rain.empty or _daily_last_rain.index.min() > start:
        return np.nan
    idx = _daily_last_rain.index
    return _daily_last_rain[(idx >= start) & (idx <= end)].sum()

rain_table_rows = []
if not _daily_last_rain.empty:
    _first_yr = max(RAIN_TABLE_FIRST_YEAR, _daily_last_rain.index.min().year)
    for yr in range(_today_norm.year, _first_yr - 1, -1):
        end = _same_day_in_year(yr, _today_norm)
        oct1 = pd.Timestamp(year=yr, month=10, day=1)
        wy_start = oct1 if end >= oct1 else pd.Timestamp(year=yr - 1, month=10, day=1)
        wy_total = _rain_between(wy_start, end)
        cy_total = _rain_between(pd.Timestamp(year=yr, month=1, day=1), end)
        if np.isnan(wy_total) and np.isnan(cy_total):
            continue  # no complete period for this year
        rain_table_rows.append((yr, wy_total, cy_total))
rain_table_rows = rain_table_rows[:RAIN_TABLE_MAX_YEARS]

# Average rain to date for past years in the table (current year excluded).
# Used for the table's bottom row and the "Avg" marks on the rain gauges.
def _nanmean_or_nan(vals):
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else np.nan

_past_rows = [r for r in rain_table_rows if r[0] != _today_norm.year]
rain_avg_wy = _nanmean_or_nan([r[1] for r in _past_rows])  # since Oct 1
rain_avg_cy = _nanmean_or_nan([r[2] for r in _past_rows])  # since Jan 1


###   Temperature and Relative Humidity   ###

# Temperature
ax_temp = fig.add_axes(fig_rect(GRAPH_X, GRAPH_TOP - 1.2, GRAPH_W, GRAPH_H))
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
ax_press = fig.add_axes(fig_rect(GRAPH_X, GRAPH_TOP - 3.0, GRAPH_W, GRAPH_H))
sns.lineplot(data=wx, x='TIMESTAMP', y='BP_inHg', ax=ax_press, color='purple')
ax_press.xaxis.set_major_formatter(mdates.DateFormatter(time_fmt))
ax_press.set_xlabel('')
ax_press.set_ylabel('Pressure (inHg)', color='purple')
ax_press.set_title('Atmospheric Pressure (MSLP)', fontsize=12)
ax_press.set_xlim(xmin, xmax)
ax_rh.set_ylim(max(0,round(wx['RH_Avg'].min() // 10)*10), min(100,(round(wx['RH_Avg'].max() // 10) + 1) * 10))


###   Precipitation   ###
# Liquid precipitation
ax_precip = fig.add_axes(fig_rect(GRAPH_X, GRAPH_TOP - 4.8, GRAPH_W, GRAPH_H))
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
ax_wind = fig.add_axes(fig_rect(GRAPH_X, GRAPH_TOP - 6.84, GRAPH_W, GRAPH_H))
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
# Only every other arrow: on the hour and on the half hour
wx15_plot = wx15_plot[wx15_plot['TIMESTAMP'].dt.minute % 30 == 0]

# u, v components of the vector the wind is blowing TOWARD
wind_dir_rad = np.radians(wx15_plot['WindDir'])
# Arrow length = speed / max(15 mph, fastest speed shown)
speed_ratio = wx15_plot['AveWindSp'] / max(15, wx15_plot['AveWindSp'].max())
u = -np.sin(wind_dir_rad) * speed_ratio
v = -np.cos(wind_dir_rad) * speed_ratio

ax_dir = fig.add_axes(fig_rect(GRAPH_X, GRAPH_TOP - 5.94, GRAPH_W, 0.6), sharex=ax_wind)
ax_dir.axis('off')
ax_dir.set_xlim(xmin, xmax)
ax_dir.set_title('Wind Speed and Direction', fontsize=12)

trans2 = transforms.blended_transform_factory(ax_dir.transData, ax_dir.transAxes)
ax_dir.quiver(wx15_plot['TIMESTAMP'], [0.5]*len(wx15_plot), u, v,
              transform=trans2, scale=25, width=0.003,
              headwidth=3, headlength=4, color='black', alpha=0.5)

###   Solar Radiation   ###
ax_solar = fig.add_axes(fig_rect(GRAPH_X, GRAPH_TOP - 8.64, GRAPH_W, GRAPH_H))
sns.lineplot(data=wx, x='TIMESTAMP', y='SlrkW', ax=ax_solar, color='orange')
ax_solar.xaxis.set_major_formatter(mdates.DateFormatter(time_fmt))
ax_solar.set_xlabel('')
ax_solar.set_xlim(xmin, xmax)
ax_solar.set_ylabel(r'$\text{Power (}kW/m^2\text{)}$', color='orange')
ax_solar.set_title('Solar Irradiance', fontsize=12)


###   7-Day Summary Table   ###
# Right column, under the Rain Through table.
# Rain Through table sizes (inches) -- needed here to place this table below it
rt_title_h = 0.54    # title + column-header band
rt_row_h = 0.174     # height of each year row
rt_pad = 0.096       # bottom padding
rt_h = rt_title_h + rt_row_h * (RAIN_TABLE_MAX_YEARS + 1) + rt_pad  # fixed size: 6 years + average row
week_table_top = panel_y - GAP - rt_h - GAP
ax_table = fig.add_axes(fig_rect(RIGHT_X, week_table_top - WEEK_TABLE_H, RIGHT_W, WEEK_TABLE_H))
ax_table.axis('off')
ax_table.set_xlim(0, 1)
ax_table.set_ylim(0, 1)

# Background card, matching the other panels
ax_table.add_patch(FancyBboxPatch(
    (0.0, 0.0), 1.0, 1.0,
    boxstyle="round,pad=0.01,rounding_size=0.03",
    linewidth=1, edgecolor='#cccccc', facecolor='#f7f7f7',
    transform=ax_table.transAxes, zorder=0
))

ax_table.text(0.5, 0.93, "7-DAY SUMMARY", ha='center', va='center',
              fontsize=11, fontweight='bold', color='#333333',
              transform=ax_table.transAxes)

# Daily stats from wx_5min: high/low temp, and each day's final running
# totals for Rain and Heated Precip (same approach as the rain gauge panel)
daily_summary = wx_5min.sort_values('TIMESTAMP').groupby('Date').agg(
    MaxTemp=('AirTF_Avg', 'max'),
    MinTemp=('AirTF_Avg', 'min'),
    TotalPrecip=('RainRunTot', 'last'),
    TotalHeatedPrecip=('HeatedRunTot', 'last'),
).reset_index()

last7 = daily_summary.tail(7).sort_values('Date', ascending=False).reset_index(drop=True)

col_x = [0.11, 0.30, 0.48, 0.67, 0.88]
headers = ['Date', 'High (\u00b0F)', 'Low (\u00b0F)', 'Precip (in)', 'Heated (in)']

header_y = 0.80
for cx, h in zip(col_x, headers):
    ax_table.text(cx, header_y, h, ha='center', va='center',
                  fontsize=8.5, fontweight='bold', color='#555555',
                  transform=ax_table.transAxes)
ax_table.plot([0.03, 0.97], [header_y - 0.06, header_y - 0.06], color='#bbbbbb',
              linewidth=0.8, transform=ax_table.transAxes)

row_ys = np.linspace(header_y - 0.14, 0.06, len(last7))
for i, (row_y, (_, row)) in enumerate(zip(row_ys, last7.iterrows())):
    if i % 2 == 1:
        ax_table.add_patch(Rectangle((0.02, row_y - 0.045), 0.96, 0.09,
                                      facecolor='#eeeeee', edgecolor='none',
                                      transform=ax_table.transAxes, zorder=0.5))
    date_str = pd.Timestamp(row['Date']).strftime(f'%b {date_flag}')
    values = [date_str, f"{row['MaxTemp']:0.1f}", f"{row['MinTemp']:0.1f}",
              f"{row['TotalPrecip']:0.2f}", f"{row['TotalHeatedPrecip']:0.2f}"]
    for cx, v in zip(col_x, values):
        ax_table.text(cx, row_y, v, ha='center', va='center',
                       fontsize=8, color='#333333', transform=ax_table.transAxes)

        

###   Current Conditions Panel   ###

ax_panel = fig.add_axes(fig_rect(panel_x, panel_y, panel_w, panel_h))
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
              high=None, low=None, unit='', unit_below=None,
              hl_decimals=1, hl_unit_below=False, value_y=None, shift=0.0):
    """Draw one label / big-value / high-low stat block in axes-fraction coords.

    value_y: put the big value at this height instead of the default
    shift:   move the big value and the High/Low up by this amount
    """
    # Label
    ax.text(x + w/2, y + h - 0.03, label, ha='center', va='top',
            fontsize=8, fontweight='bold', color='#777777', transform=ax.transAxes)
    # Big value -- moved down (closer to High/Low) to tighten the gap
    vy = value_y if value_y is not None else y + h*0.44 + 0.03 + shift
    ax.text(x + w/2, vy, value, ha='center', va='center',
            fontsize=19, fontweight='bold', color=color, transform=ax.transAxes)
    # Optional small unit directly below the value (e.g. "inHg")
    if unit_below:
        ax.text(x + w/2, vy - 0.04, unit_below, ha='center', va='top',
                fontsize=7, color=color, transform=ax.transAxes)
    # High / Low, stacked, pulled up closer to the value
    if high is not None and low is not None:
        # Unit right after the number, or on its own line below it (e.g. "inHg")
        # (with the unit on its own line, the text is anchored at the top so
        #  "High"/"Low" line up with the other cards and the extra line hangs below)
        sep = '\n' if hl_unit_below else ''
        hl_y, hl_va = (y + h*0.08 + 0.10, 'top') if hl_unit_below else (y + h*0.08 + 0.02, 'bottom')
        hl_y += shift
        ax.text(x + w*0.28, hl_y, f"High\n{high:0.{hl_decimals}f}{sep}{unit}",
                ha='center', va=hl_va, fontsize=7, color='#555555',
                transform=ax.transAxes, linespacing=1.4)
        ax.text(x + w*0.72, hl_y, f"Low\n{low:0.{hl_decimals}f}{sep}{unit}",
                ha='center', va=hl_va, fontsize=7, color='#555555',
                transform=ax.transAxes, linespacing=1.4)

stats = [
    dict(label="TEMPERATURE", col='AirTF_Avg', color='firebrick',
         value=lambda v: f"{v:0.1f}\u00b0F", unit='\u00b0', unit_below=None),
    dict(label="DEW POINT",   col='TdC_Avg',   color='teal',
         value=lambda v: f"{v:0.1f}\u00b0F",   unit='\u00b0', unit_below=None),
    dict(label="HUMIDITY",    col='RH_Avg',    color='seagreen',
         value=lambda v: f"{v:0.0f}%",         unit='%', unit_below=None),
    dict(label="PRESSURE",    col='BP_inHg',   color='indigo',
         value=lambda v: f"{v:0.2f}",          unit='inHg', unit_below='inHg',
         hl_decimals=2, hl_unit_below=True, value_at_precip=True),
]

# Grid geometry for the wide panel: 3 cards on top (Temperature, Dew Point,
# Humidity), 2 centered cards below (Pressure, Precipitation)
col_w  = 0.30
gap_x  = 0.02   # space between columns
x_left  = 0.03
x_mid   = x_left + col_w + gap_x
x_right = x_mid + col_w + gap_x
x_row2_left  = x_left + (col_w + gap_x) / 2   # row 2 is centered under row 1
x_row2_right = x_row2_left + col_w + gap_x

row_h  = 0.40
y_row1 = 0.47
y_row2 = 0.05

ROW1_SHIFT = 0.06        # top-row cards: value and High/Low moved up (closer to the label)
PRECIP_VALUE_FRAC = 0.60 # height of the daily precip value in its card (pressure matches it)

positions_rows12 = [(x_left, y_row1), (x_mid, y_row1), (x_right, y_row1),
                     (x_row2_left, y_row2)]

for (x, y), s in zip(positions_rows12, stats):
    now_val = latest[s['col']].values[0]
    hi = wx_5min.loc[today_mask, s['col']].max()
    lo = wx_5min.loc[today_mask, s['col']].min()
    stat_card(ax_panel, x, y, col_w, row_h,
              s['label'], s['value'](now_val), s['color'],
              high=hi, low=lo, unit=s['unit'], unit_below=s['unit_below'],
              hl_decimals=s.get('hl_decimals', 1),
              hl_unit_below=s.get('hl_unit_below', False),
              value_y=(y + row_h*PRECIP_VALUE_FRAC) if s.get('value_at_precip') else None,
              shift=ROW1_SHIFT if y == y_row1 else 0.0)

# --- Precipitation card (row 2, right of Pressure) ---
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
        ax.text(x + w/2, y + h*PRECIP_VALUE_FRAC, f"{rain_total:0.2f}\"", ha='center', va='center',
                fontsize=15, fontweight='bold', color='steelblue', transform=ax.transAxes)
        ax.text(x + w/2, y + h*0.28, f"{heated_total:0.2f}\"", ha='center', va='center',
                fontsize=13, fontweight='bold', color='darkorange', transform=ax.transAxes)
        ax.text(x + w/2, y + h*0.28 - 0.065, "Heated", ha='center', va='top',
                fontsize=6.5, color='darkorange', transform=ax.transAxes)
    elif show_heated_only:
        ax.text(x + w/2, y + h*PRECIP_VALUE_FRAC, f"{heated_total:0.2f}\"", ha='center', va='center',
                fontsize=17, fontweight='bold', color='darkorange', transform=ax.transAxes)
        ax.text(x + w/2, y + h*PRECIP_VALUE_FRAC - 0.075, "Heated Precip", ha='center', va='top',
                fontsize=7, color='darkorange', transform=ax.transAxes)
    else:  # regular only
        ax.text(x + w/2, y + h*PRECIP_VALUE_FRAC, f"{rain_total:0.2f}\"", ha='center', va='center',
                fontsize=17, fontweight='bold', color='steelblue', transform=ax.transAxes)
        ax.text(x + w/2, y + h*PRECIP_VALUE_FRAC - 0.075, "Precipitation", ha='center', va='top',
                fontsize=7, color='steelblue', transform=ax.transAxes)

min_temp_today = wx_5min.loc[today_mask, 'AirTF_Avg'].min()
precip_card(ax_panel, x_row2_right, y_row2, col_w, row_h, min_temp_today)


###   Current Wind Panel   ###
# (unchanged)

# Right of Current Conditions, same height (panel2_x/y/w/h set in the layout section)
ax_windpanel = fig.add_axes(fig_rect(panel2_x, panel2_y, panel2_w, panel2_h))
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



compass_cx = panel2_x + panel2_w * compass_cx_frac        # inches
compass_cy = panel2_y + panel2_h * row1_center + 0.12     # inches

ax_compass = fig.add_axes(fig_rect(
    compass_cx - compass_size/2, compass_cy - compass_size/2,
    compass_size, compass_size
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

# Arrow starts at the rim (direction the wind comes FROM) and points to the center
ax_compass.annotate('', xy=(0, 0), xytext=(needle_x*0.92, needle_y*0.92),
                     arrowprops=dict(arrowstyle='-|>', color='crimson',
                                     linewidth=2, mutation_scale=14),
                     zorder=3)
ax_compass.add_patch(Circle((0, 0), 0.06, facecolor='crimson',
                             edgecolor='none', zorder=2))

ax_windpanel.text(compass_cx_frac, row1_center - 0.24, f"{wind_dir_now:0.0f}\u00b0",
                   ha='center', va='center', fontsize=9,
                   fontweight='bold', color='#555555',
                   transform=ax_windpanel.transAxes)

wind_speed_now = latest['AveWindSp'].values[0]
# The Public table has no gust field, so use the most recent 5-min reading for "current" gust
wind_gust_now = wx_5min['WindGust'].iloc[-1]
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


###   Rain Gauge Panel   ###
 
# Top row, right end; the Rain Through table (sized in the 7-Day section) sits right below it.
panel3_y = panel_y
panel3_h = RAIN_PANEL_H
rt_y = panel3_y - GAP - rt_h

ax_rainpanel = fig.add_axes(fig_rect(RIGHT_X, panel3_y, RIGHT_W, panel3_h))
ax_rainpanel.axis('off')
ax_rainpanel.set_xlim(0, 1)
ax_rainpanel.set_ylim(0, 1)
 
ax_rainpanel.add_patch(FancyBboxPatch(
    (0.02, 0.02), 0.96, 0.96,
    boxstyle="round,pad=0.02,rounding_size=0.04",
    linewidth=1, edgecolor='#cccccc', facecolor='#f7f7f7',
    transform=ax_rainpanel.transAxes, zorder=0
))
 
ax_rainpanel.text(0.5, 0.90, "PRECIPITATION TOTALS", ha='center', va='center',
                   fontsize=11, fontweight='bold', color='#333333',
                   transform=ax_rainpanel.transAxes)
 
def rain_gauge(ax, x_center, y_bottom, y_top, width, value, max_value,
               color, label, avg=None):
    """Draw one vertical rain-gauge bar (fraction-filled tube) in axes coords."""
    # Outer tube
    ax.add_patch(FancyBboxPatch(
        (x_center - width/2, y_bottom), width, y_top - y_bottom,
        boxstyle="round,pad=0,rounding_size=0.01",
        linewidth=1.2, edgecolor='#888888', facecolor='white',
        transform=ax.transAxes, zorder=1
    ))
    # Fill, clipped to the gauge's max
    frac = max(0.0, min(1.0, value / max_value))
    fill_h = (y_top - y_bottom) * frac
    if fill_h > 0:
        ax.add_patch(FancyBboxPatch(
            (x_center - width/2, y_bottom), width, fill_h,
            boxstyle="round,pad=0,rounding_size=0.01",
            linewidth=0, facecolor=color, alpha=0.85,
            transform=ax.transAxes, zorder=2
        ))
    # Tick marks every 2 inches
    for tick in range(0, int(max_value) + 1, 2):
        ty = y_bottom + (y_top - y_bottom) * (tick / max_value)
        ax.plot([x_center - width/2 - 0.015, x_center - width/2], [ty, ty],
                color='#888888', linewidth=0.8, transform=ax.transAxes, zorder=3)
        ax.text(x_center - width/2 - 0.025, ty, f"{tick}", ha='right', va='center',
                fontsize=7.0, color='#888888', transform=ax.transAxes)
    # Average-to-date marker: dashed line across the tube, labeled on the right
    if avg is not None and not np.isnan(avg):
        ay = y_bottom + (y_top - y_bottom) * max(0.0, min(1.0, avg / max_value))
        ax.plot([x_center - width/2 - 0.01, x_center + width/2 + 0.01], [ay, ay],
                color='#333333', linewidth=1.2, linestyle=(0, (3, 1.5)),
                transform=ax.transAxes, zorder=4)
        ax.text(x_center + width/2 + 0.015, ay, f"Avg {avg:0.2f}\"",
                ha='left', va='center', fontsize=6.5, fontweight='bold',
                color='#333333', transform=ax.transAxes, zorder=4)
    # Value, above the tube
    ax.text(x_center, y_top + 0.04, f"{value:0.2f}\"", ha='center', va='bottom',
            fontsize=11, fontweight='bold', color=color, transform=ax.transAxes)
    # Caption, below the tube
    ax.text(x_center, y_bottom - 0.03, label, ha='center', va='top',
            fontsize=7.5, color='#555555', transform=ax.transAxes)
 
gauge_y_bottom, gauge_y_top = 0.20, 0.72
gauge_width = 0.32 / RIGHT_W   # 0.32" wide tubes, same as before
 
rain_gauge(ax_rainpanel, 0.30, gauge_y_bottom, gauge_y_top, gauge_width,
           water_year_precip, 16.0, 'steelblue', 'Water Year\nSince Oct 1',
           avg=rain_avg_wy)
rain_gauge(ax_rainpanel, 0.72, gauge_y_bottom, gauge_y_top, gauge_width,
           calendar_year_precip, 16.0, 'forestgreen', 'Year\nSince Jan 1',
           avg=rain_avg_cy)


###   Rain-to-Date by Year Table (below the Rain Gauge Panel)   ###

ax_raintable = fig.add_axes(fig_rect(RIGHT_X, rt_y, RIGHT_W, rt_h))
ax_raintable.axis('off')
ax_raintable.set_xlim(0, 1)
ax_raintable.set_ylim(0, 1)

# Same card x-extent as the rain gauge panel, so the widths match
ax_raintable.add_patch(FancyBboxPatch(
    (0.02, 0.02), 0.96, 0.96,
    boxstyle="round,pad=0.02,rounding_size=0.04",
    linewidth=1, edgecolor='#cccccc', facecolor='#f7f7f7',
    transform=ax_raintable.transAxes, zorder=0
))

def _rt_y(offset_from_top):
    """Offset from the top of the table (inches) -> axes fraction."""
    return 1 - offset_from_top / rt_h

thru_str = _today_norm.strftime(f'%b {date_flag}')
ax_raintable.text(0.5, _rt_y(0.144), f"RAIN THROUGH {thru_str.upper()}",
                  ha='center', va='center', fontsize=10, fontweight='bold',
                  color='#333333', transform=ax_raintable.transAxes)

rt_col_x = [0.20, 0.50, 0.79]
rt_headers = ['Year', 'Since Oct 1', 'Since Jan 1']
for cx, h in zip(rt_col_x, rt_headers):
    ax_raintable.text(cx, _rt_y(0.372), h, ha='center', va='center',
                      fontsize=7.5, fontweight='bold', color='#555555',
                      transform=ax_raintable.transAxes)
ax_raintable.plot([0.05, 0.95], [_rt_y(0.48), _rt_y(0.48)], color='#bbbbbb',
                  linewidth=0.8, transform=ax_raintable.transAxes)

def _fmt_rain(v):
    return '\u2014' if np.isnan(v) else f"{v:0.2f}"

if not rain_table_rows:
    ax_raintable.text(0.5, _rt_y(rt_title_h + rt_row_h/2), "No data",
                      ha='center', va='center', fontsize=8, color='#777777',
                      transform=ax_raintable.transAxes)

for i, (yr, wy_total, cy_total) in enumerate(rain_table_rows):
    row_y = _rt_y(rt_title_h + rt_row_h * (i + 0.5))
    if i % 2 == 1:
        band_h = rt_row_h / rt_h
        ax_raintable.add_patch(Rectangle((0.04, row_y - band_h/2), 0.92, band_h,
                                         facecolor='#eeeeee', edgecolor='none',
                                         transform=ax_raintable.transAxes, zorder=0.5))
    weight = 'bold' if i == 0 else 'normal'  # highlight the current year
    for cx, v, c in zip(rt_col_x,
                        [str(yr), _fmt_rain(wy_total), _fmt_rain(cy_total)],
                        ['#333333', 'steelblue', 'forestgreen']):
        ax_raintable.text(cx, row_y, v, ha='center', va='center',
                          fontsize=8, fontweight=weight, color=c,
                          transform=ax_raintable.transAxes)

# Bottom row (fixed slot below the 6 year rows): average of past years
avg_row_y = _rt_y(rt_title_h + rt_row_h * (RAIN_TABLE_MAX_YEARS + 0.5))
avg_line_y = avg_row_y + (rt_row_h / 2) / rt_h
ax_raintable.plot([0.05, 0.95], [avg_line_y, avg_line_y], color='#bbbbbb',
                  linewidth=0.8, transform=ax_raintable.transAxes)
for cx, v, c in zip(rt_col_x,
                    ['Avg', _fmt_rain(rain_avg_wy), _fmt_rain(rain_avg_cy)],
                    ['#333333', 'steelblue', 'forestgreen']):
    ax_raintable.text(cx, avg_row_y, v, ha='center', va='center',
                      fontsize=8, fontweight='bold', fontstyle='italic', color=c,
                      transform=ax_raintable.transAxes)




###   Output   ###
fig.savefig(file_output, dpi=300)
# plt.show()

###   Upload via FTP  ###

# import paramiko

# sftp_host = "144.17.90.20"
# sftp_port = 22
# sftp_user = "weather"
# sftp_password = "hidden"   # Replace with the actual FTP password

# local_files = [
#     r"C:/Users/weather/Weather/Weather1.png",  # matches the savefig path above
#     r"C:/Campbellsci/LoggerNet/SnowWeather_Public.dat"]
# remote_files = [
#     "/var/www/html/community/weather/images/Weather1.png",
#     "/var/www/html/community/weather/SnowWeather_Public.dat"]

# for local_file,remote_file in zip(local_files,remote_files):
#     try:
#         transport = paramiko.Transport((sftp_host, sftp_port))
#         transport.connect(username=sftp_user, password=sftp_password)
#         sftp = paramiko.SFTPClient.from_transport(transport)

#         sftp.put(local_file, remote_file)

#         sftp.close()
#         transport.close()
#         print("Upload successful")
#     except Exception as e:
#         print(f"Upload failed: {e}")