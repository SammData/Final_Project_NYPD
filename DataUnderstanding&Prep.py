# %% read csv file
import pandas as pd

df = pd.read_csv("2012.csv")

# %% Quick summary: index and column data types, non-null values and memory usage
df.info()
# %% Data shape & descriptive statistics summarizing the central tendency & dispersion
df.describe()
# %% statistics for numerical columns only
import numpy as np
df.describe(include=[np.number])
# %% statistics for string/object type columns only
df.describe(include=['O'])

# %% make sure numeric columns have numbers, remove rows that are not
cols = [
    "perobs",
    "perstop",
    "age",
    "weight",
    "ht_feet",
    "ht_inch",
    "datestop",
    "timestop",
    "xcoord",
    "ycoord",
]

for col in cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()


# %% make datetime column
df["datestop"] = df["datestop"].astype(str).str.zfill(8)
df["timestop"] = df["timestop"].astype(str).str.zfill(4)

from datetime import datetime

def make_datetime(datestop, timestop):
    year = int(datestop[-4:])
    month = int(datestop[:2])
    day = int(datestop[2:4])

    hour = int(timestop[:2])
    minute = int(timestop[2:])

    return datetime(year, month, day, hour, minute)


df["datetime"] = df.apply(
    lambda r: make_datetime(r["datestop"], r["timestop"]),
    axis=1
)

# %% # %% Total time spent on observation before taking any action and total time of stop for suspect
total_time_of_obs = df["perobs"].sum()
total_time_of_stop = df["perstop"].sum()

print(total_time_of_obs)
print(total_time_of_stop)

# %% Man hours spent on this program
# Assumption: Average total number of officers in New York City:36000
# Officer works for 5 days a week for 8 hours a day

working_days = 366-(52*2)
total_amount_of_man_hours_NYPD_per_year = 36000 * working_days * 8

total_time_obs_stop_per_year = (total_time_of_obs + total_time_of_stop)/60
hours_spent_obs_stop_per_year = (total_time_obs_stop_per_year)/36000
percent_of_man_hour_spent_on_obs_stop = ((total_time_obs_stop_per_year)/total_amount_of_man_hours_NYPD_per_year)*100
print(f"{percent_of_man_hour_spent_on_obs_stop :.2f} %")

# %% convert all value to label in the dataframe, remove rows that cannot be mapped
import numpy as np
from tqdm import tqdm

value_label = pd.read_excel(
    "2012 SQF File Spec.xlsx",
    sheet_name="Value Labels",
    skiprows=range(4)
)
value_label["Field Name"] = value_label["Field Name"].fillna(
    method="ffill"
)
value_label["Field Name"] = value_label["Field Name"].str.lower()
value_label["Value"] = value_label["Value"].fillna(" ")
value_label = value_label.groupby("Field Name").apply(
    lambda x: dict([(row["Value"], row["Label"]) for row in x.to_dict("records")])
)

cols = [col for col in df.columns if col in value_label]

for col in tqdm(cols):
    df[col] = df[col].apply(
        lambda val: value_label[col].get(val, np.nan)
    )

df["trhsloc"] = df["trhsloc"].fillna("P (unknown)")
df = df.dropna()


# %% convert xcoord and ycoord to (lon, lat)
import pyproj

srs = "+proj=lcc +lat_1=41.03333333333333 +lat_2=40.66666666666666 +lat_0=40.16666666666666 +lon_0=-74 +x_0=300000.0000000001 +y_0=0 +ellps=GRS80 +datum=NAD83 +to_meter=0.3048006096012192 +no_defs"
p = pyproj.Proj(srs)

df["coord"] = df.apply(
    lambda r: p(r["xcoord"], r["ycoord"], inverse=True), axis=1
)

# %% convert height in feet/inch to cm
df["height"] = (df["ht_feet"] * 12 + df["ht_inch"]) * 2.54


# %% remove outlier
df = df[(df["weight"] <= 350) & (df["weight"] >= 50)]
df = df[(df["age"] <= 100) & (df["age"] >= 10)]

# %% delete columns that are not needed
df = df.drop(
    columns=[
        # processed columns
        "datestop",
        "timestop",
        "ht_feet",
        "ht_inch",
        "xcoord",
        "ycoord",

        # not useful
        "year",
        "recstat",
        "crimsusp",
        "dob",
        "ser_num",
        "arstoffn",
        "sumoffen",
        "compyear",
        "comppct",
        "othfeatr",
        "adtlrept",
        "dettypcm",
        "linecm",
        "repcmd",
        "revcmd",

        # location of stop
        # only use coord and city
        "addrtyp",
        "rescode",
        "premtype",
        "premname",
        "addrnum",
        "stname",
        "stinter",
        "crossst",
        "aptnum",
        "state",
        "zip",
        "addrpct",
        "sector",
        "beat",
        "post",
    ]
)

# %% Highest stop Precincts
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df, y="pct", order=df["pct"].value_counts().iloc[:5].index)
ax.set(xlabel='2012 Total Stops')
ax.set_title('Highest Stop Precincts')
plt.savefig('Person_Stopped.png')

# %% Lowest stop Precincts
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df, y="pct", order=df["pct"].value_counts().iloc[71:79].index)
ax.set(xlabel='2012 Total Stops')
ax.set_title('Lowest Stop Precincts')
plt.savefig('Person_Stopped.png')

# %% Person Stopped by Race
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df,x="race")
ax.set(ylabel="2012 Total Stops ")
ax.set(xlabel="Race")
ax.set_title('Persons Stopped by Race')
ax.set_xticklabels(
    ax.get_xticklabels(), rotation=45, fontsize=6
)
for p in ax.patches:
    percentage = p.get_height() * 100 / df.shape[0]
    txt = f"{percentage:.1f}%"
    x_loc = p.get_x()
    y_loc = p.get_y() + p.get_height()
    ax.text(x_loc, y_loc, txt)
plt.savefig('Person_Stopped.png')

# %% Highest stops by Race and Age group
import matplotlib.pyplot as plt
import pylab as plot
plt.figure(figsize=(8,12))
ax = sns.countplot(df["age"], hue=df["race"])
ax.set(xlabel='Age')
ax.set(ylabel='2012 Total Stops')
ax.set_title('Highest Age Group Stops by Race')

params = {'legend.fontsize': 5,
          'legend.handlelength': 1}
plot.rcParams.update(params)

# %% Frisked percentage between Black and White races in each Borough
is_frisked = df["arstmade"] == "YES"
in_selected_races = df["race"].isin(["BLACK", "WHITE"])
df_frisked_selected = df[is_frisked & in_selected_races]
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df_frisked_selected, x="race", hue="city")
ax.set(ylabel="2012 Total Stops ")
ax.set(xlabel="Race",)
ax.set_title('Frisked by Race')
ax.set_xticklabels(
    ax.get_xticklabels(), rotation=45, fontsize=6
)

for p in ax.patches:
    percentage = p.get_height() * 100 / df.shape[0]
    txt = f"{percentage:.1f}%"
    x_loc = p.get_x()
    y_loc = p.get_y() + p.get_height()
    ax.text(x_loc, y_loc, txt)
plt.savefig('Frisked_Race.png')


# %%
nyc = (40.730610, -73.935242)

m = folium.Map(location=nyc)


# %%
for coord in df.loc[df["detailCM"]=="MURDER", "coord"]:
    folium.CircleMarker(
        location=(coord[1], coord[0]), radius=1, color="red"
    ).add_to(m)

m


# %%
for coord in df.loc[df["detailCM"]=="MURDER", "coord"]:
    folium.CircleMarker(
        location=(coord[1], coord[0]), radius=1, color="yellow"
    ).add_to(m)

m

# %% show all columns

pd.options.display.max_columns = None
pd.options.display.max_rows = None
df.head()

# %% Pearson correlation
import seaborn as sb
pearsoncorr = df.corr(method='pearson')
forces = ["pf_hands", "pf_wall", "pf_grnd", "pf_drwep", "pf_hcuff", "pf_pepsp"]
result = ["arstmade", "sumissue", "searched", "pistol", "frisked", "contrabn"]

subset = df[forces + result]
subset = (subset == "YES").astype(int)

plt.figure(figsize=(8,9))
ax = sb.heatmap(subset.corr(), annot=True)
ax.set_xlabel('forces', fontsize=12)
ax.set_ylabel('result', fontsize=12)
ax.set_title('Heatmap Correlation')
plt.savefig('Correlation.png')

# %% Scatterplt (arstmade vs searched)
#import matplotlib.pyplot as plt
import numpy as np




# %%
df.to_pickle("sqf.pkl")
