import pandas as pd
from pathlib import Path
import re
import pprint
import matplotlib.dates as mdates 

resources_folder = Path("./resources/")

def load_dataset(csv_file: Path):
    # function that takes forecast csv file and removes first 50 rows and the nreturns df
    df = pd.read_csv(csv_file)
    # delete the first 50 rows
    return df.iloc[50:]

def forecast_dataset() -> pd.DataFrame:
    # function that returns a dataframe with merged forecast files 
    list_of_dfs = []
    for i in resources_folder.glob("Forecast*"):
        list_of_dfs.append(load_dataset(i))
    # Merge DFs into a single one
    df_final = pd.concat(list_of_dfs, ignore_index=True)
    # Sort the DF by time
    df_final = df_final.sort_values(by='time')
    # Reset the indecies of the Df to 0
    df_final = df_final.reset_index(drop=True)
    # pprint.pprint(df_final)
    # df_final.to_csv('./heh.csv')
    return df_final

def get_set_of_datetimes(df: pd.DataFrame) -> set:
    # function taht takes a dataframe as input (usually the dataframe comprising the forecast) and gets teh datetimes and appends them into a set
    s = set()
    for dt in df['time']:
        s.add(dt)
    # pprint.pprint(s)
    print(len(s))
    return s

def create_realtemp_df(path_vlinder: Path, s: set)->pd.DataFrame:
    file = Path(path_vlinder)
    df = pd.read_csv(file)
    count = 0
    real_temps_df = pd.DataFrame(columns=['real_temp'])
    for index, row in df.iterrows():
        if row['datetime'] in s:
            # print(f"Date: {row['datetime']} found and its temp is {row['temp']}")
            count += 1
            for _ in range(0,50):
                new_row = pd.DataFrame({"real_temp": [round(row['temp'] + 273.15,2)]})
                real_temps_df = pd.concat([real_temps_df,new_row], ignore_index=True)
    
    # print(count)
    # pprint.pprint(real_temps_df)
    return real_temps_df


def get_combined_df():
    df = forecast_dataset()
    s = get_set_of_datetimes(df)
    real_temps_df = create_realtemp_df("resources\Vlinder_2024.csv",s)
    # concat real_temp values to the larger forecast df
    df['real_temp'] = real_temps_df['real_temp']
    #turn time into numerical value
    df['time'] = pd.to_datetime(df['time'])
    for index, row in df.iterrows():
        df.at[index, 'time'] = mdates.date2num(df.at[index, 'time'])
    # pprint.pprint(df)
    return df

def get_pruned_df():
    df = get_combined_df()
    df = df.drop(columns=['number','longitude','latitude','cin'])
    df.to_csv("see.csv")
    return df

# df = get_combined_df()
# df.to_csv('wow.csv')