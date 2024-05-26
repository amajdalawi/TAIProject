import pandas as pd
from pathlib import Path
import re
import pprint


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


get_set_of_datetimes(forecast_dataset())