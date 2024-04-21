
import pandas as pd
import os 

df = pd.read_csv(os.path.join(os.getcwd(),'data','Hourly_clean.csv'))

def check(df):
    print(df.head())
    print(df.info())
    print(df.describe())

def clean(df,save_path ='Hourly_clean2.csv'):
    df['datetime'] = pd.to_datetime(df['Date'] + df['Time'], format='%m/%d/%Y%H:%M')
    df = df.drop(['Date','Time'])
    print(df.head())
    df.to_csv(os.path.join(os.getcwd(),'data',save_path),index = False)

### Structure ###
#
# check(df)
# clean(df)
# - save
