import pandas as pd
import os


def create_master(dirName):
    directory = os.path.join(os.getcwd(),dirName)
    df = pd.DataFrame()
    for file in os.listdir(directory):
        temp = pd.read_csv(os.path.join(directory,file))
        df = pd.concat([df,temp])
    print('Final DataFrame shape:', df.shape)
    df.to_csv(os.path.join(os.getcwd(),f'{dirName}.csv'),index=False)

def recursivePrintShape(dirName):
    directory = os.path.join(os.getcwd(),dirName)
    for file in os.listdir(directory):
        temp = pd.read_csv(os.path.join(directory,file))
        print(f'{file} shape:', temp.shape)

def na_check(df):
    pass

if __name__ == "__main__":
    recursivePrintShape('Hourly')
    recursivePrintShape('Daily')
    # create_master('Hourly')
    # create_master('Daily')
