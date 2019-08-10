import pandas
from pathlib import Path

###################################### Function to read data ###########################################################
def read_data(start_date,uptill_date ):
    p = Path('/Volumes/Seagate Backup Plus Drive/Uptake_DS_practicum_Backblaze/2017/data_Q1_2017')

    yr_2017_Q1_data = pandas.DataFrame()
    for child in p.iterdir():
        if (str(child) <= (str(p) + uptill_date)):
            print(child)
            yr_2017_Q1_data = pandas.concat([yr_2017_Q1_data, pandas.read_csv(str(child))])
    return yr_2017_Q1_data