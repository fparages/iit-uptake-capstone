from functools import partial

import pandas
from pathlib import Path
from pandas import Series
from pandas.core.groupby import DataFrameGroupBy


def read_data(start_date,uptill_date ):
    p = Path('/Volumes/Seagate Backup Plus Drive/Uptake_DS_practicum_Backblaze/2017/data_Q1_2017')

    yr_2017_Q1_data = pandas.DataFrame()
    for child in p.iterdir():
        if (str(child) >= (str(p) + start_date) and str(child) <= (str(p) + uptill_date)):
            print(child)
            yr_2017_Q1_data = pandas.concat([yr_2017_Q1_data, pandas.read_csv(str(child))], ignore_index=True)
    return yr_2017_Q1_data

start_date  = "/2017-01-10.csv"
uptill_date = "/2017-01-19.csv"  # change it to 15 days of 2018 & predict features of 2017 for supervised model
yr_2017_data = read_data(start_date,uptill_date)
#data_raw = yr_2017_data.drop(columns = ["date", "serial_number","model","capacity_bytes","failure", "","", "", ""])

############################## Data Preparation  #######################################################################

df_transfd = yr_2017_data.drop(["date", "model", "capacity_bytes", "smart_1_normalized","smart_2_normalized",
                                "smart_3_normalized",
                                "smart_4_normalized",
                                "smart_5_normalized",
                                "smart_7_normalized",
                                "smart_8_normalized",
                                "smart_9_normalized",
                                "smart_10_normalized",
                                "smart_11_normalized",
                                "smart_12_normalized",
                                "smart_13_normalized",
                                "smart_15_normalized",
                                "smart_22_normalized",
                                "smart_183_normalized",
                                "smart_184_normalized",
                                "smart_187_normalized",
                                "smart_188_normalized",
                                "smart_189_normalized",
                                "smart_190_normalized",
                                "smart_191_normalized",
                                "smart_192_normalized",
                                "smart_193_normalized",
                                "smart_194_normalized",
                                "smart_195_normalized",
                                "smart_196_normalized",
                                "smart_197_normalized",
                                "smart_198_normalized",
                                "smart_199_normalized",
                                "smart_200_normalized",
                                "smart_201_normalized", # has all values as "NaN"
                                "smart_220_normalized",  # has all values as "NaN" and "0"
                                "smart_222_normalized",
                                "smart_223_normalized",
                                "smart_224_normalized", # has all values as "NaN" and "0"
                                "smart_225_normalized",
                                "smart_226_normalized",
                                "smart_240_normalized",
                                "smart_241_normalized",
                                "smart_242_normalized",
                                "smart_250_normalized",
                                "smart_251_normalized",
                                "smart_252_normalized",
                                "smart_254_normalized",
                                "smart_255_normalized",
                                ], axis=1)

print("Grouping Starts")
# first we will group the drive data by serial number
data_by_drive: DataFrameGroupBy = df_transfd.groupby("serial_number")
#import ipdb;ipdb.set_trace()
print("Grouping Finished")
# to predict failure 5 days prior we will shift the lable in column failure by 5 rows to do 5 day ahead prediction
sign = lambda x: (1, -1)[x < 0]
# following is the user-defined function for curve shifting


def curve_shift(df, shift_by):
    '''
    This function is used to shift the binary labels in dataframe. The curve shift will be with respect to the 1s.
    For example, if shift is -5, the following process will happen: if row n is labeled as 1, then
    - Make row (n+shift_by):(n+shift_by-1) = 1.
    - Remove row n.
    i.e. the labels will be shifted up to 5 rows up.

    Inputs:
    df       A pandas dataframe with a binary labeled column.
             This labeled column should be named as 'y'.
    shift_by An integer denoting the number of rows to shift.

    Output
    df       A dataframe with the binary labels shifted by shift.
    '''

    vector: Series = df['failure'].copy()
    for s in range(abs(shift_by)):
        tmp = vector.shift(sign(shift_by))
        tmp = tmp.fillna(0)
        vector += tmp
    labelcol = 'failure'
    # Add vector to the df
    df.insert(loc=0, column=labelcol + 'tmp', value=vector)
    # Remove the rows with labelcol == 1.
    #df = df.drop(df[df[labelcol] == 1].index)
    # Drop labelcol and rename the tmp col as labelcol
    df = df.drop(labelcol, axis=1)
    df = df.rename(columns={labelcol + 'tmp': labelcol})
    # Make the labelcol binary
    df.loc[df[labelcol] > 0, labelcol] = 1
    return df


curve_shift_partial = partial(curve_shift, shift_by=-3)
print("curve shifting begins")
obs_drives = pandas.DataFrame() # this is to observe behaviour of failed drives for 6 days before failure
aa = data_by_drive.apply(curve_shift_partial)

#### write into csv using
aa.to_csv("/Users/shikha/Documents/iit-uptake-capstone/failure_modelling/data/processed/shifted_data_2017_10to19.csv", index=False)