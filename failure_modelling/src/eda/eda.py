import pandas
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.core.groupby import GroupBy, DataFrameGroupBy
from matplotlib.axes import Axes
import pyarrow.feather as feather

###################################### Reading data  for 2017 and 2015 Q3, Q4  #########################################
p = Path('/Volumes/Seagate Backup Plus Drive/Uptake_DS_practicum_Backblaze/2017/data_Q1_2017')
start_date  = "/2017-01-01.csv"
uptill_date = "/2017-01-31.csv"

yr_2017_Q1_data = pandas.DataFrame()

for child in p.iterdir():
    if(str(child) <= (str(p) + uptill_date)):
        print(child)
        yr_2017_Q1_data = pandas.concat([yr_2017_Q1_data, pandas.read_csv(str(child))])


###################################### Data Preprocessing for 2017 and 2015 Q3, Q4  #####################################
print(yr_2017_Q1_data.columns)

#series = feather.from_csv("/Volumes/Seagate Backup Plus Drive/Uptake_DS_practicum_Backblaze/data_Q1_2018/2018-01-02.csv", header =0)
#Q1_feather = feather.FeatherReader.read_pandas()

print(yr_2017_Q1_data.head())  # will print only first column containing attribute names of each column
print(yr_2017_Q1_data.loc[0]) # will print first data row
print(yr_2017_Q1_data.dtypes)  # to know the object type of each column
print("The size of data for year 2017 is :",yr_2017_Q1_data.shape)
print("The number of drives represented by the data are ", len(yr_2017_Q1_data["serial_number"].unique()))

yr_2017_one_day = pandas.read_csv("/Volumes/Seagate Backup Plus Drive/Uptake_DS_practicum_Backblaze/2017/data_Q1_2017/2017-01-01.csv") 
len(yr_2017_one_day) == len(yr_2017_one_day["serial_number"].unique()) # test to check if serial number column is enough as identifier

percent_missing = yr_2017_Q1_data.isnull().sum() * 100 / len(yr_2017_Q1_data)
missing_value_df = pandas.DataFrame({'column_name': yr_2017_Q1_data.columns, 'percent_missing': percent_missing})
missing_value_df.sort_values('percent_missing', inplace=True)

neglected_df_records = []
for index,row in missing_value_df.iterrows():
    if (row["percent_missing"]>75.0):
        print (row["column_name"], row["percent_missing"])
        neglected_df_records.append({'column_name': row["column_name"], 'percent_missing': row["percent_missing"]})

neglected_df = pandas.DataFrame.from_records(neglected_df_records)  # this dataframe contains the smart stats with 75% empty values
# gives 36 columns, left are ( 2015-17 contains
# 90 cols of raw and norm smart stats)90-36, 54 columns left


yr_2017_Q1_data["failure"] = yr_2017_Q1_data["failure"].astype('category')
data_by_failure: DataFrameGroupBy = yr_2017_Q1_data.groupby("failure")
ax = sns.countplot(x = yr_2017_Q1_data["failure"], data = yr_2017_Q1_data)
plt.show()

data_by_failure = data_by_failure.get_group(1)
axs: Axes = plt.gca()
axs.hist(data_by_failure['failure'])
axs.set_xlabel("failure column values")
axs.set_ylabel("failure counts")
plt.show()
print("The number of failures cases in data is ", len(data_by_failure))

######### exploraing data of one drive #########
data_by_drive: DataFrameGroupBy = yr_2017_Q1_data.groupby("serial_number")
drive_list = list(data_by_drive.groups.keys()) # get all the serial number for operational drives in a list
data_one_drive = data_by_drive.get_group("Z30271GD")   # take into list and iterate
axs: Axes = plt.gca()
ax = sns.lineplot(x=data_one_drive["date"], y=data_one_drive["smart_5_raw"],  data=data_one_drive)
ax = sns.lineplot(x=data_one_drive["date"], y=data_one_drive["smart_188_raw"], data=data_one_drive)
plt.show()

axs: Axes = plt.gca()
ax = sns.lineplot(x=data_one_drive["date"], y=data_one_drive["smart_1_raw"], label="smart_1_raw", data=data_one_drive)
ax = sns.lineplot(x=data_one_drive["date"], y=data_one_drive["smart_2_raw"], label="smart_2_raw", data=data_one_drive)
ax = sns.lineplot(x=data_one_drive["date"], y=data_one_drive["smart_3_raw"], label="smart_3_raw", data=data_one_drive)
ax = sns.lineplot(x=data_one_drive["date"], y=data_one_drive["smart_4_raw"], label="smart_4_raw", data=data_one_drive)
ax = sns.lineplot(x=data_one_drive["date"], y=data_one_drive["smart_5_raw"], label="smart_5_raw", data=data_one_drive)
ax = sns.lineplot(x=data_one_drive["date"], y=data_one_drive["smart_7_raw"], label="smart_7_raw", data=data_one_drive)
ax = sns.lineplot(x=data_one_drive["date"], y=data_one_drive["smart_8_raw"], label="smart_8_raw", data=data_one_drive)
ax = sns.lineplot(x=data_one_drive["date"], y=data_one_drive["smart_9_raw"], label="smart_9_raw", data=data_one_drive)
ax = sns.lineplot(x=data_one_drive["date"], y=data_one_drive["smart_10_raw"], label="smart_10_raw", data=data_one_drive)
ax = sns.lineplot(x=data_one_drive["date"], y=data_one_drive["smart_12_raw"], label="smart_12_raw", data=data_one_drive)
ax = sns.lineplot(x=data_one_drive["date"], y=data_one_drive["smart_183_raw"], label="smart_183_raw", data=data_one_drive)
ax = sns.lineplot(x=data_one_drive["date"], y=data_one_drive["smart_188_raw"],label="smart_188_raw", data=data_one_drive)
plt.ylabel("One drive's monthy stats")
plt.xlabel("date")
plt.legend()
plt.show()

axs: Axes = plt.gca()
ax = sns.lineplot(x=data_one_drive["date"], y=data_one_drive["smart_12_raw"], label="smart_12_raw", data=data_one_drive)
ax = sns.lineplot(x=data_one_drive["date"], y=data_one_drive["smart_183_raw"], label="smart_183_raw", data=data_one_drive)
ax = sns.lineplot(x=data_one_drive["date"], y=data_one_drive["smart_188_raw"],label="smart_188_raw", data=data_one_drive)
plt.ylabel("One drive's monthy stats")
plt.xlabel("date")
plt.legend()
plt.show()


# remove neglected cols
# multiple series in one
# prob dist


# lag plot
# arima