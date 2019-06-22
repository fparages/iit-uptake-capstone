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
uptill_date = "/2017-03-31.csv"

yr_2017_Q1_data = pandas.DataFrame()

for child in p.iterdir():
    if(str(child) <= (str(p) + uptill_date)):
        print(child)
        yr_2017_Q1_data = pandas.concat([yr_2017_Q1_data, pandas.read_csv(str(child))])


###################################### Data Preprocessing for 2017 and 2015 Q3, Q4  ####################################
print(yr_2017_Q1_data.columns)
print(yr_2017_Q1_data.head())  # will print only first column containing attribute names of each column
print(yr_2017_Q1_data.loc[0]) # will print first data row
print(yr_2017_Q1_data.dtypes)  # to know the object type of each column
print("The size of data for year 2017 is :",yr_2017_Q1_data.shape)
print("The number of drives represented by the data are ", len(yr_2017_Q1_data["serial_number"].unique()))

yr_2017_one_day = pandas.read_csv("/Volumes/Seagate Backup Plus Drive/Uptake_DS_practicum_Backblaze/2017/data_Q1_2017/2017-01-01.csv") 
len(yr_2017_one_day) == len(yr_2017_one_day["serial_number"].unique()) # test to check if serial number column is enough as identifier

###################################### Data Exploration for 2017 and 2015 Q3, Q4  ####################################
data_raw_n_norm = yr_2017_Q1_data.drop(columns = ["date", "serial_number","model","capacity_bytes","failure"])
percent_missing = data_raw_n_norm.isnull().sum() * 100 / len(data_raw_n_norm)
missing_value_df = pandas.DataFrame({'column_name': data_raw_n_norm.columns, 'percent_missing': round(percent_missing,2)})
missing_value_df.sort_values('percent_missing', inplace=True)

ax = sns.countplot(x = missing_value_df["percent_missing"],data = missing_value_df)
ax.set_xticklabels(ax.get_xticklabels(),rotation = 45, horizontalalignment='right')
for p in ax.patches:
 ax.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha ='center', va ='center',
             xytext =(0, 10), textcoords ='offset points')
ax.set_ylabel("Number of columns")
plt.show()


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
failed_drive_list = list(data_by_failure.groups.keys())

######### exploring data of one drive #########
data_by_drive: DataFrameGroupBy = yr_2017_Q1_data.groupby("serial_number")
drive_list = list(data_by_drive.groups.keys()) # get all the serial number for operational drives in a list
data_one_drive = data_by_drive.get_group("Z30271GD")   # take into list and iterate
axs: Axes = plt.gca()
ax = sns.lineplot(x=data_one_drive["date"], y=data_one_drive["smart_5_raw"],  data=data_one_drive)
ax = sns.lineplot(x=data_one_drive["date"], y=data_one_drive["smart_188_raw"], data=data_one_drive)
plt.show()


axs: Axes = plt.gca()
ax = sns.lineplot(x=data_one_drive["date"], y=data_one_drive["smart_12_raw"], label="smart_12_raw", data=data_one_drive)
ax = sns.lineplot(x=data_one_drive["date"], y=data_one_drive["smart_183_raw"], label="smart_183_raw", data=data_one_drive)
ax = sns.lineplot(x=data_one_drive["date"], y=data_one_drive["smart_188_raw"],label="smart_188_raw", data=data_one_drive)
plt.ylabel("One drive's monthy stats")
plt.xlabel("date")
plt.legend()
plt.show()

failure_instances = yr_2017_Q1_data[yr_2017_Q1_data["failure"] == 1]
if((yr_2017_Q1_data["failure"] == 1)) : print(yr_2017_Q1_data["serial_number"], yr_2017_Q1_data["date"],yr_2017_Q1_data["failure"])


# the one failed on 2017-01-20 -> Z3051NNA
f_drive = data_by_drive.get_group("Z3051NNA") # failed 20th day


f_drive = data_by_drive.get_group("Z4D02HXT") # failed 5th day of 2017
axs: Axes = plt.gca()
ax = sns.lineplot(x=f_drive["date"], y=f_drive["smart_1_raw"], label="smart_1_raw", data=f_drive)
ax = sns.lineplot(x=f_drive["date"], y=f_drive["smart_2_raw"], label="smart_2_raw", data=f_drive)
ax = sns.lineplot(x=f_drive["date"], y=f_drive["smart_3_raw"], label="smart_3_raw", data=f_drive)
ax = sns.lineplot(x=f_drive["date"], y=f_drive["smart_4_raw"], label="smart_4_raw", data=f_drive)
ax = sns.lineplot(x=f_drive["date"], y=f_drive["smart_5_raw"], label="smart_5_raw", data=f_drive)
ax = sns.lineplot(x=f_drive["date"], y=f_drive["smart_7_raw"], label="smart_7_raw", data=f_drive)
ax = sns.lineplot(x=f_drive["date"], y=f_drive["smart_8_raw"], label="smart_8_raw", data=f_drive)
ax = sns.lineplot(x=f_drive["date"], y=f_drive["smart_9_raw"], label="smart_9_raw", data=f_drive)
ax = sns.lineplot(x=f_drive["date"], y=f_drive["smart_10_raw"], label="smart_10_raw", data=f_drive)
plt.ylabel("One drive's 5 days uptill failure stats ")
plt.xlabel("date")
plt.legend()
plt.show()

axs: Axes = plt.gca()
ax = sns.lineplot(x=f_drive["date"], y=f_drive["smart_12_raw"], label="smart_12_raw", data=f_drive)
ax = sns.lineplot(x=f_drive["date"], y=f_drive["smart_183_raw"], label="smart_183_raw", data=f_drive)
ax = sns.lineplot(x=f_drive["date"], y=f_drive["smart_184_raw"], label="smart_184_raw", data=f_drive)
ax = sns.lineplot(x=f_drive["date"], y=f_drive["smart_187_raw"], label="smart_187_raw", data=f_drive)
ax = sns.lineplot(x=f_drive["date"], y=f_drive["smart_188_raw"], label="smart_188_raw", data=f_drive)
ax = sns.lineplot(x=f_drive["date"], y=f_drive["smart_189_raw"], label="smart_189_raw", data=f_drive)
ax = sns.lineplot(x=f_drive["date"], y=f_drive["smart_190_raw"], label="smart_190_raw", data=f_drive)
ax = sns.lineplot(x=f_drive["date"], y=f_drive["smart_191_raw"], label="smart_191_raw", data=f_drive)
ax = sns.lineplot(x=f_drive["date"], y=f_drive["smart_192_raw"], label="smart_192_raw", data=f_drive)
plt.ylabel("One drive's 5 days uptill failure stats ")
plt.xlabel("date")
plt.legend()
plt.show()

axs: Axes = plt.gca()
ax = sns.lineplot(x=f_drive["date"], y=f_drive["smart_193_raw"], label="smart_193_raw", data=f_drive)
ax = sns.lineplot(x=f_drive["date"], y=f_drive["smart_194_raw"], label="smart_194_raw", data=f_drive)
ax = sns.lineplot(x=f_drive["date"], y=f_drive["smart_196_raw"], label="smart_196_raw", data=f_drive)
ax = sns.lineplot(x=f_drive["date"], y=f_drive["smart_197_raw"], label="smart_197_raw", data=f_drive)
ax = sns.lineplot(x=f_drive["date"], y=f_drive["smart_198_raw"], label="smart_198_raw", data=f_drive)
ax = sns.lineplot(x=f_drive["date"], y=f_drive["smart_199_raw"], label="smart_199_raw", data=f_drive)
ax = sns.lineplot(x=f_drive["date"], y=f_drive["smart_240_raw"], label="smart_240_raw", data=f_drive)
ax = sns.lineplot(x=f_drive["date"], y=f_drive["smart_241_raw"], label="smart_241_raw", data=f_drive)
ax = sns.lineplot(x=f_drive["date"], y=f_drive["smart_242_raw"], label="smart_242_raw", data=f_drive)
plt.ylabel("One drive's 5 days uptill failure stats ")
plt.xlabel("date")
plt.legend()
plt.show()

axs: Axes = plt.gca()
ax = sns.lineplot(x=f_drive["date"], y=f_drive["smart_191_raw"], label="smart_191_raw", data=f_drive)
plt.ylabel("One drive's 5 days uptill failure stats ")
plt.xlabel("date")
plt.legend()
plt.show()

axs: Axes = plt.gca()
ax = sns.lineplot(x=f_drive["date"], y=f_drive["smart_190_raw"], label="smart_191_raw", data=f_drive)
plt.ylabel("One drive's 5 days uptill failure stats ")
plt.xlabel("date")
plt.legend()
plt.show()

axs: Axes = plt.gca()
ax = sns.lineplot(x=f_drive["date"], y=f_drive["smart_240_raw"], label="smart_240_raw", data=f_drive)
plt.ylabel("One drive's 5 days uptill failure stats ")
plt.xlabel("date")
plt.legend()
plt.show()


axs: Axes = plt.gca()
ax = sns.lineplot(x=f_drive["date"], y=f_drive["smart_194_raw"], label="smart_194_raw", data=f_drive)
plt.ylabel("One drive's 5 days uptill failure stats ")
plt.xlabel("date")
plt.legend()
plt.show()

f_drive = data_by_drive.get_group("Z3051NNA")


# whisker plots for ranges of smart stats
sns.set(style="ticks", palette="pastel")
sns.boxplot(x="smart_1_raw", y = "failure",
            hue="failure", palette=["m", "g"],
            data=yr_2017_Q1_data)
sns.despine(offset=10, trim=True)
plt.show()

sns.set(style="ticks", palette="pastel")
sns.boxplot(x="smart_2_raw", y = "failure",
            hue="failure", palette=["m", "g"],
            data=yr_2017_Q1_data)
sns.despine(offset=10, trim=True)
plt.show()

sns.set(style="ticks", palette="pastel")
sns.boxplot(x="smart_3_raw", y = "failure",
            hue="failure", palette=["m", "g"],
            data=yr_2017_Q1_data)
sns.despine(offset=10, trim=True)
plt.show()

sns.set(style="ticks", palette="pastel")
sns.boxplot(x="smart_4_raw", y = "failure",
            hue="failure", palette=["m", "g"],
            data=yr_2017_Q1_data)
sns.despine(offset=10, trim=True)
plt.show()

sns.set(style="ticks", palette="pastel")
sns.boxplot(x="smart_5_raw", y = "failure",
            hue="failure", palette=["m", "g"],
            data=yr_2017_Q1_data)
sns.despine(offset=10, trim=True)
plt.show()

sns.set(style="ticks", palette="pastel")
sns.boxplot(x="smart_7_raw", y = "failure",
            hue="failure", palette=["m", "g"],
            data=yr_2017_Q1_data)
sns.despine(offset=10, trim=True)
plt.show()

sns.set(style="ticks", palette="pastel")
sns.boxplot(x="smart_8_raw", y = "failure",
            hue="failure", palette=["m", "g"],
            data=yr_2017_Q1_data)
sns.despine(offset=10, trim=True)
plt.show()

sns.set(style="ticks", palette="pastel")
sns.boxplot(x="smart_9_raw", y = "failure",
            hue="failure", palette=["m", "g"],
            data=yr_2017_Q1_data)
sns.despine(offset=10, trim=True)
plt.show()

sns.set(style="ticks", palette="pastel")
sns.boxplot(x="smart_10_raw", y = "failure",
            hue="failure", palette=["m", "g"],
            data=yr_2017_Q1_data)
sns.despine(offset=10, trim=True)
plt.show()

sns.set(style="ticks", palette="pastel")
sns.boxplot(x="smart_12_raw", y = "failure",
            hue="failure", palette=["m", "g"],
            data=yr_2017_Q1_data)
sns.despine(offset=10, trim=True)
plt.show()

sns.set(style="ticks", palette="pastel")
sns.boxplot(x="smart_183_raw", y = "failure",
            hue="failure", palette=["m", "g"],
            data=yr_2017_Q1_data)
sns.despine(offset=10, trim=True)
plt.show()

sns.set(style="ticks", palette="pastel")
sns.boxplot(x="smart_184_raw", y = "failure",
            hue="failure", palette=["m", "g"],
            data=yr_2017_Q1_data)
sns.despine(offset=10, trim=True)
plt.show()

sns.set(style="ticks", palette="pastel")
sns.boxplot(x="smart_187_raw", y = "failure",
            hue="failure", palette=["m", "g"],
            data=yr_2017_Q1_data)
sns.despine(offset=10, trim=True)
plt.show()

sns.set(style="ticks", palette="pastel")
sns.boxplot(x="smart_188_raw", y = "failure",
            hue="failure", palette=["m", "g"],
            data=yr_2017_Q1_data)
sns.despine(offset=10, trim=True)
plt.show()
sns.set(style="ticks", palette="pastel")
sns.boxplot(x="smart_189_raw", y = "failure",
            hue="failure", palette=["m", "g"],
            data=yr_2017_Q1_data)
sns.despine(offset=10, trim=True)
plt.show()

sns.set(style="ticks", palette="pastel")
sns.boxplot(x="smart_190_raw", y = "failure",
            hue="failure", palette=["m", "g"],
            data=yr_2017_Q1_data)
sns.despine(offset=10, trim=True)
plt.show()

sns.set(style="ticks", palette="pastel")
sns.boxplot(x="smart_191_raw", y = "failure",
            hue="failure", palette=["m", "g"],
            data=yr_2017_Q1_data)
sns.despine(offset=10, trim=True)
plt.show()


sns.set(style="ticks", palette="pastel")
sns.boxplot(x="smart_192_raw", y = "failure",
            hue="failure", palette=["m", "g"],
            data=yr_2017_Q1_data)
sns.despine(offset=10, trim=True)
plt.show()

sns.set(style="ticks", palette="pastel")
sns.boxplot(x="smart_193_raw", y = "failure",
            hue="failure", palette=["m", "g"],
            data=yr_2017_Q1_data)
sns.despine(offset=10, trim=True)
plt.show()

sns.set(style="ticks", palette="pastel")
sns.boxplot(x="smart_194_raw", y = "failure",
            hue="failure", palette=["m", "g"],
            data=yr_2017_Q1_data)
sns.despine(offset=10, trim=True)
plt.show()

sns.set(style="ticks", palette="pastel")
sns.boxplot(x="smart_196_raw", y = "failure",
            hue="failure", palette=["m", "g"],
            data=yr_2017_Q1_data)
sns.despine(offset=10, trim=True)
plt.show()

sns.set(style="ticks", palette="pastel")
sns.boxplot(x="smart_197_raw", y = "failure",
            hue="failure", palette=["m", "g"],
            data=yr_2017_Q1_data)
sns.despine(offset=10, trim=True)
plt.show()
sns.set(style="ticks", palette="pastel")
sns.boxplot(x="smart_198_raw", y = "failure",
            hue="failure", palette=["m", "g"],
            data=yr_2017_Q1_data)
sns.despine(offset=10, trim=True)
plt.show()

sns.set(style="ticks", palette="pastel")
sns.boxplot(x="smart_199_raw", y = "failure",
            hue="failure", palette=["m", "g"],
            data=yr_2017_Q1_data)
sns.despine(offset=10, trim=True)
plt.show()

sns.set(style="ticks", palette="pastel")
sns.boxplot(x="smart_240_raw", y = "failure",
            hue="failure", palette=["m", "g"],
            data=yr_2017_Q1_data)
sns.despine(offset=10, trim=True)
plt.show()
sns.set(style="ticks", palette="pastel")
sns.boxplot(x="smart_241_raw", y = "failure",
            hue="failure", palette=["m", "g"],
            data=yr_2017_Q1_data)
sns.despine(offset=10, trim=True)
plt.show()

sns.set(style="ticks", palette="pastel")
sns.boxplot(x="smart_242_raw", y = "failure",
            hue="failure", palette=["m", "g"],
            data=yr_2017_Q1_data)
sns.despine(offset=10, trim=True)
plt.show()


failed_drive_list = list(failure_instances["serial_number"])
# we will start our dataframe with 21st drive which failed on 7th day of jan month
obs_failed_drives = pandas.DataFrame() # this is to observe behaviour of failed drives for 6 days before failure
day_before_failure = ["Day1", "Day2", "Day3", "Day4", "Day5" ,"Day6", "Day7"] # here on day7 the drive failed
for i in range(21, 35):
    dd = data_by_drive.get_group(failed_drive_list[i])[-7:]
    dd['days_before_fail'] = day_before_failure
    obs_failed_drives = obs_failed_drives.append(dd)
    #obs_failed_drives = obs_failed_drives.append(data_by_drive.get_group(failed_drive_list[i])[-7:])

sns.set(style="ticks", palette="pastel")
sns.boxplot(y="smart_190_raw", x = "days",
            hue="failure", palette=["m", "g"],
            data=obs_failed_drives)
sns.despine(offset=10, trim=True)
plt.show()

sns.set(style="ticks", palette="pastel")
sns.boxplot(y="smart_193_raw", x = "days",
            hue="failure", palette=["m", "g"],
            data=obs_failed_drives)
sns.despine(offset=10, trim=True)
plt.show()
axs: Axes



