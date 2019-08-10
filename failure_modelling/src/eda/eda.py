import pandas
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.core.groupby import GroupBy, DataFrameGroupBy
from matplotlib.axes import Axes
from ..data.make_dataset import read_data

###################################### Reading data  for 2017 and 2015 Q3, Q4  #########################################

start_date = "/2017-01-01.csv"
uptill_date = "/2017-03-31.csv"
yr_2017_Q1_data = read_data(start_date, uptill_date)

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

##################################### Whisker plots for all columns with emptiness percent less than 75%################
# whisker plots for ranges of smart stats using 15 days ( 1 million ) rows of data
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

##################################### Violin plots for few important columns ###########################################
# using same data as for whisker plots of 15 days ( 1 million ) rows of data
sns.set(style="ticks", palette="pastel")
sns.violinplot(x="smart_9_raw", y = "failure",
            hue="failure", palette=["m", "g"],
            data=yr_2017_Q1_data)
plt.show()

sns.set(style="ticks", palette="pastel")
sns.violinplot(x="smart_187_raw", y = "failure",
            hue="failure", palette=["m", "g"],
            data=yr_2017_Q1_data)
plt.show()

sns.set(style="ticks", palette="pastel")
sns.violinplot(x="smart_194_raw", y = "failure",
            hue="failure", palette=["m", "g"],
            data=yr_2017_Q1_data)
plt.show()

sns.set(style="ticks", palette="pastel")
sns.violinplot(x="smart_197_raw", y = "failure",
            hue="failure", palette=["m", "g"],
            data=yr_2017_Q1_data)
plt.show()

sns.set(style="ticks", palette="pastel")
sns.violinplot(x="smart_198_raw", y = "failure",
            hue="failure", palette=["m", "g"],
            data=yr_2017_Q1_data)
plt.show()

sns.set(style="ticks", palette="pastel")
sns.violinplot(x="smart_241_raw", y = "failure",
            hue="failure", palette=["m", "g"],
            data=yr_2017_Q1_data)
plt.show()

sns.set(style="ticks", palette="pastel")
sns.violinplot(x="smart_190_raw", y = "failure",
            hue="failure", palette=["m", "g"],
            data=yr_2017_Q1_data)
sns.despine(offset=10, trim=True)
plt.show()

# this below dataframe of obs_failed_drives is made using data of whole Q1 data of 3 months, from which data of 14 drives
# were grouped to form it.
failed_drive_list = list(failure_instances["serial_number"])
# we will start our dataframe with 21st drive which failed on 7th day of jan month
obs_failed_drives = pandas.DataFrame() # this is to observe behaviour of failed drives for 6 days before failure
day_before_failure = ["Day1", "Day2", "Day3", "Day4", "Day5" ,"Day6", "Day7"] # here on day7 the drive failed
for i in range(21, 35):
    dd = data_by_drive.get_group(failed_drive_list[i])[-7:]
    dd['days'] = day_before_failure
    obs_failed_drives = obs_failed_drives.append(dd)
    #obs_failed_drives = obs_failed_drives.append(data_by_drive.get_group(failed_drive_list[i])[-7:])

#

# we can plot for Smart 184, 187, 188, 189 together and confirm that whether they should be considered relevant
sns.set(style="ticks", palette="pastel")
sns.boxplot(y="smart_184_raw", x = "days",
            hue="failure", palette=["m", "g"],
            data=obs_failed_drives)
sns.despine(trim=True)
plt.show()
sns.set(style="ticks", palette="pastel")
sns.boxplot(y="smart_187_raw", x = "days",
            hue="failure", palette=["m", "g"],
            data=obs_failed_drives)
sns.despine(trim=True)
plt.show()
sns.set(style="ticks", palette="pastel")
sns.boxplot(y="smart_188_raw", x = "days",
            hue="failure", palette=["m", "g"],
            data=obs_failed_drives)
sns.despine(trim=True)
plt.show()
sns.set(style="ticks", palette="pastel")
sns.boxplot(y="smart_188_raw", x = "days",
            hue="failure", palette=["m", "g"],
            data=obs_failed_drives)
sns.despine(trim=True)
plt.show()
# we can observe from the plots that we can ignore, Smart 184, 188, 189. But keep Smart 187 as relevant

sns.set(style="ticks", palette="pastel")
sns.boxplot(y="smart_190_raw", x = "days",
            hue="failure", palette=["m", "g"],
            data=obs_failed_drives)
sns.despine(offset=10, trim=True)
plt.show()
sns.set(style="ticks", palette="pastel")
sns.boxplot(y="smart_191_raw", x = "days",
            hue="failure", palette=["m", "g"],
            data=obs_failed_drives)
sns.despine(trim=True)
plt.show()

sns.set(style="ticks", palette="pastel")
sns.boxplot(y="smart_192_raw", x = "days",
            hue="failure", palette=["m", "g"],
            data=obs_failed_drives)
sns.despine()
plt.show()

sns.set(style="ticks", palette="pastel")
sns.boxplot(y="smart_193_raw", x="days",
            hue="failure", palette=["m", "g"],
            data=obs_failed_drives)
sns.despine(offset=10, trim=True)
plt.show()

obs_failed_drives["smart_194_raw"] = obs_failed_drives["smart_194_raw"].astype(float)
sns.set(style="ticks", palette="pastel")
sns.boxplot(y="smart_194_raw", x = "days",
            hue="failure", palette=["m", "g"],
            data=obs_failed_drives)
sns.despine(trim=True)
plt.show()

obs_failed_drives["smart_197_raw"] = obs_failed_drives["smart_197_raw"].astype(float)
sns.set(style="ticks", palette="pastel")
sns.boxplot(y="smart_197_raw", x = "days",
            hue="failure", palette=["m", "g"],
            data=obs_failed_drives)
sns.despine(trim=True)
plt.show()

obs_failed_drives["smart_198_raw"] = obs_failed_drives["smart_198_raw"].astype(float)
sns.set(style="ticks", palette="pastel")
sns.boxplot(y="smart_198_raw", x = "days",
            hue="failure", palette=["m", "g"],
            data=obs_failed_drives)
sns.despine(trim=True)
plt.show()

obs_failed_drives["smart_199_raw"] = obs_failed_drives["smart_199_raw"].astype(float)
sns.set(style="ticks", palette="pastel")
sns.boxplot(y="smart_199_raw", x = "days",
            hue="failure", palette=["m", "g"],
            data=obs_failed_drives)
sns.despine(trim=True)
plt.show()

# now lets plot smart stats 240, 241 and 242
sns.set(style="ticks", palette="pastel")
sns.boxplot(y="smart_240_raw", x = "days",
            hue="failure", palette=["m", "g"],
            data=obs_failed_drives)
sns.despine(trim=True)
plt.show()
sns.set(style="ticks", palette="pastel")
sns.boxplot(y="smart_241_raw", x = "days",
            hue="failure", palette=["m", "g"],
            data=obs_failed_drives)
sns.despine(trim=True)
plt.show()
sns.set(style="ticks", palette="pastel")
sns.boxplot(y="smart_242_raw", x = "days",
            hue="failure", palette=["m", "g"],
            data=obs_failed_drives)
sns.despine(trim=True)
plt.show()



# plotting various types of capacity types
yr_2017_Q1_data["capacity_bytes"].unique()
yr_2017_Q1_data["capacity_bytes"].value_counts()
labels = ["3.63TB", "7.28TB", "2.73TB", "5.46TB", "0.45TB","0.15TB","0.23TB", "0.9TB","0.29TB", "4.54TB","1.36TB","1.82TB" , "0.23TB"]
sizes = list(yr_2017_Q1_data["capacity_bytes"].value_counts())
fig, axs = plt.subplots(1,2)
wedges, _, pcts = axs[0].pie(sizes, autopct='%1.1f%%', textprops=dict(visible=False))
axs[0].axis('equal', colors=labels)
axs[0].margins(1)
axs[1].legend(wedges,
list(map(lambda x: f"{x[0]} {x[1].get_text()}", zip(labels, pcts))), title='Drive data by Capacity',
        loc="upper right")
axs[1].axis("off")
plt.show()

# plotting various kinds of models among drive data

yr_2017_Q1_data["model"].unique()
fig, axs = plt.subplots(1,2)
sizes = list(yr_2017_Q1_data["model"].value_counts())
labels = list(yr_2017_Q1_data["model"].value_counts().keys())
wedges, _, pcts = axs[0].pie(sizes, autopct='%1.1f%%', textprops=dict(visible=False))
axs[0].axis('equal', colors=labels)
axs[0].margins(1)
axs[1].legend(wedges,
list(map(lambda x: f"{x[0]} {x[1].get_text()}", zip(labels, pcts))), title='Models',
        loc="upper right")
axs[1].axis("off")
plt.show()

'''
Kindly find all the plots obtained and conclusions derived at below google doc and feel free to comment

https://docs.google.com/document/d/1Rh8s-RXt37DN16zyKCV8B-QHzYn1Sl02IYkMLdWsfBc/edit
'''




"""
The smart stats which have emptiness percentage to be less 30% are -
Smart 1 
Smart 3
Smart 4
Smart 5
Smart 7
Smart 9
Smart 10
Smart 12
Smart 192
Smart 193
Smart 194
Smart 197
Smart 198
Smart 199
These 14 are the columns which can be used for baseline model

Among the 36 - 40% missing values lie below columns which can be considered for carefull imputations : 


Smart 184
Smart 187
Smart 188
Smart 189
Smart 190
Smart 191 
Smart 240
Smart 241
Smart 242


Smart 183 is not that helpful because having 49% missing it has most values as outliers.
"""
