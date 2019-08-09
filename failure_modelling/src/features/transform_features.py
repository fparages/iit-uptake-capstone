import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
from pandas import Series
from pylab import rcParams

import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

SEED = 123 #used to help randomly select the data points
DATA_SPLIT_PERCENT = 0.2

rcParams['figure.figsize'] = 8, 6
LABELS = ["Normal","Failure"]

#from ..data.make_dataset import read_data
from IPython.display import display
from pandas.core.groupby import DataFrameGroupBy
########################################## Meet the data ###############################################################
import pandas
from pathlib import Path

def read_data(start_date,uptill_date ):
    p = Path('/Volumes/Seagate Backup Plus Drive/Uptake_DS_practicum_Backblaze/2017/data_Q1_2017')

    yr_2017_Q1_data = pandas.DataFrame()
    for child in p.iterdir():
        if (str(child) <= (str(p) + uptill_date)):
            print(child)
            yr_2017_Q1_data = pandas.concat([yr_2017_Q1_data, pandas.read_csv(str(child))], ignore_index=True)
    return yr_2017_Q1_data

start_date  = "/2017-01-01.csv"
uptill_date = "/2017-01-09.csv"
yr_2017_data = read_data(start_date,uptill_date)
#data_raw = yr_2017_data.drop(columns = ["date", "serial_number","model","capacity_bytes","failure", "","", "", ""])

############################## Data Preparation  #######################################################################
# first we will group the drive data by serial number
data_by_drive: DataFrameGroupBy = yr_2017_data.groupby("serial_number")

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


obs_drives = pandas.DataFrame() # this is to observe behaviour of failed drives for 6 days before failure
for key, grp in data_by_drive:
    grp_ = curve_shift(grp, -3)
    grp_.set_index('serial_number')
    obs_drives= obs_drives.append(grp_)

import ipdb;ipdb.set_trace()
'''
Shift the data by 5 units, equal to 5 days.
Testing whether the shift happened correctly.



print('Before shifting')  # Positive labeled rows before shifting.

one_indexes = obs_drives[obs_drives['failure'] == 1].index
display(obs_drives.iloc[(one_indexes[0]-3):(one_indexes[0]+2), 0:5].head(n=5))

# Shift the response column failure by 3 rows to do a 3-day ahead prediction.
df_shiftd = curve_shift(yr_2017_data, shift_by=-3)

print('After shifting')  # Validating if the shift happened correctly.
display(df_shiftd.iloc[(one_indexes[0]-4):(one_indexes[0]+1), 0:5].head(n=5))
'''

# remove date column and categorical  or identifier ( model, capacity bytes,etc) column
df_transfd = obs_drives.drop(["date", "model", "capacity_bytes"], axis=1)


################################ Divide data into train, valid, test ###################################################
df_train, df_test = train_test_split(df_transfd, test_size=DATA_SPLIT_PERCENT, random_state=SEED)
df_train, df_valid = train_test_split(df_train, test_size=DATA_SPLIT_PERCENT, random_state=SEED)


# In autoencoder, we will be encoding only negatively labeled data( described as normal behaviour of operational drives).
# So, we will be using data with failure = 0 and build an autoencoder

# seperating  rows with failure colmn 1 and 0 for each train, test valid data
#import ipdb;ipdb.set_trace()
df_train_0 = df_train.loc[(df_train['failure'] == 0).index]
df_train_1 = df_train.loc[df_train['failure'] == 1]
df_train_0_x = df_train_0.drop(['failure'], axis=1)
df_train_1_x = df_train_1.drop(['failure'], axis=1)

df_valid_0 = df_valid.loc[df_valid['failure'] == 0]
df_valid_1 = df_valid.loc[df_valid['failure'] == 1]
df_valid_0_x = df_valid_0.drop(['failure'], axis=1)
df_valid_1_x = df_valid_1.drop(['failure'], axis=1)

df_test_0 = df_test.loc[df_test['failure'] == 0]
df_test_1 = df_test.loc[df_test['failure'] == 1]
df_test_0_x = df_test_0.drop(['failure'], axis=1)
df_test_1_x = df_test_1.drop(['failure'], axis=1)

################################ Standardize the data ##################################################################
# mostly it is better to use a standardized data (transformed to Gaussian, mean 0 and Std Deviation 1) for autoencoders
scaler = StandardScaler().fit(df_train_0_x)
df_train_0_x_rescaled = scaler.transform(df_train_0_x)
df_valid_0_x_rescaled = scaler.transform(df_valid_0_x)
df_valid_x_rescaled = scaler.transform(df_valid.drop(['failure'], axis=1))

df_test_0_x_rescaled = scaler.transform(df_test_0_x)
df_test_x_rescaled = scaler.transform(df_test.drop(['failure'], axis=1))

################################ Training  Autoencoder #################################################################
nb_epoch = 200  # number of epochs
batch_size = 128
input_dim = df_train_0_x_rescaled.shape[1]  # number of predictor variables or features used
encoding_dim = 32
hidden_dim = int(encoding_dim / 2)
learning_rate = 1e-3

input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="relu", activity_regularizer=regularizers.l1(learning_rate))(input_layer)
encoder = Dense(hidden_dim, activation="relu")(encoder)
decoder = Dense(hidden_dim, activation="relu")(encoder)
decoder = Dense(encoding_dim, activation="relu")(decoder)
decoder = Dense(input_dim, activation="linear")(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.summary()

# we will be traininng the model and saving in the file
autoencoder.compile(metrics=['accuracy'], loss='mean_squared_error', optimizer='adam')
cp = ModelCheckpoint(filepath="autoencoder_classifier.h5", save_best_only=True, verbose=0)
tb = TensorBoard(log_dir='./logs',  histogram_freq=0,  write_graph=True, write_images=True)
history = autoencoder.fit(df_train_0_x_rescaled, df_train_0_x_rescaled, epochs=nb_epoch, batch_size=batch_size,
                    shuffle=True, validation_data=(df_valid_0_x_rescaled, df_valid_0_x_rescaled),
                    verbose=1, callbacks=[cp, tb]).history
autoencoder = load_model('autoencoder_classifier.h5')

# Plotting model loss
plt.plot(history['loss'], linewidth=2, label='Train')
plt.plot(history['val_loss'], linewidth=2, label='Valid')
plt.legend(loc='upper right')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# plotting precision and recall for different threshold values
valid_x_predictions = autoencoder.predict(df_valid_x_rescaled)
mse = np.mean(np.power(df_valid_x_rescaled - valid_x_predictions, 2), axis=1)
error_df = pd.DataFrame({'Reconstruction_error': mse, 'True_class': df_valid['failure']})

precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df.True_class, error_df.Reconstruction_error)
plt.plot(threshold_rt, precision_rt[1:], label="Precision",linewidth=5)
plt.plot(threshold_rt, recall_rt[1:], label="Recall",linewidth=5)
plt.title('Precision and recall for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Precision/Recall')
plt.legend()
plt.show()

# Plotting Reconstruction error for different classes
test_x_predictions = autoencoder.predict(df_test_x_rescaled)
mse = np.mean(np.power(df_test_x_rescaled - test_x_predictions, 2), axis=1)
error_df_test = pd.DataFrame({'Reconstruction_error': mse, 'True_class': df_test['failure']})
error_df_test = error_df_test.reset_index()

threshold_fixed = 0.4
groups = error_df_test.groupby('True_class')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "Failure" if name == 1 else "Normal")
ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show()

# Plotting confusion matrix
pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]
predictions = pd.DataFrame({'true': error_df.True_class, 'predicted': pred_y})
conf_matrix = confusion_matrix(error_df.True_class, pred_y)
plt.figure(figsize=(8, 8))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

# Plotting ROC curve

false_pos_rate, true_pos_rate, thresholds = roc_curve(error_df.True_class, error_df.Reconstruction_error)
roc_auc = auc(false_pos_rate, true_pos_rate,)

plt.plot(false_pos_rate, true_pos_rate, linewidth=5, label='AUC = %0.3f'% roc_auc)
plt.plot([0,1],[0,1], linewidth=5)

plt.xlim([-0.01, 1])
plt.ylim([0, 1.01])
plt.legend(loc='lower right')
plt.title('Receiver operating characteristic curve (ROC)')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()