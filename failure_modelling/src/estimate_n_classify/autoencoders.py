from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
import numpy
import pandas as pd
import numpy as np
from pandas import Series
from pylab import rcParams
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
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
"""
def read_data(start_date,uptill_date ):
    p = Path('/Volumes/Seagate Backup Plus Drive/Uptake_DS_practicum_Backblaze/2017/data_Q1_2017')

    yr_2017_Q1_data = pandas.DataFrame()
    for child in p.iterdir():
        if (str(child) <= (str(p) + uptill_date)):
            print(child)
            yr_2017_Q1_data = pandas.concat([yr_2017_Q1_data, pandas.read_csv(str(child))], ignore_index=True)
    return yr_2017_Q1_data

start_date  = "/2017-01-01.csv"
uptill_date = "/2017-01-09.csv"  # change it to 15 days of 2018 & predict features of 2017 for supervised model 
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
import ipdb;ipdb.set_trace()
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
"""
#### write into csv using
# aa.to_csv("/Users/shikha/Documents/iit-uptake-capstone/failure_modelling/data/processed/shifted_data_next15days.csv", index=False)

df_transfd = pandas.read_csv("/Users/shikha/Documents/iit-uptake-capstone/failure_modelling/data/processed/shifted_data.csv")
#import ipdb;ipdb.set_trace()

df_transfd.set_index("serial_number", inplace=True)
 ########  73721 groups of data for 9 days ################3
features = ["failure","smart_1_raw","smart_3_raw","smart_9_raw","smart_192_raw","smart_194_raw","smart_197_raw","smart_198_raw"]
df_transfd = df_transfd[features]   # taking only relevant features
df_transfd.dropna(inplace=True)  # drop any missing values
df_transfd.reset_index(inplace=True)
################################ Divide data into train, valid, test ###################################################
#df_train, df_test = train_test_split(df_transfd, test_size=DATA_SPLIT_PERCENT, random_state=SEED)
df_train, df_valid = train_test_split(df_transfd, test_size=DATA_SPLIT_PERCENT, random_state=SEED)
df_train = df_train[features]
df_valid = df_valid[features]
df_train = df_train.dropna()
#df_test = df_test.dropna()
df_valid = df_valid.dropna()

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

# df_test_0 = df_test.loc[df_test['failure'] == 0]
# df_test_1 = df_test.loc[df_test['failure'] == 1]
# df_test_0_x = df_test_0.drop(['failure'], axis=1)
# df_test_1_x = df_test_1.drop(['failure'], axis=1)

# we only use df_train_0_x, df_valid_0_x for autoencoder as we are training only for normal behaviour of drive stats
################################ Standardize the data ##################################################################

df_train_0_x_rescaled=(df_train_0_x-numpy.mean(df_train_0_x))/(numpy.std(df_train_0_x)).values  #Standardization
df_valid_0_x_rescaled=(df_valid_0_x-numpy.mean(df_valid_0_x))/(numpy.std(df_valid_0_x)).values

################################ Defining Autoencoder ##################################################################
nb_epoch = 50  # number of epochs
batch_size = 30 # just like a document is divided in parts, training set is divided in parts
input_dim = df_train_0_x_rescaled.shape[1]  # number of predictor variables or features used
encoding_dim = 6
encoding_dim_2 = int(encoding_dim / 2)
#encoding_dim_2 = int(encoding_dim / 2)
hidden_dim =2
#hidden_dim = int(encoding_dim / 4)
learning_rate = 1e-7
input_layer = Input(shape=(input_dim, ))

def autoencoder_defined(input_layer):
    """
    A typical autoencoder architecture comprises three components : -
     -- Encoding Architecture  The encoder architecture comprises of series of layers with decreasing number of nodes
                               and ultimately reduces to a latent view repersentation.
     --Latent View Repersentation : Latent view represents the lowest level space in which the inputs are reduced
                                and information is preserved.
     --Decoding Architecture : The decoding architecture is the mirro image of the encoding architecture but in which
                                number of nodes in every layer increases and ultimately outputs the
                                similar (almost) input.

    :param input_dim:
    :return:
    """
    ## input layer
    #input_layer = Input(shape=(input_dim, ))
    #
    # next is encoding architecture
    # Dense implements operation
    # output = activation(dot(input, kernel) where activation is the element-wise activation function
    # we are using Rectified linear unit as
    # activation and  not using sigmoid as activation function as neither inputs nor outputs are in range[0,1]
    # here encoding dim is dimensionality of output space
    encoder_layer1 = Dense(encoding_dim, activation = "relu")(input_layer)
    encoder_layer2 = Dense(encoding_dim_2, activation = "relu")(encoder_layer1)
    #encoder_layer3 = Dense(encoding_dim, activation="relu")(encoder_layer2)
    #
    ## latent view
    latent_view =Dense (hidden_dim, activation="relu")(encoder_layer2)
    #
    ## decoding architecture
    decoder_layer1 = Dense(encoding_dim_2, activation="relu")(latent_view)
    decoder_layer2 = Dense(encoding_dim, activation="relu")(decoder_layer1)
    #decoder_layer3 = Dense(encoding_dim, activation="relu")(decoder_layer2)
    #
    ## output layer
    output_layer = Dense(input_dim)(decoder_layer2)    # by default the activation is linear
    #
    return output_layer


autoencoder = Model(input_layer, autoencoder_defined(input_layer))
autoencoder.summary()

# NN seek to minimize the error, hence loss function computes error/loss
################################ Training  Autoencoder #################################################################
# we will be traininng the model and saving in the file
autoencoder.compile(metrics = ["accuracy","mae"],loss='mean_squared_error', optimizer='adam') # Adam is method of Stochastic Optimization

cp = ModelCheckpoint(filepath="autoencoder_classifier.h5", save_best_only=True, verbose=0)

tb = TensorBoard(log_dir='./logs',  histogram_freq=0,  write_graph=True, write_images=True)
model = autoencoder.fit(df_train_0_x_rescaled, df_train_0_x_rescaled, epochs=nb_epoch, batch_size=batch_size,
                     validation_data=(df_valid_0_x_rescaled, df_valid_0_x_rescaled))

history = model.history
                    #verbose=1, callbacks=[cp, tb]).history # obtaining the history object after training autoencoder
#autoencoder = load_model('autoencoder_classifier.h5')


model_json = autoencoder.to_json()
f = open("/Users/shikha/Documents/iit-uptake-capstone/failure_modelling/trained_models/autoencoder.json", mode="w")
f.write(model_json)
f.close()
autoencoder.save_weights(
    "/Users/shikha/Documents/iit-uptake-capstone/failure_modelling/trained_models/autoencode_fit.h5")

################################ Plotting  Autoencoder model loss ###################################################
# Plotting model loss
loss = history['loss']
val_loss =history['val_loss']

print("The loss for trained autoencoder is as follows : ", loss)
print("The validation loss for trained autoencoder is as follows : ", val_loss)
print("The average loss for trained autoencoder is as follows : ", numpy.mean(loss))
print("The average validation loss for trained autoencoder is as follows : ", numpy.mean(val_loss))

epochs = range(1, nb_epoch + 1)
plt.figure()
axs = plt.gca()
plt.plot(epochs, loss, 'bo--', label='Training loss')
plt.plot(epochs, val_loss, 'go--', label='Validation loss')
plt.title('Training and validation loss')
axs.set_xlabel("Epochs")
axs.set_xticks(epochs)
axs.set_ylabel("Loss value")
plt.legend()
plt.show()

"""
df_transfd_to_predict = pandas.read_csv("/Users/shikha/Documents/iit-uptake-capstone/failure_modelling/data/processed/shifted_data_2017_10to19.csv")
import ipdb;ipdb.set_trace()
df_transfd_to_predict.set_index("serial_number", inplace=True)

df_transfd_to_predict = df_transfd_to_predict[features]
df_transfd_to_predict.dropna()
df_transfd_to_predict_x = df_transfd_to_predict.drop(['failure'], axis=1)
#remove extra features
x_estimates = autoencoder.predict(df_transfd_to_predict_x)

x_estimate_df = pd.DataFrame(x_estimates)
#x_estimates as numpy array

x_estimate_df.to_csv("/Users/shikha/Documents/iit-uptake-capstone/failure_modelling/data/processed/estimated_features.csv")
# 1, 3, 9, 194, 197, 198 Smart stats
# df

"""