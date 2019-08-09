from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
import numpy
import pandas
from pylab import rcParams
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from sklearn.model_selection import train_test_split, KFold
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
SEED = 123 #used to help randomly select the data points
DATA_SPLIT_PERCENT = 0.2
rcParams['figure.figsize'] = 8, 6
LABELS = ["Normal","Failure"]

########################################## Meet the data ###############################################################
df_transfd = pandas.read_csv("/Users/shikha/Documents/iit-uptake-capstone/failure_modelling/data/processed/shifted_data.csv")
df_transfd.set_index("serial_number", inplace=True)
features = ["failure","smart_1_raw","smart_3_raw","smart_9_raw","smart_192_raw","smart_194_raw","smart_197_raw","smart_198_raw"]
df_transfd = df_transfd[features]   # taking only relevant features
df_transfd.dropna(inplace=True)  # drop any missing values
df_transfd.reset_index(inplace=True)

############################################# Preprocess the data  #####################################################
df_train = df_transfd[features]
df_train = df_train.dropna()
df_train_0 = df_train.loc[(df_transfd['failure'] == 0).index]
df_train_1 = df_train.loc[df_transfd['failure'] == 1]
df_train_0_x = df_train_0.drop(['failure'], axis=1)
df_train_1_x = df_train_1.drop(['failure'], axis=1)

################################ Standardize the data ##################################################################
# mostly it is better to use a standardized data (transformed to Gaussian, mean 0 and Std Deviation 1) for autoencoders
df_train_0_x_rescaled=(df_train_0_x-numpy.mean(df_train_0_x))/(numpy.std(df_train_0_x)).values  #Standardization

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
    output_layer = Dense(input_dim)(decoder_layer2)    #by default the activation is linear
    #
    return output_layer


autoencoder = Model(input_layer, autoencoder_defined(input_layer))
autoencoder.summary()
autoencoder.compile(metrics = ["accuracy","mae"],loss='mean_squared_error', optimizer='adam') # Adam is method of Stochastic Optimization
# NN seek to minimize the error, hence loss function computes error/loss
################################ Training  Autoencoder using cross-validation ##########################################

"""
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
"""

kf = KFold(n_splits=5,  random_state=500)
#kf.get_n_splits(df_train_0_x_rescaled)
folds = kf.split(df_train_0_x_rescaled)
loss_list = []
val_loss_list = []

for j, (train_idx, val_idx) in enumerate(folds):
    print('\nFold ', j+1)
    X_train_cv = df_train_0_x_rescaled.iloc[train_idx]
    X_valid_cv = df_train_0_x_rescaled.iloc[val_idx]
    name_weights = "final_model_fold" + str(j) + "_weights.h5"
    autoencoder = Model(input_layer, autoencoder_defined(input_layer))
    autoencoder.compile(metrics=["accuracy", "mae"], loss='mean_squared_error',
                        optimizer='adam')  # Adam is method of Stochastic Optimization
    model = autoencoder.fit(
        X_train_cv,X_train_cv,
        #steps_per_epoch=len(X_train_cv) / batch_size,
        batch_size = batch_size,
        epochs=nb_epoch,
        validation_data=(X_valid_cv,X_valid_cv))
    history = model.history
    loss_list.append(history["loss"])
    val_loss_list.append(history["val_loss"])
    print(autoencoder.evaluate(X_valid_cv, X_valid_cv))

################################ Plotting  Autoencoder model loss ######################################################
# Plotting model loss

loss = loss_list
val_loss = val_loss_list

print("The loss for trained autoencoder is as follows : ", loss)
print("The validation loss for trained autoencoder is as follows : ", val_loss)
print("The average loss for trained autoencoder is as follows : ", numpy.mean(loss))
print("The average validation loss for trained autoencoder is as follows : ", numpy.mean(val_loss))


epochs = range(1, nb_epoch + 1)
plt.figure()
axs = plt.gca()
plt.plot(epochs, loss[0], 'bo--', label='Training loss')
plt.plot(epochs, val_loss[0], 'go--', label='Validation loss')
plt.title('Training and validation loss')
axs.set_xlabel("Epochs")
axs.set_xticks(epochs)
axs.set_ylabel("Loss value")
plt.legend()
plt.show()

################################ Saving the trained Autoencoder   ######################################################

model_json = autoencoder.to_json()
f = open("/Users/shikha/Documents/iit-uptake-capstone/failure_modelling/trained_models/autoencoder_after_cv.json", mode="w")
f.write(model_json)
f.close()
autoencoder.save_weights(
    "/Users/shikha/Documents/iit-uptake-capstone/failure_modelling/trained_models/autoencode_after_cv_fit.h5")



