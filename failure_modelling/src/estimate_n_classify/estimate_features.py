import pickle
import pandas
from keras.models import Model, load_model, model_from_json
from sklearn.model_selection import train_test_split

SEED = 123 #used to help randomly select the data points
DATA_SPLIT_PERCENT = 0.2

autoencoder_json_f = open("/Users/shikha/Documents/iit-uptake-capstone/failure_modelling/trained_models/autoencoder.json",
                        mode="r")
autoencoder_json = autoencoder_json_f.read()
autoencoder_json_f.close()
autoencoder = model_from_json(autoencoder_json)
autoencoder.load_weights(
    "/Users/shikha/Documents/iit-uptake-capstone/failure_modelling/trained_models/autoencode_fit.h5")
autoencoder.compile(metrics = ["accuracy","mae"],loss='mean_squared_error', optimizer='adam')
features = ["failure", "smart_1_raw", "smart_3_raw", "smart_9_raw", "smart_192_raw", "smart_194_raw", "smart_197_raw",
            "smart_198_raw"]
features_w = features.copy()
features_w.remove("failure")
df_transfd_to_predict = pandas.read_csv(
    "/Users/shikha/Documents/iit-uptake-capstone/failure_modelling/data/processed/shifted_data_2017_10to19.csv")

# df_transfd_to_predict.set_index("serial_number", inplace=True)

df_transfd_to_predict = df_transfd_to_predict[features]
df_transfd_to_predict.dropna(inplace=True)
df_transfd_to_predict.reset_index(drop=True, inplace=True)
df_transfd_to_predict_x = df_transfd_to_predict.drop(['failure'], axis=1)
# TODO: rename columns of residuals dataframe
#remove extra features
x_estimates = autoencoder.predict(df_transfd_to_predict_x)

residuals = pandas.DataFrame(df_transfd_to_predict_x.to_numpy() - x_estimates)
df_with_residuals = pandas.concat([df_transfd_to_predict, residuals], axis=1)
#x_estimates as numpy array

#x_estimate_df.to_csv("/Users/shikha/Documents/iit-uptake-capstone/failure_modelling/data/processed/estimated_features.csv")
# 1, 3, 9, 194, 197, 198 Smart stats
# df
# 736588
df_train, df_test = train_test_split(df_with_residuals, test_size=DATA_SPLIT_PERCENT, random_state=SEED)
train_indices = df_train.index
test_indices = df_test.index
df_train_x = df_train.drop("failure", axis=1)
df_test_x = df_test.drop("failure", axis=1)
y_train = df_train['failure']
y_test = df_test['failure']

import numpy
df_train_x_rescaled=(df_train_x-numpy.mean(df_train_x))/(numpy.std(df_train_x)).values
df_test_x_rescaled=(df_test_x-numpy.mean(df_test_x))/(numpy.std(df_test_x)).values

logistic_regression = pickle.load(
    open("/Users/shikha/Documents/iit-uptake-capstone/failure_modelling/trained_models/logistic_reg_i", mode="rb"))

logistic_regression.score(df_test_x, y_test)

'''
df_train, df_valid = train_test_split(df_transfd_to_predict, test_size=DATA_SPLIT_PERCENT, random_state=SEED)
train_indices = df_train.index
valid_indices = df_valid.index

aa = pipeline.predict(df_test_x_rescaled)
confusion_matrix(aa, y_test)
'''

#precision-recall curves are appropriate for imbalanced datasets, ROC are better when we have balanced datasets