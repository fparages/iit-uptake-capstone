import pandas
from keras.models import Model, load_model, model_from_json
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, precision_recall_curve
import numpy

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score

autoencoder_json_f = open("/Users/shikha/Documents/iit-uptake-capstone/failure_modelling/trained_models/autoencoder_after_cv.json",
                        mode="r")
autoencoder_json = autoencoder_json_f.read()
autoencoder_json_f.close()
autoencoder = model_from_json(autoencoder_json)
autoencoder.load_weights(
    "/Users/shikha/Documents/iit-uptake-capstone/failure_modelling/trained_models/autoencode_after_cv_fit.h5")
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
labels = df_transfd_to_predict["failure"]
df_transfd_to_predict_x = df_transfd_to_predict.drop(['failure'], axis=1)
df_transfd_x_rescaled=(df_transfd_to_predict_x-df_transfd_to_predict_x.mean())/df_transfd_to_predict_x.std()

# TODO: rename columns of residuals dataframe
#remove extra features
# x_estimates = autoencoder.predict(df_transfd_x_rescaled)
# x_est_df = pandas.DataFrame(x_estimates)

smote = SMOTE(sampling_strategy='minority', n_jobs=4,random_state=20)
xsamp, ysamp = smote.fit_resample(df_transfd_x_rescaled, labels)
ysamp = pandas.Series(ysamp)
xsamp = pandas.DataFrame(xsamp)
xsamp.columns = df_transfd_x_rescaled.columns
x_estimates = autoencoder.predict(xsamp)
x_est_df = pandas.DataFrame(x_estimates)

################################## Ploting actual and estimated features as scatter plots ##############################


plt.figure()
axs: Axes = plt.gca()
axs.set_xlabel("Actual Smart 1 Values")
axs.set_ylabel("Estimated Smart 1 Values")
axs.scatter(xsamp["smart_1_raw"][ysamp == 1], x_est_df[1][ysamp == 1], label="Failure", alpha=0.025)
axs.scatter(xsamp["smart_1_raw"][ysamp == 0], x_est_df[1][ysamp == 0], label="Non-Failure", alpha=0.025)
axs.legend()
plt.show()

plt.figure()
axs: Axes = plt.gca()
axs.set_xlabel("Actual Smart 3 Values")
axs.set_ylabel("Estimated Smart 3 Values")
axs.scatter(xsamp["smart_3_raw"][ysamp == 1], x_est_df[1][ysamp == 1], label="Failure", alpha=0.025)
axs.scatter(xsamp["smart_3_raw"][ysamp == 0], x_est_df[1][ysamp == 0], label="Non-Failure", alpha=0.025)
axs.legend()
plt.show()

plt.figure()
axs: Axes = plt.gca()
axs.set_xlabel("Actual Smart 9 Values")
axs.set_ylabel("Estimated Smart 9 Values")
axs.scatter(xsamp["smart_9_raw"][ysamp == 1], x_est_df[1][ysamp == 1], label="Failure", alpha=0.025)
axs.scatter(xsamp["smart_9_raw"][ysamp == 0], x_est_df[1][ysamp == 0], label="Non-Failure", alpha=0.025)
axs.legend()
plt.show()

plt.figure()
axs: Axes = plt.gca()
axs.set_xlabel("Actual Smart 192 Values")
axs.set_ylabel("Estimated Smart 192 Values")
axs.scatter(xsamp["smart_192_raw"][ysamp == 1], x_est_df[1][ysamp == 1], label="Failure", alpha=0.025)
axs.scatter(xsamp["smart_192_raw"][ysamp == 0], x_est_df[1][ysamp == 0], label="Non-Failure", alpha=0.025)
axs.legend()
plt.show()

plt.figure()
axs: Axes = plt.gca()
axs.set_xlabel("Actual Smart 194 Values")
axs.set_ylabel("Estimated Smart 194 Values")
axs.scatter(xsamp["smart_194_raw"][ysamp == 1], x_est_df[1][ysamp == 1], label="Failure", alpha=0.025)
axs.scatter(xsamp["smart_194_raw"][ysamp == 0], x_est_df[1][ysamp == 0], label="Non-Failure", alpha=0.025)
axs.legend()
plt.show()

plt.figure()
axs: Axes = plt.gca()
axs.set_xlabel("Actual Smart 197 Values")
axs.set_ylabel("Estimated Smart 197 Values")
axs.scatter(xsamp["smart_197_raw"][ysamp == 1], x_est_df[1][ysamp == 1], label="Failure", alpha=0.025)
axs.scatter(xsamp["smart_197_raw"][ysamp == 0], x_est_df[1][ysamp == 0], label="Non-Failure", alpha=0.025)
axs.legend()
plt.show()

plt.figure()
axs: Axes = plt.gca()
axs.set_xlabel("Actual Smart 198 Values")
axs.set_ylabel("Estimated Smart 198 Values")
axs.scatter(xsamp["smart_198_raw"][ysamp == 1], x_est_df[1][ysamp == 1], label="Failure", alpha=0.025)
axs.scatter(xsamp["smart_198_raw"][ysamp == 0], x_est_df[1][ysamp == 0], label="Non-Failure", alpha=0.025)
axs.legend()
plt.show()

################################## Ploting cosine similarity of features as vectors, using density plot ################
cos_sim = []
for ind in range(xsamp.shape[0]):
    print(ind)
    res = cosine_similarity(xsamp.iloc[ind, :].values.reshape(1, -1),
                            x_est_df.iloc[ind, :].values.reshape(1, -1)).flatten().tolist()
    cos_sim.extend(res)

cs_df = pandas.DataFrame(cos_sim)
cos_sim_f = cs_df[ysamp == 1]
cos_sim_nf = cs_df[ysamp == 0]
sns.kdeplot(cos_sim_nf[0], shade = True, color="b", label = "Non-Failure")
sns.kdeplot(cos_sim_f[0], shade = True, color="r", label = "Failure")
plt.xlabel("Cosine Similarity metric of actual and estimated feature vectors")
plt.ylabel("Density")
plt.legend()
plt.show()

# df_transfd_x_rescaled has 735669 data points
################################## Ploting residuals of features as vectors, using density plot ########################
residuals = pandas.DataFrame(xsamp.to_numpy() - x_estimates)
df_with_residuals = pandas.concat([xsamp, residuals], axis=1)

res_smart_1_f = residuals[0][ysamp == 1]
res_smart_1_nf =  residuals[0][ysamp == 0]
sns.kdeplot(res_smart_1_nf, shade = True, color="b", label = "Non-Failure")
sns.kdeplot(res_smart_1_f, shade = True, color="r", label = "Failure")
plt.xlabel("Smart 1 residuals")
plt.ylabel("Density")
plt.legend()
plt.show()

res_smart_3_f = residuals[1][ysamp == 1]
res_smart_3_nf =  residuals[1][ysamp == 0]
sns.kdeplot(res_smart_3_nf, shade = True, color="b", label = "Non-Failure")
sns.kdeplot(res_smart_3_f, shade = True, color="r", label = "Failure")
plt.xlabel("Smart 3 residuals")
plt.ylabel("Density")
plt.legend()
plt.show()

res_smart_9_f = residuals[2][ysamp == 1]
res_smart_9_nf = residuals[2][ysamp == 0]
sns.kdeplot(res_smart_9_nf, shade = True, color="b", label = "Non-Failure")
sns.kdeplot(res_smart_9_f, shade = True, color="r", label = "Failure")
plt.xlabel("Smart 9 residuals")
plt.ylabel("Density")
plt.legend()
plt.show()

res_smart_192_f = residuals[3][ysamp == 1]
res_smart_192_nf = residuals[3][ysamp == 0]
sns.kdeplot(res_smart_192_nf, shade = True, color="b", label = "Non-Failure")
sns.kdeplot(res_smart_192_f, shade = True, color="r", label = "Failure")
plt.xlabel("Smart 192 residuals")
plt.ylabel("Density")
plt.legend()
plt.show()

res_smart_194_f = residuals[4][ysamp == 1]
res_smart_194_nf = residuals[4][ysamp == 0]
sns.kdeplot(res_smart_194_nf, shade = True, color="b", label = "Non-Failure")
sns.kdeplot(res_smart_194_f, shade = True, color="r", label = "Failure")
plt.xlabel("Smart 194 residuals")
plt.ylabel("Density")
plt.legend()
plt.show()
res_smart_197_f = residuals[5][ysamp == 1]
res_smart_197_nf =  residuals[5][ysamp == 0]
sns.kdeplot(res_smart_197_nf, shade = True, color="b", label = "Non-Failure")
sns.kdeplot(res_smart_197_f, shade = True, color="r", label = "Failure")
plt.xlabel("Smart 197 residuals")
plt.ylabel("Density")
plt.legend()
plt.show()
res_smart_198_f = residuals[6][ysamp == 1]
res_smart_198_nf =  residuals[6][ysamp == 0]
sns.kdeplot(res_smart_198_nf, shade = True, color="b", label = "Non-Failure")
sns.kdeplot(res_smart_198_f, shade = True, color="r", label = "Failure")
plt.xlabel("Smart 198 residuals")
plt.ylabel("Density")
plt.legend()
plt.show()

################################## Random Forest without feeding  residuals ############################################
X_train, X_test, y_train, y_test = train_test_split(df_with_residuals, ysamp, test_size=0.2, random_state=20)
y_train = y_train.astype('int64')
y_train = y_train.astype('category')
y_test = y_test.astype('int64')
y_test = y_test.astype('category')
random_forest_model = RandomForestClassifier(random_state= 20, max_depth=7) # instantiate model
random_forest_model.fit(X_train.iloc[:, :7], y_train) # fitting the model on training data
predictions = random_forest_model.predict(X_test.iloc[:, :7])
print("Accuracy:",random_forest_model.score(X_test.iloc[:, :7], y_test))
print("Accuracy:",random_forest_model.score(X_train.iloc[:, :7], y_train))
# Visualise classical Confusion Matrix
CM = confusion_matrix(y_test, predictions)
print(CM)
fpr, tpr, th = roc_curve(y_test, random_forest_model.predict_proba(X_test.iloc[:, :7])[:, 1])
plt.figure()
axs = plt.gca()
axs.plot(fpr, tpr, 'b--', label = "Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(" ROC curve for Random Forest without residuals")
plt.plot([0.0,1.0],[0.0,1.0], 'r--', label = "No skill curve")
plt.legend()
plt.show()

################################## Random Forest after feeding  residuals ##############################################
random_forest_model = RandomForestClassifier(random_state= 20, max_depth=7) # instantiate model
random_forest_model.fit(X_train, y_train) # fitting the model on training data

predictions = random_forest_model.predict(X_test)
print("Accuracy:",random_forest_model.score(X_test, y_test))
print("Accuracy:",random_forest_model.score(X_train, y_train))
# Visualise classical Confusion Matrix
CM = confusion_matrix(y_test, predictions)
print(CM)

# prec, rec, th = precision_recall_curve(y_test, random_forest_model.predict_proba(X_test)[:, 1])
# plt.figure()
# axs = plt.gca()
# axs.plot(rec, prec, 'bo--')
# plt.show()

fpr, tpr, th = roc_curve(y_test, random_forest_model.predict_proba(X_test)[:, 1])
plt.figure()
axs = plt.gca()
axs.plot(fpr, tpr, 'b--', label = "Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(" ROC curve for Random Forest after feeding residuals")
plt.plot([0.0,1.0],[0.0,1.0], 'r--', label = "No skill curve")
plt.legend()
plt.show()
# Conclusion : Increase in recall with little decrease i precision. Moreover,  missed alarms have decreased which is
# very important.

fimp = pandas.DataFrame()
cols = X_train.columns[:7].tolist()
cols = cols + [f"residual_{i}" for i in range(7)]
fimp['col'] = cols
fimp['imp'] = random_forest_model.feature_importances_
fimp.sort_values('imp', ascending=False, inplace=True)

feature_mappings = ['smart_1', 'smart_3', 'smart_9', 'smart_192',
                    'smart_194', 'smart_197', 'smart_198', 'residual_1', 'residual_3', 'residual_9', 'residual_192',
                    'residual_194', 'residual_197', 'residual_198']
feature_mappings = pandas.Series(feature_mappings)
fimp['ncol'] = feature_mappings[fimp.index]

axs = sns.barplot('imp', 'ncol', data=fimp)
axs.set_xlabel("Feature Importance")
axs.set_ylabel("Columns")
axs.set_title("Random forest with residuals")
plt.show()

# Kindly find all the relevant and final plots with respect to autoencoders - https://docs.google.com/document/d/1aHOepvTe56HxDjetrt9pPZ6EwtKqed8LwTb_DjLGrD0/edit#
