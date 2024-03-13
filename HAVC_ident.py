# -*- coding: utf-8 -*-
"""
Created on Wed May 31 20:00:20 2023

@author: SHARIFIM
"""
"""HVAC identifiaction from time series of elctricity consummptions using different ML techniques """

#%%
import pandas,h5py,numpy,copy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
# a class for extracting the subsets of the datsets from hdf5 files
class H5ls:
    def __init__(self):
        # Store an empty list for dataset names
        self.names = []
    def __call__(self, name, node):
        # only h5py datasets have dtype attribute, so we can search on this
        if hasattr(node, 'dtype') and name not in self.names:
            self.names += [name]
#%% time series from heat pump consumptions from the German dataset as part of exploratory data analysis EDA
# reference: https://doi.org/10.1038/s41597-022-01156-1
fileName="HP_powerTimeseries/2018_data_15min.hdf5"
file = h5py.File(fileName, 'r')
visitor = H5ls()
file.visititems(visitor)
dset_names = visitor.names
# some normal graphs for exploratory data analysis
### cases with PV panel
# Load duration curves
NoHP_input_raw=pandas.read_hdf(fileName,dset_names[78]).drop(columns="index")
plt.plot(numpy.sort(NoHP_input_raw[NoHP_input_raw["P_TOT"]>0].dropna()["P_TOT"]/1000)[::-1],linestyle ='dashed')
HP_input_raw=pandas.read_hdf(fileName,dset_names[79]).drop(columns="index")
plt.plot(numpy.sort(HP_input_raw[HP_input_raw["P_TOT"]>0].dropna()["P_TOT"]/1000)[::-1],linestyle ='dashed')
Tot_input_raw=NoHP_input_raw["P_TOT"]+HP_input_raw["P_TOT"]
plt.plot(numpy.sort(Tot_input_raw[Tot_input_raw>0].dropna()/1000)[::-1],linestyle ='dashed')
plt.ylabel("Laod [kW]")
plt.xlabel("Time step [15 mins]")
plt.ylim([0, 12])
plt.legend([ 'HP','Appliances','Total'])
plt.show()
### cases NO PV panel
# Load duration curves
HP_input_raw=pandas.read_hdf(fileName,dset_names[70]).drop(columns="index")
plt.plot(numpy.sort(HP_input_raw[HP_input_raw["P_TOT"]>0].dropna()["P_TOT"]/1000)[::-1])
NoHP_input_raw=pandas.read_hdf(fileName,dset_names[71]).drop(columns="index")
plt.plot(numpy.sort(NoHP_input_raw[NoHP_input_raw["P_TOT"]>0].dropna()["P_TOT"]/1000)[::-1])
Tot_input_raw=NoHP_input_raw["P_TOT"]+HP_input_raw["P_TOT"]
plt.plot(numpy.sort(Tot_input_raw[Tot_input_raw>0].dropna()/1000)[::-1])
plt.ylabel("Laod [kW]")
plt.xlabel("Time step [15 mins]")
plt.ylim([0, 12])
plt.legend([ 'HP','Appliances','Total'])
plt.show()
### NO PV panel
# time series comparison as part of EDA
start =25000
day_number=2
plt.plot(numpy.arange(0, 24*4*day_number,1, dtype=int),NoHP_input_raw.P_TOT.to_numpy()[start:start+24*4*day_number]/1000, alpha=0.5)
plt.plot(numpy.arange(0, 24*4*day_number,1, dtype=int),HP_input_raw.P_TOT.to_numpy()[start:start+24*4*day_number]/1000)
plt.plot(numpy.arange(0, 24*4*day_number,1, dtype=int),Tot_input_raw.to_numpy()[start:start+24*4*day_number]/1000)
plt.ylabel("Laod [kW]")
plt.xlabel("Time step [15 mins]")
plt.ylim([0, 5])
plt.legend(['Appliances', 'HP','Total'])
plt.show()
#%% 15 mins time step
## 
"""skip this part"""
# 15 mins time step
# fileName="HP_powerTimeseries/2018_data_15min.hdf5"
# file = h5py.File(fileName, 'r')
# visitor = H5ls()
# file.visititems(visitor)
# dset_names = visitor.names
# for i in range(len(dset_names)):
#     HPimput_raw_15=pandas.read_hdf(fileName,dset_names[4]).drop(columns="index")
#     if HPimput_raw_15["P_TOT"].max()<25000:
#         plt.plot(numpy.sort(HPimput_raw_15[HPimput_raw_15["P_TOT"]>0].dropna()["P_TOT"]/1000)[::-1])
# ##
# # 60 mins time step
# fileName="HP_powerTimeseries/2018_data_60min.hdf5"
# file = h5py.File(fileName, 'r')
# visitor = H5ls()
# file.visititems(visitor)
# dset_names = visitor.names
# for i in range(len(dset_names)):
#     HPimput_raw_60min=pandas.read_hdf(fileName,dset_names[i]).drop(columns="index")
#     if HPimput_raw_60min["P_TOT"].max()<25000:
#         plt.plot(numpy.sort(HPimput_raw_60min[HPimput_raw_60min["P_TOT"]>0]/1000)[::-1])
# ##
# # 10 seconds time step
# fileName="HP_powerTimeseries/2018_data_10s.hdf5"
# file = h5py.File(fileName, 'r')
# visitor = H5ls()
# file.visititems(visitor)
# dset_names = visitor.names
# for i in range(len(dset_names)):
#     HPimput_raw_10s=pandas.read_hdf(fileName,dset_names[i]).drop(columns="index")
#     if HPimput_raw_10s["P_TOT"].max()<50000:
#         plt.plot(numpy.sort(HPimput_raw_10s[HPimput_raw_10s["P_TOT"]>0]/1000)[::-1])
#%% time series classification using supervised techniques with pre-defined features
###### ###### ######  ###### ######
###### ###### ###### ###### ###### ######
###### ###### ###### ###### ###### ######
fileName="c:\\Users\\SHARIFIM\\OneDrive - VITO\\Moderate\\3_Working documents\\WP4 - Data enhancement\\HVAC_spatial_allocation\\Datasets\\HP_powerTimeseries/2018_data_15min.hdf5"
file = h5py.File(fileName, 'r')
visitor = H5ls()
file.visititems(visitor)
dset_names = visitor.names
Load_time_series=pandas.DataFrame()
for i in range(len(dset_names)):
    Load_time_series['2018_'+dset_names[i]]=pandas.read_hdf(fileName,dset_names[i]).drop(columns="index").P_TOT
fileName="c:\\Users\\SHARIFIM\\OneDrive - VITO\\Moderate\\3_Working documents\\WP4 - Data enhancement\\HVAC_spatial_allocation\\Datasets\\HP_powerTimeseries/2019_data_15min.hdf5"
file = h5py.File(fileName, 'r')
visitor = H5ls()
file.visititems(visitor)
dset_names = visitor.names
for i in range(len(dset_names)):
    Load_time_series['2019_'+dset_names[i]]=pandas.read_hdf(fileName,dset_names[i]).drop(columns="index").P_TOT    
#
# Drop the columns that are not usefull due to lack of information about them
Load_time_series = Load_time_series.drop(columns=Load_time_series.columns[Load_time_series.columns.str.contains("MISC")])
#Load_time_series = Load_time_series.drop(columns=Load_time_series.columns[Load_time_series.columns.str.contains("WITH_PV")])
#mask = Load_time_series.columns.str.contains("HOUSEHOLD")
#print(mask)
labels = [
    "WITH_HP_NO_PV" if "NO_PV" in string and "HEATPUMP" in string
    else "WITH_HP_WITH_PV" if "WITH_PV" in string and "HEATPUMP" in string
    else "NO_HP_NO_PV" if "NO_PV" in string and "HOUSEHOLD" in string
    else "NO_HP_WITH_PV" if "WITH_PV" in string and "HOUSEHOLD" in string
    else "Unknown"
    for string in Load_time_series.columns
]
#
start_time = pandas.Timestamp('2023-01-01 00:00:00')
end_time = pandas.Timestamp('2023-12-31 23:45:00')
index = pandas.date_range(start=start_time, end=end_time, freq='15T')
Load_time_series.set_index(index, inplace=True)
Load_time_series_temp=copy.copy(Load_time_series)
Load_time_series=pandas.DataFrame()
for c in range(0,len(Load_time_series_temp.columns)):
    if 'HEATPUMP' in Load_time_series_temp.columns[c]: 
        Load_time_series[Load_time_series_temp.columns[c]+"TotalWithHP"]=Load_time_series_temp[Load_time_series_temp.columns[c]]+Load_time_series_temp[Load_time_series_temp.columns[c+1]]
    else:
        Load_time_series[Load_time_series_temp.columns[c]]=Load_time_series_temp[Load_time_series_temp.columns[c]]
### we prepare the data for training and test set with lables and features
train_set=pandas.DataFrame()
train_data=numpy.array(Load_time_series)
train_labels = numpy.array([pandas.Series(labels)])
### ##
train_set['Labels']=pandas.Series(labels)
# ## ##
train_set["ind_A"]=Load_time_series.max().values/Load_time_series.mean().values
# ## ##
ratio_values = Load_time_series.resample('M').mean()/Load_time_series.resample('M').max()
ratio_values.index=['ind_M_{}'.format(i) for i in range(1, 13)]
ratio_values.columns=[i for i in range(0, len(ratio_values.columns))]
train_set=pandas.concat([train_set,ratio_values.transpose()],axis=1)
## ##
## ## ##
#sns.boxplot(
#    data=train_set, x="Labels", y="ind_A",
#   notch=True, showcaps=False,
#    flierprops={"marker": "x"},
#    boxprops={"facecolor": (.4, .6, .8, .5)},
#    medianprops={"color": "coral"})
###### ###### ######  ###### ######
###### ###### ###### ###### ###### ######
###### ###### ###### ###### ###### ######
df = copy.copy(train_set)
# Assuming you have your feature matrix in X and labels in y
# in the line below we want to peak the months that the are used for training range 11:13 means months 11 and 12 of the year. in this case 0 means aggreagetd annual consumption being used for training as well
Month_start=1
Month_end=12+1 #typical python problem with indices
X=df[['ind_M_{}'.format(i) for i in range(Month_start, Month_end)]]
#check how much of data are missing, if less 5 %, use imputation to give lalels as defined by physics of the system
missing_count_per_column = numpy.sum(numpy.isnan(X), axis=0)

###############################
# Calculate the ratio of missing data in each column
# ##
# total_rows = X.shape[0]
# missing_ratio_per_column = missing_count_per_column / total_rows
# Ratio=0
# print("Ratio of missing data in each column:")
# for i, ratio in enumerate(missing_ratio_per_column):
#    print(f"Column {i + 1}: {ratio:.2f}")
#    Ratio=+ratio
# ##############
imputer = SimpleImputer(strategy='mean')
X=imputer.fit_transform(X)

#give the lables to the dataset to prepare for training and test sets
######################
Y=df.Labels
# here we split the data to training and test sets
test_size_ratio=0.2
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size_ratio , shuffle=True,random_state=None)
train_counts = pandas.DataFrame(y_train).value_counts()
test_counts = pandas.DataFrame(y_test).value_counts()
categories = numpy.array(pandas.DataFrame(list(train_counts.index)))
# Set the width for each bar
bar_width = 0.3
# Compute the x-axis positions for the bars
x_pos = numpy.arange(len(categories))
# Plot the train bars
plt.bar(x_pos, train_counts, width=bar_width, label='Training_set')
# Adjust the x-axis positions for the test bars
x_pos = x_pos + bar_width
# Plot the test bars
plt.bar(x_pos, test_counts, width=bar_width, color='r', label='Test_set')
# Set the x-axis tick positions and labels
plt.xticks(x_pos - bar_width/2, categories[:,0])
# Set the labels and legend
plt.xlabel('Labels')
plt.ylabel('Counts')
plt.legend()
# Display the plot
plt.show()
###### ###### ######  ###### ######
###### ###### ###### ###### ###### ######
###### ###### ###### ###### ###### ######
##
accuracy_scores = []
classifiers = [
    DecisionTreeClassifier(),
    HistGradientBoostingClassifier(),
    SVC(),
    KNeighborsClassifier(),
    GaussianNB(),
    MLPClassifier(),
    RandomForestClassifier()]
# we iterate over classifiers and make models with almost all of them and compare them later
for classifier in classifiers:
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
#
Method_List = ['Decision Tree','Histogram Gradient Boosting', 'SVM', 'KNN', 'Naive Bayes', 'Neural Networks', 'Random Forest']
#
plt.bar(Method_List, accuracy_scores)
plt.xlabel('Classifier', fontsize="12")
plt.ylabel('Accuracy', fontsize="12")
plt.title('Accuracy of Different Classifiers', fontsize="12")
plt.ylim([0, 1])
plt.xticks(rotation=90)
plt.show()
# we make the confusion matrix to know where it goes wrong
matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(matrix)
disp.plot()
#######
Feature_importance = pandas.Series(classifier.feature_importances_, index=['ind_M_{}'.format(i) for i in range(Month_start, Month_end)])
fig, ax = plt.subplots()
Feature_importance.plot.bar()
ax.set_title("Feature importances based on mean decrease in impurity ()")
ax.set_ylabel("Mean decrease in impurity")
ax.set_xlabel("Feature")
fig.tight_layout()
########
# %%
# after this excercise we train the final model with only the second half and only using histogram gradient boosting. 
from sklearn.model_selection import GridSearchCV
# Assume you have your data loaded and split into X_train, X_test, y_train, y_test
Month_start=1
Month_end=12+1 #typical python problem with indices
X=df[['ind_M_{}'.format(i) for i in range(Month_start, Month_end)]]
X=imputer.fit_transform(X)
#give the lables to the dataset to prepare for training and test sets
######################
Y=df.Labels
# here we split the data to training and test sets
test_size_ratio=0.2
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size_ratio , shuffle=True,random_state=None)

# Define the HistGradientBoostingClassifier
hist_gbm = HistGradientBoostingClassifier()
# Define the parameter grid for hyperparameter tuning
param_grid = {
    'learning_rate': [0.01, 0.1, 0.5],
    'max_iter': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'min_samples_leaf': [1, 2, 4]
}
# Perform grid search cross-validation
grid_search = GridSearchCV(estimator=hist_gbm, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameters and best estimator
best_params = grid_search.best_params_
best_hist_gbm = grid_search.best_estimator_

# Evaluate the model on the test set
y_pred = best_hist_gbm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

def HP_PV_label(Elec_timeSeries):
    ## whcih features you want to use? for now months 6 and 12 are prefered
    Month_start=4
    Month_end=12
    x = test_data_cleaned.resample('M').mean()/test_data_cleaned.resample('M').max()
    predictions = grid_search.predict(x[Month_start:Month_end].transpose())
    return predictions    

#####
#find the data here
#https://opendata.fluvius.be/explore/dataset/1_50-verbruiksprofielen-dm-elek-kwartierwaarden-voor-een-volledig-jaar/information/

test_data=pandas.read_excel("electericty_profile_test.xlsx")
###make sure that you prepare the time series for resampling DatetimeIndex
## you need the index to be time
start_time = pandas.Timestamp(test_data.ReadStartDateTime[0])
end_time = pandas.Timestamp(test_data.ReadStartDateTime[len(test_data)-1])
# we have 15 mins of time steps
index_time = pandas.date_range(start=start_time, end=end_time, freq='15T')

test_data_cleaned=pandas.DataFrame()
test_data_cleaned["VolumeAfname_TH_kWh"]=test_data.VolumeAfname_TH_kWh
test_data_cleaned.set_index(index_time, inplace=True)
#you need to avoid NAN
# make sure your missing points are less than 5% of total numebr of samples
HP_PV_label(test_data_cleaned)
#### test