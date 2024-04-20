import pickle, pandas
from sklearn.ensemble import HistGradientBoostingClassifier

# Load the model from the file
with open('HVAC_ident_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file) 
# Load the time series from the file
test_data=pandas.read_excel("electericty_profile_test.xlsx")
###make sure that you prepare the time series for resampling DatetimeIndex
## you need the index to be time
start_time = pandas.Timestamp(test_data.ReadStartDateTime[0])
end_time = pandas.Timestamp(test_data.ReadStartDateTime[len(test_data)-1])
# we have 15 mins of time steps
index_time = pandas.date_range(start=start_time, end=end_time, freq='15T')
## whcih features you want to use? for now months 6 and 12 are prefered
Month_start=4
Month_end=12
test_data_cleaned=pandas.DataFrame()
test_data_cleaned["VolumeAfname_TH_kWh"]=test_data.VolumeAfname_TH_kWh
test_data_cleaned.set_index(index_time, inplace=True)    
x = test_data_cleaned.resample('M').mean()/test_data_cleaned.resample('M').max()
predictions = loaded_model.predict(x[Month_start:Month_end].transpose())
print(predictions)