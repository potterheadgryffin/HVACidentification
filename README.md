# HVAC_ident
This is the temporary repository for sending the HVAC identification codes to MODERATE colleagues.
The first part of the code trains the models using different methods. the data trequired for the first part are found in here https://github.com/ISFH/WPuQ/tree/master 
The second part of the code uses the method that outperform on the training set and then a function is defined named HP_PV_label
HP_PV_label funtions recives the electercity consumtpio time seroes for one year and predcits if HP or PV has been installed on the building associated to the time series.
