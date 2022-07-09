#TODO load and process the raw data
import pandas as pd
def read_country(filename,column,location):
    
    data = pd.read_csv(filename)
    continent = data.groupby(column)
    con=continent.get_group(location)
    return con
#read_country('time_series_covid19_confirmed_US.csv','Province_State','Alabama')
