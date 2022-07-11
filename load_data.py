#TODO load and process the raw data
import pandas as pd
def read_data(filename,column = 'all',location = 'all'):
    """read the data from the csv file and return a dataframe

    Args:
        filename (str):  the name of the csv file
        column (str): colomn name to be used as index
        location (str): data to be extracted

    Returns:
        dataframe: a dataframe with the data
    e.g. 
    read_data('time_series_covid19_confirmed_US.csv','Province_State','Alabama')
    """    
    data_raw = pd.read_csv(filename)
    if location == 'all':
        return data_raw
    continent = data_raw.groupby(column)
    data = continent.get_group(location)
    return data 

def extract_data(data_raw,column):
    """ extract the data from the dataframe and return a dataframe

    Args:
        data_raw (dataframe):   the dataframe with the data
        column (list):  list of column names to be used as index
        
    Returns:
        dataframe: a dataframe with the wanted data
    e.g.
    extract_data(data_raw,['Province_State','Country_Region','Lat','Long'])
    
    """     
    data = data_raw.loc[:,column]
    return data