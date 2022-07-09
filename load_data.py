#TODO load and process the raw data
import pandas as pd
def read_country(a):
    
    data = pd.read_csv('owid-covid-data.csv')
    continent = data.groupby('location')
    con=continent.get_group(a)
    return con
