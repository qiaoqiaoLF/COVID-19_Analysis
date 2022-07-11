#TODO visualize the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
import sklearn
import plotly
from folium import plugins
import plotly.express as px
import scipy
import load_data
def plot(x,y):
    plt.figure()
    plt.plot(x,y)
    for a,b in zip(x,y):
        plt.text(a,b,(a,b),ha='center',va='bottom',fontsize=15)
    plt.show()
#x = [1,2,3,4,5]
#y = [2,5,7,3,4]
#plot(x,y)

def draw_new_cases():
    confirmed_global_OWID = load_data.read_data('./data/owid-covid-data.csv')
    confirmed_global_OWID = load_data.extract_data(confirmed_global_OWID,['location','date','new_cases_smoothed'])
    fig_global = px.line(confirmed_global_OWID,line_group= 'location' , x= 'date', y='new_cases_smoothed',color_discrete_sequence= px.colors.qualitative.D3,color="location", title="New cases in the world")
    plotly.offline.plot(fig_global, filename='./result/plotly_global.html')
    
def draw_map_of_total_cases_global(map):
    confirmed_global_JHU = load_data.read_data('./data/time_series_covid19_confirmed_global.csv')
    confirmed_global_JHU = load_data.extract_data(confirmed_global_JHU,['Country/Region','Province/State','Long','Lat','7/5/22'])
    confirmed_global_JHU = confirmed_global_JHU.drop(index=[42,43,53,89,106,175,281])
    incidents = plugins.MarkerCluster().add_to(map)
    for country ,province,lat, long, number in zip(confirmed_global_JHU['Country/Region'],confirmed_global_JHU['Province/State'],confirmed_global_JHU['Lat'],confirmed_global_JHU['Long'],confirmed_global_JHU['7/5/22']):
        folium.Marker([lat, long], popup=(str(country) +'-'+ (str(province)) + '-' +str(number))).add_to(incidents)
    map.add_child(incidents)
    
def draw_map_of_total_cases_US(map):    
    confirmed_US_JHU = load_data.read_data('./data/time_series_covid19_confirmed_US.csv')
    confirmed_US_JHU = load_data.extract_data(confirmed_US_JHU,['Province_State','Admin2','Lat','Long_','7/5/22'])
    incidents = plugins.MarkerCluster().add_to(map)
    for province,city,lat, long, number in zip(confirmed_US_JHU['Province_State'],confirmed_US_JHU['Admin2'],confirmed_US_JHU['Lat'],confirmed_US_JHU['Long_'],confirmed_US_JHU['7/5/22']):
        folium.Marker([lat, long], popup=('US' +'-'+ (str(province)) +'-'+ str(city) +  '-' +str(number))).add_to(incidents)
    map.add_child(incidents)
    
  
    
def main():
    draw_new_cases()
    map =  folium.Map( location=[0,0], zoom_start=2)
    draw_map_of_total_cases_global(map)
    draw_map_of_total_cases_US(map)
    map.save('./result/map.html')

main()