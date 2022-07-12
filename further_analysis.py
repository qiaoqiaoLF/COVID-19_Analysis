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
import argparse

def analyze_plot(column_1,column_2,kerne):

    #read data
    data_OWID = load_data.read_data("./data/owid-covid-data.csv")
    data_OWID = load_data.extract_data(data_OWID,[column_1,column_2,'new_cases_per_million'],)
    a = np.array(data_OWID[column_1])
    b = np.array(data_OWID[column_2])
    total = np.array(data_OWID['new_cases_per_million'])

    #process data
    valid_a_b = (~np.isnan(a)) * (~np.isnan(total)) * (total > 1) * (~np.isnan(b))

    a_percentage_b = a[valid_a_b] / total[valid_a_b]

    from sklearn.svm import SVR

    model = SVR(kernel = kerne)
    model.fit(
        b[valid_a_b][a_percentage_b < 5].reshape(-1,1),
        a_percentage_b[a_percentage_b < 5],
        )
    X_fit = np.arange(100).reshape(-1,1)
    Y_fit = model.predict(X_fit)
    plt.figure(figsize=(8,8))
    plt.plot(X_fit, Y_fit)
    Title = 'relation \n between ' + column_1 + '\n' + 'and ' + column_2
    plt.title(Title)
    Result_name = 'relation between ' + column_1 + ' and ' + column_2
    plt.savefig("./result/{0}.png".format(Result_name))



def analyze_smokers_and_COVID():
    data_OWID = load_data.read_data("./data/owid-covid-data.csv")
    data_OWID = load_data.extract_data(
        data_OWID,
        [
            "new_deaths_per_million",
            "new_cases_per_million",
            "male_smokers",
            "weekly_icu_admissions_per_million",
            "weekly_hosp_admissions_per_million",
        ],
    )

    deaths = np.array(data_OWID["new_deaths_per_million"])
    smokers = np.array(data_OWID["male_smokers"])
    new_cases = np.array(data_OWID["new_cases_per_million"])
    icu = np.array( data_OWID["weekly_icu_admissions_per_million"])
    hosp =  np.array( data_OWID["weekly_hosp_admissions_per_million"])

    valid_data_deaths = (
        (~np.isnan(deaths))
        * (~np.isnan(smokers))
        * (new_cases > 1)
        * (smokers < 80)
        * (smokers > 10)
    )
    
    valid_data_icu = (
        (~np.isnan(icu))
        * (~np.isnan(smokers))
        * (new_cases > 1)    
        * (smokers < 80)
        * (smokers > 10)
    )
    valid_data_hosp = (
        (~np.isnan(hosp))
        * (~np.isnan(smokers))
        * (new_cases > 1)
        * (smokers < 80)
        * (smokers > 10)
    )
    
    deaths_percentage = deaths[valid_data_deaths] / new_cases[valid_data_deaths]
    smokers_deaths = smokers[valid_data_deaths]
    icu_percentage = icu[valid_data_icu] / new_cases[valid_data_icu]
    hosp_percentage = hosp[valid_data_hosp] / new_cases[valid_data_hosp]
    smokers_icu = smokers[valid_data_icu]
    smokers_hosp = smokers[valid_data_hosp]

    average_death_rate = []
    average_death_rate.append(deaths_percentage[smokers_deaths < 20].mean())
    average_death_rate.append(
        deaths_percentage[((smokers_deaths >= 20) * (smokers_deaths < 40))].mean()
    )
    average_death_rate.append(
        deaths_percentage[((smokers_deaths >= 40) * (smokers_deaths < 60))].mean()
    )
    average_death_rate.append(
        deaths_percentage[((smokers_deaths >= 60) * (smokers_deaths < 80))].mean()
    )

    plt.figure()
    plt.bar(["0-20", "20-40", "40-60", "60-80"], height=average_death_rate)
    plt.savefig("./result/death_relation_with_smokers.png")

    average_icu_rate = []
    # average_icu_rate.append(icu_percentage[smokers_icu < 20].mean())
    average_icu_rate.append(
        icu_percentage[((smokers_icu >= 20) * (smokers_icu < 40))].mean()
    )
    average_icu_rate.append(
        icu_percentage[((smokers_icu >= 40) * (smokers_icu < 60))].mean()
    )
    # average_icu_rate.append(
    #     icu_percentage[((smokers_icu >= 60) * (smokers_icu < 70))].mean()
    # )
     
    plt.figure()
    plt.bar(["< 40", " > 40"], height=average_icu_rate)
    plt.savefig("./result/icu_relation_with_smokers.png")
     
    average_hosp_rate = []
    # average_hosp_rate.append(hosp_percentage[smokers_hosp < 20].mean())
    average_hosp_rate.append(
        hosp_percentage[((smokers_hosp >= 20) * (smokers_hosp < 40))].mean()
    )
    average_hosp_rate.append(
        hosp_percentage[((smokers_hosp >= 40) * (smokers_hosp < 60))].mean()
    )
    # average_hosp_rate.append(
    #     hosp_percentage[((smokers_hosp >= 60) * (smokers_hosp < 80))].mean()
    # )
     
    plt.figure()
    plt.bar([ "< 40", " > 40"], height=average_hosp_rate)
    plt.savefig("./result/hosp_relation_with_smokers.png")

def main():
    parser = argparse.ArgumentParser(description='Funtion : Generate Data')
    parser.add_argument("column1", type=str, help='properties of column1')
    parser.add_argument("column2", type=str, help='properties of column2')
    parser.add_argument("kernel", type=str, help='kernel')
    args = parser.parse_args()
    analyze_plot(args.column1, args.column2, args.kernel)
    
    analyze_smokers_and_COVID()


main()
