# TODO analyze the relation


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


def analyze_vaccination_and_hospitalization():
    # read data
    data_OWID = load_data.read_data("./data/owid-covid-data.csv")
    data_OWID = load_data.extract_data(
        data_OWID,
        [
            "new_cases_per_million",
            "weekly_icu_admissions_per_million",
            "weekly_hosp_admissions_per_million",
            "people_vaccinated_per_hundred",
            "people_fully_vaccinated_per_hundred",
            "total_boosters_per_hundred",
        ],
    )
    icu = np.array(data_OWID["weekly_icu_admissions_per_million"])
    hosp = np.array(data_OWID["weekly_hosp_admissions_per_million"])
    total = np.array(data_OWID["new_cases_per_million"])
    vac = np.array(data_OWID["people_vaccinated_per_hundred"])
    f_vac = np.array(data_OWID["people_fully_vaccinated_per_hundred"])
    bos = np.array(data_OWID["total_boosters_per_hundred"])

    # process data
    valid_icu_vac = (
        (~np.isnan(icu)) * (~np.isnan(total)) * (total > 1) * (~np.isnan(vac))
    )
    valid_hosp_vac = (
        (~np.isnan(hosp)) * (~np.isnan(total)) * (total > 1) * (~np.isnan(vac))
    )
    valid_icu_f = (
        (~np.isnan(icu)) * (~np.isnan(total)) * (total > 1) * (~np.isnan(f_vac))
    )
    valid_hosp_f = (
        (~np.isnan(hosp)) * (~np.isnan(total)) * (total > 1) * (~np.isnan(f_vac))
    )
    valid_icu_bos = (
        (~np.isnan(icu)) * (~np.isnan(total)) * (total > 1) * (~np.isnan(bos))
    )
    valid_hosp_bos = (
        (~np.isnan(hosp)) * (~np.isnan(total)) * (total > 1) * (~np.isnan(bos))
    )

    icu_percentage = icu[valid_icu_vac] / total[valid_icu_vac]
    hosp_percentage = hosp[valid_hosp_vac] / total[valid_hosp_vac]
    icu_percentage_f = icu[valid_icu_f] / total[valid_icu_f]
    hosp_percentage_f = hosp[valid_hosp_f] / total[valid_hosp_f]
    icu_percentage_bos = icu[valid_icu_bos] / total[valid_icu_bos]
    hosp_percentage_bos = hosp[valid_hosp_bos] / total[valid_hosp_bos]

    # regression
    from sklearn.svm import SVR

    model = SVR()
    model.fit(
        f_vac[valid_hosp_f][hosp_percentage_f < 5].reshape(-1, 1),
        hosp_percentage_f[hosp_percentage_f < 5],
    )
    X_fit = np.arange(100).reshape(-1, 1)
    Y_fit = model.predict(X_fit)
    plt.figure()
    plt.plot(X_fit, Y_fit)
    plt.savefig("./result/hosp_relation_with_fully_vaccination.png")

    model = SVR()
    model.fit(
        vac[valid_hosp_vac][hosp_percentage < 5].reshape(-1, 1),
        hosp_percentage[hosp_percentage < 5],
    )
    X_fit = np.arange(100).reshape(-1, 1)
    Y_fit = model.predict(X_fit)
    plt.figure()
    plt.plot(X_fit, Y_fit)
    plt.savefig("./result/hosp_relation_with_vaccination.png")

    model = SVR()
    model.fit(
        vac[valid_icu_vac][icu_percentage < 5].reshape(-1, 1),
        icu_percentage[icu_percentage < 5],
    )
    X_fit = np.arange(100).reshape(-1, 1)
    Y_fit = model.predict(X_fit)
    plt.figure()
    plt.plot(X_fit, Y_fit)
    plt.savefig("./result/icu_relation_with_vaccination.png")

    model = SVR()
    model.fit(
        f_vac[valid_icu_f][icu_percentage_f < 5].reshape(-1, 1),
        icu_percentage_f[icu_percentage_f < 5],
    )
    X_fit = np.arange(100).reshape(-1, 1)
    Y_fit = model.predict(X_fit)
    plt.figure()
    plt.plot(X_fit, Y_fit)
    plt.savefig("./result/icu_relation_with_fully_vaccination.png")

    model = SVR()
    model.fit(
        bos[valid_icu_bos][icu_percentage_bos < 5].reshape(-1, 1),
        icu_percentage_bos[icu_percentage_bos < 5],
    )
    X_fit = np.arange(100).reshape(-1, 1)
    Y_fit = model.predict(X_fit)
    plt.figure()
    plt.plot(X_fit, Y_fit)
    plt.savefig("./result/icu_relation_with_boosters.png")

    model = SVR()
    model.fit(
        bos[valid_hosp_bos][hosp_percentage_bos < 5].reshape(-1, 1),
        hosp_percentage_bos[hosp_percentage_bos < 5],
    )
    X_fit = np.arange(100).reshape(-1, 1)
    Y_fit = model.predict(X_fit)
    plt.figure()
    plt.plot(X_fit, Y_fit)
    plt.savefig("./result/hosp_relation_with_boosters.png")


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
    analyze_vaccination_and_hospitalization()
    analyze_smokers_and_COVID()


main()
