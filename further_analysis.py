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
from sklearn.svm import SVR


def analyze_vaccination_and_hospitalization():
    def analyzer(target1,target2,knl,factor_name1,factor_name2):
        """helper function to analyze the relation between vaccination and COVID

        Args:
            target1 (str): covid data
            target2 (str): vaccination data
            knl (str): kernel function
        """        
        data_OWID = load_data.read_data("./data/owid-covid-data.csv")
        data_OWID = load_data.extract_data(data_OWID,[target1,target2,"new_cases_per_million"])
        data_target1 = np.array( data_OWID[target1] )
        data_target2 = np.array( data_OWID[target2] )
        data_total = np.array( data_OWID["new_cases_per_million"] )
        valid_data = (~np.isnan(data_target1)) * (~np.isnan(data_target2)) * (~np.isnan(data_total)) * (data_total > 1)
        target1_percentage = data_target1[valid_data] / data_total[valid_data]
        model = sklearn.svm.SVR(kernel=knl)
        model.fit(data_target2[valid_data].reshape(-1,1)[target1_percentage < 5], target1_percentage[target1_percentage < 5])
        X_fit = np.arange(100).reshape(-1, 1)
        Y_fit = model.predict(X_fit)
        plt.figure()
        plt.plot(X_fit, Y_fit)
        plt.xlabel(factor_name2)
        plt.ylabel(factor_name1)
        plt.savefig("./result/" + factor_name1  + "_relation_with" + factor_name2 + ".png")        
        
    analyzer("weekly_icu_admissions_per_million","people_vaccinated_per_hundred","rbf",'icu index','vaccination rate')
    analyzer("weekly_icu_admissions_per_million","people_fully_vaccinated_per_hundred","rbf",'icu index','fully vaccinated rate')
    analyzer("weekly_icu_admissions_per_million","total_boosters_per_hundred","rbf", 'icu index','boosters rate')
    analyzer("weekly_hosp_admissions_per_million","people_vaccinated_per_hundred","rbf",'hosp index','vaccination rate')
    analyzer("weekly_hosp_admissions_per_million","people_fully_vaccinated_per_hundred","rbf", 'hosp index','fully vaccinated rate')
    analyzer("weekly_hosp_admissions_per_million","total_boosters_per_hundred","poly",'hosp index','boosters rate')


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
    icu = np.array(data_OWID["weekly_icu_admissions_per_million"])
    hosp = np.array(data_OWID["weekly_hosp_admissions_per_million"])

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
    plt.ylabel("Average Death Rate")
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
    plt.xlabel("Smokers share")
    plt.ylabel("Average ICU admissions rate")
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
    plt.bar(["< 40", " > 40"], height=average_hosp_rate)
    plt.xlabel("Smokers share")
    plt.ylabel("Average hospital admission rate")
    plt.savefig("./result/hosp_relation_with_smokers.png")


def analyze_policy_and_COVID():
    data_OWID = load_data.read_data("./data/owid-covid-data.csv")
    data_OWID = load_data.extract_data(
        data_OWID, ["date", "new_cases_per_million", "stringency_index"]
    )

    new_cases = np.array(data_OWID["new_cases_per_million"])
    policy = np.array(data_OWID["stringency_index"])

    valid_policy_cases = (~np.isnan(new_cases)) * (~np.isnan(policy)) * (policy > 10)

    container_policy_cases = []
    for i in range(10, 100, 5):
        container_policy_cases.append(
            new_cases[valid_policy_cases][
                (
                    (policy[valid_policy_cases] >= i)
                    * (policy[valid_policy_cases] < i + 5)
                )
            ].mean()
        )
    plt.figure()
    plt.scatter(range(10, 100, 5), container_policy_cases)
    plt.ylabel("Average new cases per million")
    plt.xlabel("Stringency index")
    plt.savefig("./result/policy_relation_with_cases_raw.png")
    from sklearn.svm import SVR

    model = SVR(kernel="poly")
    model.fit(np.array(range(10, 100, 5)).reshape(-1, 1), container_policy_cases)

    X_fit = np.arange(100).reshape(-1, 1)
    Y_fit = model.predict(X_fit)
    plt.figure()
    plt.plot(X_fit, Y_fit)
    plt.ylabel("Average new cases per million")
    plt.xlabel("Stringency index")
    plt.savefig("./result/policy_relation_with_cases.png")

    # def county_policy(country):
    #     Country_data = load_data.read_data(
    #         "./data/owid-covid-data.csv", "location", country
    #     )
    #     Country_data = load_data.extract_data(
    #         Country_data, ["stringency_index", "date"]
    #     )
    #     date = np.array(Country_data["date"])
    #     policy = np.array(Country_data["stringency_index"])
    #     # valid_data = (~np.isnan(policy)) * (~np.isnan(date))
    #     plt.plot(date, policy)

    # plt.figure()
    # county_policy("Japan")
    # plt.show()
    Country_data = load_data.read_data(
        "./data/owid-covid-data.csv"
    )
    Country_data = load_data.extract_data(Country_data, ["stringency_index", "date"])
    date_ = np.array(Country_data['date'])
    policy_ = np.array(Country_data["stringency_index"])
    date_iter =     Country_data = load_data.read_data(
        "./data/owid-covid-data.csv",'location','United States'
    )
    valid_data = (~np.isnan(policy_))
    policy_ = policy_[valid_data]
    date_ = date_[valid_data]
    date_iter = np.array(load_data.extract_data(Country_data, ["date"]))
    policy_container = []
    for i in date_iter:
        policy_container.append(policy_[date_ == i].mean())
    plt.figure()
    plt.xlabel("Date")
    plt.ylabel("Stringency index")
    plt.xticks(range(0,date_iter.shape[0],200))
    # date__ = list(date_iter.squeeze())
    # plt.xticks([date__.index('2020-12-01'),date__.index('2021-02-01'),date__.index('2021-03-01'),date__.index('2021-11-01')])
    plt.plot(date_iter.squeeze(),policy_container)
    plt.savefig( "./result/policy_relation_with_date.png")
    
def analyze_economy_and_COVID():
    #human_development_index
    data_OWID = load_data.read_data("./data/owid-covid-data.csv")
    data_OWID = load_data.extract_data(
        data_OWID, ["date", "new_cases_per_million", "gdp_per_capita", 'human_development_index']
    )

    new_cases = np.array(data_OWID["new_cases_per_million"])
    economy = np.array(data_OWID["gdp_per_capita"])
    development = np.array(data_OWID['human_development_index'])

    valid_economy_cases = (~np.isnan(new_cases)) * (~np.isnan(economy))
    valid_development_cases = (~np.isnan(new_cases)) * (~np.isnan(development))
    
    container_economy_cases = []
    for i in range(0,110000,10000):
        container_economy_cases.append(
            new_cases[valid_economy_cases][
                (
                    (economy[valid_economy_cases] >= i)
                    * (economy[valid_economy_cases] < i + 10000)
                )
            ].mean()
        )  
    container_development_cases = []
    for i in np.linspace(0.4,0.9,12):
        container_development_cases.append(
            new_cases[valid_development_cases][
                (
                    (development[valid_development_cases] >= i)
                    * (development[valid_development_cases] < i + 0.05)
                )
            ].mean()
        )      
     
    plt.figure()
    plt.scatter(economy[valid_economy_cases], new_cases[valid_economy_cases])
    plt.ylabel("new cases per million")
    plt.xlabel("GDP per capita")
    plt.savefig("./result/economy_relation_with_cases_raw.png")
    plt.figure()
    plt.scatter(development[valid_development_cases], new_cases[valid_development_cases])
    plt.ylabel("new cases per million")
    plt.xlabel("Human development index")
    plt.savefig("./result/development_relation_with_cases_raw.png")
    
    plt.figure()
    plt.plot(list(range(0,110000,10000)), container_economy_cases)
    plt.ylabel("Average new cases per million")
    plt.xlabel("GDP per capita")
    plt.savefig("./result/economy_relation_with_cases.png")
    plt.figure()
    plt.plot( np.linspace(0.4,0.9,12), container_development_cases)
    plt.ylabel("Average new cases per million")
    plt.xlabel("Human development index")
    plt.savefig("./result/development_relation_with_cases.png")
     




def main():
    # analyze_vaccination_and_hospitalization()
    # analyze_smokers_and_COVID()
    # analyze_policy_and_COVID()
    analyze_economy_and_COVID()

main()
