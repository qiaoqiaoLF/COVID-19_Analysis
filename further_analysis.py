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
    def analyzer(target1,target2,knl):
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
        plt.xlabel(target1)
        plt.ylabel(target2)
        plt.savefig("./result/" + target1  + "_relation_with" + target2 + ".png")        
        
    analyzer("weekly_icu_admissions_per_million","people_vaccinated_per_hundred","rbf")
    analyzer("weekly_icu_admissions_per_million","people_fully_vaccinated_per_hundred","rbf")
    analyzer("weekly_icu_admissions_per_million","total_boosters_per_hundred","rbf")
    analyzer("weekly_hosp_admissions_per_million","people_vaccinated_per_hundred","rbf")
    analyzer("weekly_hosp_admissions_per_million","people_fully_vaccinated_per_hundred","rbf")
    analyzer("weekly_hosp_admissions_per_million","total_boosters_per_hundred","poly")
 


        
    # # read data
    # data_OWID = load_data.read_data("./data/owid-covid-data.csv")
    # data_OWID = load_data.extract_data(
    #     data_OWID,
    #     [
    #         "new_cases_per_million",
    #         "weekly_icu_admissions_per_million",
    #         "weekly_hosp_admissions_per_million",
    #         "people_vaccinated_per_hundred",
    #         "people_fully_vaccinated_per_hundred",
    #         "total_boosters_per_hundred",
    #     ],
    # )
    # icu = np.array(data_OWID["weekly_icu_admissions_per_million"])
    # hosp = np.array(data_OWID["weekly_hosp_admissions_per_million"])
    # total = np.array(data_OWID["new_cases_per_million"])
    # vac = np.array(data_OWID["people_vaccinated_per_hundred"])
    # f_vac = np.array(data_OWID["people_fully_vaccinated_per_hundred"])
    # bos = np.array(data_OWID["total_boosters_per_hundred"])       
    # # process data
    # valid_icu_vac = (
    #     (~np.isnan(icu)) * (~np.isnan(total)) * (total > 1) * (~np.isnan(vac))
    # )
    # valid_hosp_vac = (
    #     (~np.isnan(hosp)) * (~np.isnan(total)) * (total > 1) * (~np.isnan(vac))
    # )
    # valid_icu_f = (
    #     (~np.isnan(icu)) * (~np.isnan(total)) * (total > 1) * (~np.isnan(f_vac))
    # )
    # valid_hosp_f = (
    #     (~np.isnan(hosp)) * (~np.isnan(total)) * (total > 1) * (~np.isnan(f_vac))
    # )
    # valid_icu_bos = (
    #     (~np.isnan(icu)) * (~np.isnan(total)) * (total > 1) * (~np.isnan(bos))
    # )
    # valid_hosp_bos = (
    #     (~np.isnan(hosp)) * (~np.isnan(total)) * (total > 1) * (~np.isnan(bos))
    # )

    # icu_percentage = icu[valid_icu_vac] / total[valid_icu_vac]
    # hosp_percentage = hosp[valid_hosp_vac] / total[valid_hosp_vac]
    # icu_percentage_f = icu[valid_icu_f] / total[valid_icu_f]
    # hosp_percentage_f = hosp[valid_hosp_f] / total[valid_hosp_f]
    # icu_percentage_bos = icu[valid_icu_bos] / total[valid_icu_bos]
    # hosp_percentage_bos = hosp[valid_hosp_bos] / total[valid_hosp_bos]

    # # regression


    # model = SVR()
    # model.fit(
    #     f_vac[valid_hosp_f][hosp_percentage_f < 5].reshape(-1, 1),
    #     hosp_percentage_f[hosp_percentage_f < 5],
    # )
    # X_fit = np.arange(100).reshape(-1, 1)
    # Y_fit = model.predict(X_fit)
    # plt.figure()
    # plt.plot(X_fit, Y_fit)
    # plt.xlabel("People fully vaccinated per hundred")
    # plt.ylabel("Hospitalization percentage")
    # plt.savefig("./result/hosp_relation_with_fully_vaccination.png")

    # model = SVR()
    # model.fit(
    #     vac[valid_hosp_vac][hosp_percentage < 5].reshape(-1, 1),
    #     hosp_percentage[hosp_percentage < 5],
    # )
    # X_fit = np.arange(100).reshape(-1, 1)
    # Y_fit = model.predict(X_fit)
    # plt.figure()
    # plt.plot(X_fit, Y_fit)
    # plt.xlabel("People vaccinated per 100,000")
    # plt.ylabel("Hospitalization percentage")
    # plt.savefig("./result/hosp_relation_with_vaccination.png")

    # model = SVR()
    # model.fit(
    #     vac[valid_icu_vac][icu_percentage < 5].reshape(-1, 1),
    #     icu_percentage[icu_percentage < 5],
    # )
    # X_fit = np.arange(100).reshape(-1, 1)
    # Y_fit = model.predict(X_fit)
    # plt.figure()
    # plt.plot(X_fit, Y_fit)
    # plt.xlabel("People vaccinated per 100,000")
    # plt.ylabel("ICU admissions per 100,000")
    # plt.savefig("./result/icu_relation_with_vaccination.png")

    # model = SVR()
    # model.fit(
    #     f_vac[valid_icu_f][icu_percentage_f < 5].reshape(-1, 1),
    #     icu_percentage_f[icu_percentage_f < 5],
    # )
    # X_fit = np.arange(100).reshape(-1, 1)
    # Y_fit = model.predict(X_fit)
    # plt.figure()
    # plt.plot(X_fit, Y_fit)
    # plt.xlabel("Fully vaccinated percentage")
    # plt.ylabel("ICU percentage")
    # plt.savefig("./result/icu_relation_with_fully_vaccination.png")

    # model = SVR()
    # model.fit(
    #     bos[valid_icu_bos][icu_percentage_bos < 5].reshape(-1, 1),
    #     icu_percentage_bos[icu_percentage_bos < 5],
    # )
    # X_fit = np.arange(100).reshape(-1, 1)
    # Y_fit = model.predict(X_fit)
    # plt.figure()
    # plt.plot(X_fit, Y_fit)
    # plt.xlabel("People vaccinated per 100,000")
    # plt.ylabel("ICU admissions per 100,000")
    # plt.savefig("./result/icu_relation_with_boosters.png")

    # model = SVR(kernel="poly")
    # model.fit(
    #     bos[valid_hosp_bos][hosp_percentage_bos < 5].reshape(-1, 1),
    #     hosp_percentage_bos[hosp_percentage_bos < 5],
    # )
    # X_fit = np.arange(100).reshape(-1, 1)
    # Y_fit = model.predict(X_fit)
    # plt.figure()
    # plt.plot(X_fit, Y_fit)
    # plt.xlabel("Boosters per 100,000")
    # plt.ylabel("Hospitalization rate per 100,000")
    # plt.savefig("./result/hosp_relation_with_boosters.png")


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
    



def main():
    analyze_vaccination_and_hospitalization()
    analyze_smokers_and_COVID()
    analyze_policy_and_COVID()


main()
