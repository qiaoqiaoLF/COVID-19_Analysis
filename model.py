# TODO build a NN
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import load_data

data_raw = load_data.read_data("./data/owid-covid-data.csv")
data = load_data.extract_data(
    data_raw,
    [
        'reproduction_rate',
        'total_vaccinations',
        'people_vaccinated',
        'people_fully_vaccinated',
        'total_boosters',
        'new_vaccinations',
        'new_vaccinations_smoothed',
        'total_vaccinations_per_hundred',
        'people_vaccinated_per_hundred',
        'people_fully_vaccinated_per_hundred',
        'total_boosters_per_hundred',
        'new_vaccinations_smoothed_per_million',
        'new_people_vaccinated_smoothed',
        'new_people_vaccinated_smoothed_per_hundred',
        'stringency_index',
        'population',
        'population_density',
        'median_age',
        'aged_65_older',
        'aged_70_older',
        'gdp_per_capita',
        'extreme_poverty',
        'cardiovasc_death_rate',
        'diabetes_prevalence',
        'female_smokers',
        'male_smokers',
        'handwashing_facilities',
        'hospital_beds_per_thousand',
        'life_expectancy',
        'human_development_index',
    ],
)
print(data)
