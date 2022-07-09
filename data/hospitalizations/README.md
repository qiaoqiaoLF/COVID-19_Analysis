# Data on COVID-19 (coronavirus) hospitalizations and intensive care by _Our World in Data_

For more general information on our COVID-19 data, see our main README file in [`/public/data`](https://github.com/owid/covid-19-data/tree/master/public/data).

The data is collected from official sources and collated by Our World in Data. We make it available as a [CSV 💾](covid-hospitalizations.csv).

The complete list of country-by-country sources is available in [`locations.csv`](https://github.com/owid/covid-19-data/blob/master/public/data/hospitalizations/locations.csv)


### Fields

| Column field | Description                                                                  |
|--------------|------------------------------------------------------------------------------|
| `entity`     | Name of the country (or region within a country)                            |
| `iso_code`   | ISO 3166-1 alpha-3 – three-letter country code |
| `date`       | Date of the observation                                                     |
| `indicator`  | Indicator name. See below our list of indicators and their definition |
| `value`      | Value of the `indicator` |

#### Indicators

| Indicator name | Description |
|----------------|-------------|
| `Daily hospital occupancy`                      | Number of COVID-19 patients in hospital on a given day |
| `Daily hospital occupancy per million`          | `Daily hospital occupancy` per million people |
| `Daily ICU occupancy`                           | Number of COVID-19 patients in ICU on a given day |
| `Daily ICU occupancy per million`               | `Daily ICU occupancy` per million people |
| `Weekly new hospital admissions`                 | Number of COVID-19 patients newly admitted to hospitals in a given week (reporting date and the preceeding 6 days) |
| `Weekly new hospital admissions per million`     | `Weekly new hospital admissions` per million people |
| `Weekly new ICU admissions`                      | Number of COVID-19 patients newly admitted to ICU in a given week (reporting date and the preceeding 6 days) |
| `Weekly new ICU admissions per million`          | `Weekly new ICU admissions` per million people |


## License

All visualizations, data, and code produced by _Our World in Data_ are completely open access under the [Creative Commons BY license](https://creativecommons.org/licenses/by/4.0/). You have the permission to use, distribute, and reproduce these in any medium, provided the source and authors are credited.
