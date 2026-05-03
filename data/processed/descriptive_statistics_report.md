# Descriptive Statistics Report

## Main empirical setup

- Main outcome: `eligibility_rate`.

- Robustness outcomes: `recipient_rate`, `acceptance_rate`.

- Main exposure: `exposure_index_pointmax`.

- Robustness exposure: `exposure_index_stormmax` and hurricane-distance indicators.


## Sample summary

| statistic | value |
| --- | --- |
| school_year_observations | 6323.0000 |
| unique_nces_schools | 311.0000 |
| first_graduation_year | 1999.0000 |
| last_graduation_year | 2025.0000 |
| total_students_processed | 655153.0000 |
| mean_students_per_school_year | 103.6143 |
| median_students_per_school_year | 75.0000 |
| mean_eligibility_rate | 0.6291 |
| mean_recipient_rate | 0.4188 |
| mean_acceptance_rate | 0.6408 |
| mean_pointmax_exposure | 0.3545 |
| mean_stormmax_exposure | 0.5815 |
| share_within_50km_hurricane | 0.0327 |
| share_within_100km_hurricane | 0.1015 |
| share_within_50km_major_hurricane | 0.0085 |


## Variable summary

| variable | count | mean | std | min | p25 | median | p75 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eligibility_rate | 6323 | 0.6291 | 0.1836 | 0.0000 | 0.5099 | 0.6439 | 0.7576 | 1.0000 |
| recipient_rate | 6323 | 0.4188 | 0.2061 | 0.0000 | 0.2610 | 0.4361 | 0.5714 | 0.9615 |
| acceptance_rate | 6318 | 0.6408 | 0.2130 | 0.0000 | 0.5385 | 0.6897 | 0.7978 | 1.0000 |
| students_processed | 6323 | 103.6143 | 83.4396 | 20.0000 | 40.0000 | 75.0000 | 142.0000 | 522.0000 |
| exposure_index_pointmax | 6323 | 0.3545 | 0.3445 | 0.0549 | 0.1211 | 0.2215 | 0.4820 | 2.7727 |
| exposure_index_stormmax | 6323 | 0.5815 | 0.6115 | 0.0584 | 0.1426 | 0.3287 | 0.8290 | 4.8845 |
| exposure_pointmax_z | 6323 | -0.0000 | 1.0001 | -0.8698 | -0.6775 | -0.3860 | 0.3704 | 7.0202 |
| min_distance_any_tropical_km | 6323 | 293.6482 | 266.2966 | 2.6387 | 75.9509 | 189.7652 | 487.6833 | 1181.0135 |
| min_distance_hurricane_km | 6323 | 657.6080 | 575.3219 | 6.3916 | 203.1317 | 493.6424 | 948.4090 | 2624.6395 |
| within_50km_hurricane | 6323 | 0.0327 | 0.1780 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| within_100km_hurricane | 6323 | 0.1015 | 0.3021 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| within_50km_major_hurricane | 6323 | 0.0085 | 0.0920 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |


## Exposure indicator summary

| indicator | count_true | count_total | share_true | students_true | students_total | student_share_true |
| --- | --- | --- | --- | --- | --- | --- |
| within_50km_any_tropical | 923 | 6323 | 0.1460 | 100671 | 655153 | 0.1537 |
| within_100km_any_tropical | 1987 | 6323 | 0.3142 | 205031 | 655153 | 0.3130 |
| within_50km_hurricane | 207 | 6323 | 0.0327 | 26602 | 655153 | 0.0406 |
| within_100km_hurricane | 642 | 6323 | 0.1015 | 76412 | 655153 | 0.1166 |
| within_50km_major_hurricane | 54 | 6323 | 0.0085 | 8098 | 655153 | 0.0124 |


## Highest exposure years by mean pointmax exposure

| graduation_year | rows | students | mean_eligibility_rate | mean_exposure_pointmax | p90_exposure_pointmax | share_within_100km_hurricane |
| --- | --- | --- | --- | --- | --- | --- |
| 2021 | 276 | 30903 | 0.5849 | 0.9934 | 1.6270 | 0.5833 |
| 2022 | 268 | 31005 | 0.5918 | 0.8050 | 1.5090 | 0.4254 |
| 2009 | 219 | 20897 | 0.6129 | 0.7113 | 1.1048 | 0.3470 |
| 2006 | 213 | 19571 | 0.6096 | 0.6834 | 1.2166 | 0.2676 |
| 2003 | 214 | 20347 | 0.5412 | 0.6324 | 1.0896 | 0.1262 |
| 2013 | 249 | 27131 | 0.5953 | 0.5948 | 0.9386 | 0.1847 |
| 2025 | 264 | 25504 | 0.6940 | 0.5277 | 0.8467 | 0.3182 |
| 2020 | 276 | 31939 | 0.6049 | 0.5010 | 0.8516 | 0.1377 |
| 2018 | 277 | 35236 | 0.5756 | 0.4323 | 0.6653 | 0.0325 |
| 1999 | 141 | 9879 | 0.8781 | 0.3572 | 0.6173 | 0.0284 |


## Outcomes by exposure quartile

| exposure_pointmax_quartile | rows | students | mean_exposure_pointmax | mean_eligibility_rate | student_weighted_eligibility_rate | mean_recipient_rate | mean_acceptance_rate | share_within_50km_hurricane | share_within_100km_hurricane |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q1 lowest | 1581 | 160210 | 0.0902 | 0.6263 | 0.6434 | 0.4374 | 0.6744 | 0.0000 | 0.0000 |
| Q2 | 1581 | 158707 | 0.1651 | 0.6516 | 0.6580 | 0.4409 | 0.6471 | 0.0000 | 0.0000 |
| Q3 | 1580 | 161234 | 0.3317 | 0.6209 | 0.6319 | 0.4105 | 0.6371 | 0.0000 | 0.0000 |
| Q4 highest | 1581 | 175002 | 0.8309 | 0.6176 | 0.6365 | 0.3863 | 0.6045 | 0.1309 | 0.4061 |


## Robustness outcomes excluding 2025

Recipient and acceptance outcomes may be incomplete for the latest cohort, so robustness checks using these outcomes should exclude 2025.


| sample | variable | count | mean | std | min | p25 | median | p75 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| excluding_2025 | recipient_rate | 6059 | 0.4356 | 0.1934 | 0.0000 | 0.2857 | 0.4483 | 0.5765 | 0.9615 |
| excluding_2025 | acceptance_rate | 6054 | 0.6668 | 0.1758 | 0.0000 | 0.5610 | 0.7000 | 0.8000 | 1.0000 |