# Baseline TWFE Regression Results

Main outcome: `eligibility_rate`. Robustness outcomes: `recipient_rate` and `acceptance_rate`.

Standard errors are clustered by NCES school. Weighted models use `students_processed`.

| model_name                       | outcome          | exposure                | estimate   | se       |   p_value |   n_obs |   n_schools | fixed_effects   | weighted   | exclude_2025   |   r_squared |
|:---------------------------------|:-----------------|:------------------------|:-----------|:---------|----------:|--------:|------------:|:----------------|:-----------|:---------------|------------:|
| M1_year_fe                       | eligibility_rate | exposure_index_pointmax | 0.0271*    | (0.0161) |     0.093 |    6323 |         311 | year            | False      | False          |       0.188 |
| M2_twfe_unweighted               | eligibility_rate | exposure_index_pointmax | 0.0017     | (0.0058) |     0.766 |    6323 |         311 | school_year     | False      | False          |       0.764 |
| M3_twfe_weighted_main            | eligibility_rate | exposure_index_pointmax | 0.0058     | (0.0049) |     0.241 |    6323 |         311 | school_year     | True       | False          |       0.802 |
| M4_twfe_weighted_stormmax        | eligibility_rate | exposure_index_stormmax | 0.0002     | (0.0031) |     0.956 |    6323 |         311 | school_year     | True       | False          |       0.802 |
| M5_twfe_weighted_100km_hurricane | eligibility_rate | within_100km_hurricane  | 0.0012     | (0.0038) |     0.761 |    6323 |         311 | school_year     | True       | False          |       0.802 |
| M6_twfe_weighted_50km_hurricane  | eligibility_rate | within_50km_hurricane   | 0.0062     | (0.0062) |     0.317 |    6323 |         311 | school_year     | True       | False          |       0.802 |
| M7_twfe_weighted_pointmax_z      | eligibility_rate | exposure_pointmax_z     | 0.0020     | (0.0017) |     0.241 |    6323 |         311 | school_year     | True       | False          |       0.802 |
| R1_recipient_rate_no2025         | recipient_rate   | exposure_index_pointmax | 0.0037     | (0.0046) |     0.419 |    6059 |         309 | school_year     | True       | True           |       0.864 |
| R2_acceptance_rate_no2025        | acceptance_rate  | exposure_index_pointmax | -0.0007    | (0.0046) |     0.887 |    6054 |         307 | school_year     | True       | True           |       0.771 |

Significance: * p < 0.10, ** p < 0.05, *** p < 0.01.
