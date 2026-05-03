# Hurricane Exposure Diagnostics

## Basic coverage

- Analysis panel rows: 12,297

- Rows with exposure: 7,949

- Rows without exposure: 4,348

- Correlation between pointmax and stormmax exposure: 0.859


## Highest mean exposure years

|   graduation_year |   rows |   mean_pointmax |   p90_pointmax |   max_pointmax |   mean_stormmax |   max_stormmax |
|------------------:|-------:|----------------:|---------------:|---------------:|----------------:|---------------:|
|              2021 |    322 |        1.00838  |       1.6557   |        3.18628 |        1.50819  |        4.49828 |
|              2022 |    324 |        0.786076 |       1.48843  |        2.77269 |        1.26315  |        3.49213 |
|              2009 |    282 |        0.687025 |       1.07311  |        2.80111 |        1.55379  |        4.88445 |
|              2006 |    278 |        0.684911 |       1.23242  |        3.11968 |        1.31469  |        4.8355  |
|              2003 |    275 |        0.615646 |       1.07538  |        2.06448 |        1.34199  |        4.301   |
|              2013 |    290 |        0.589105 |       0.936194 |        2.10765 |        0.821656 |        2.45893 |
|              2025 |    328 |        0.50224  |       0.769675 |        1.77096 |        0.865963 |        2.73701 |
|              2020 |    316 |        0.494593 |       0.82841  |        1.45411 |        0.671156 |        2.18374 |


## Threshold indicator summary

| indicator                   |   count_true |   count_total |   share_true |   students_true |   students_total |   student_share_true |
|:----------------------------|-------------:|--------------:|-------------:|----------------:|-----------------:|---------------------:|
| within_50km_any_tropical    |         1108 |          7949 |   0.139389   |          102550 |           671601 |            0.152695  |
| within_100km_any_tropical   |         2428 |          7949 |   0.305447   |          209589 |           671601 |            0.312074  |
| within_50km_hurricane       |          244 |          7949 |   0.0306957  |           26926 |           671601 |            0.0400923 |
| within_100km_hurricane      |          751 |          7949 |   0.0944773  |           77485 |           671601 |            0.115374  |
| within_50km_major_hurricane |           67 |          7949 |   0.00842873 |            8192 |           671601 |            0.0121977 |


## Top storms by max point-level exposure

|   storm_year | max_exposure_storm_name   |   exposed_school_years |   exposed_students |   mean_pointmax |   max_pointmax |   schools_within_50km_hurricane |   schools_within_100km_hurricane |
|-------------:|:--------------------------|-----------------------:|-------------------:|----------------:|---------------:|--------------------------------:|---------------------------------:|
|         2020 | DELTA                     |                    133 |              12048 |        0.870452 |        3.18628 |                              20 |                               60 |
|         2005 | RITA                      |                    169 |              10476 |        0.54465  |        3.11968 |                               3 |                               23 |
|         2008 | GUSTAV                    |                    276 |              21392 |        0.691568 |        2.80111 |                              16 |                               85 |
|         2020 | LAURA                     |                     83 |               5917 |        0.996403 |        2.78562 |                              17 |                               37 |
|         2021 | IDA                       |                    282 |              29024 |        0.832301 |        2.77269 |                              84 |                              133 |
|         2005 | KATRINA                   |                    109 |               9734 |        0.902382 |        2.56999 |                              16 |                               52 |
|         2020 | ZETA                      |                     89 |              11281 |        1.26313  |        2.3069  |                              52 |                               78 |
|         2012 | ISAAC                     |                    290 |              27573 |        0.589105 |        2.10765 |                               8 |                               53 |
|         2007 | HUMBERTO                  |                    280 |              21527 |        0.360371 |        2.07942 |                              12 |                               36 |
|         2002 | LILI                      |                    208 |              13260 |        0.55617  |        2.06448 |                               0 |                               31 |
|         2002 | ISIDORE                   |                     64 |               7572 |        0.812174 |        1.80007 |                               0 |                                0 |
|         2024 | FRANCINE                  |                    188 |              17382 |        0.662323 |        1.77096 |                              12 |                               97 |
|         2017 | HARVEY                    |                    220 |              22313 |        0.409424 |        1.54756 |                               0 |                                0 |
|         2019 | BARRY                     |                    316 |              32378 |        0.494593 |        1.45411 |                               3 |                               40 |
|         2003 | BILL                      |                    180 |              15945 |        0.406737 |        1.26599 |                               0 |                                0 |


## Top 20 school-years by point-level exposure

|   graduation_year | parish               | hs_name                     | max_exposure_storm_name   |   max_exposure_distance_km |   max_exposure_wind_kt |   exposure_index_pointmax | within_50km_hurricane   |
|------------------:|:---------------------|:----------------------------|:--------------------------|---------------------------:|-----------------------:|--------------------------:|:------------------------|
|              2021 | CAMERON              | SOUTH CAMERON HIGH SCHOOL   | DELTA                     |                    1.67684 |                    120 |                   3.18628 | True                    |
|              2006 | CAMERON              | JOHNSON BAYOU HIGH SCHOOL   | RITA                      |                    7.05458 |                    155 |                   3.11968 | True                    |
|              2009 | ST. MARY             | CENTERVILLE HIGH SCHOOL     | GUSTAV                    |                    5.34507 |                    135 |                   2.80111 | True                    |
|              2021 | CAMERON              | HACKBERRY HIGH SCHOOL       | LAURA                     |                   21.6683  |                    130 |                   2.78562 | True                    |
|              2022 | ST. JAMES            | ST. JAMES HIGH SCHOOL       | IDA                       |                   12.8694  |                    130 |                   2.77269 | True                    |
|              2022 | ST. JOHN THE BAPTIST | WEST ST. JOHN HIGH SCHOOL   | IDA                       |                   13.7361  |                    130 |                   2.71065 | True                    |
|              2021 | VERNON               | LEESVILLE HIGH SCHOOL       | LAURA                     |                    6.8575  |                    130 |                   2.66813 | True                    |
|              2021 | VERNON               | ANACOCO HIGH SCHOOL         | LAURA                     |                    6.90497 |                    130 |                   2.66416 | True                    |
|              2022 | JEFFERSON            | GRAND ISLE HIGH SCHOOL      | IDA                       |                   24.517   |                    130 |                   2.62536 | True                    |
|              2006 | ST. TAMMANY          | NORTHSHORE HIGH SCHOOL      | KATRINA                   |                   15.8561  |                    150 |                   2.56999 | True                    |
|              2021 | CAMERON              | GRAND LAKE HIGH SCHOOL      | LAURA                     |                   26.3507  |                    130 |                   2.53161 | True                    |
|              2009 | ST. MARY             | FRANKLIN SENIOR HIGH SCHOOL | GUSTAV                    |                    9.83416 |                    135 |                   2.44013 | True                    |
|              2022 | ST. JAMES            | LUTCHER HIGH SCHOOL         | IDA                       |                   18.6408  |                    130 |                   2.40601 | True                    |
|              2022 | ST. CHARLES          | HAHNVILLE HIGH SCHOOL       | IDA                       |                   18.7212  |                    130 |                   2.40158 | True                    |
|              2021 | CALCASIEU            | BELL CITY HIGH SCHOOL       | DELTA                     |                    6.39158 |                    120 |                   2.38918 | True                    |
|              2006 | ST. TAMMANY          | SALMEN HIGH SCHOOL          | KATRINA                   |                   19.0036  |                    150 |                   2.38617 | True                    |
|              2006 | ST. TAMMANY          | SLIDELL HIGH SCHOOL         | KATRINA                   |                   19.2382  |                    150 |                   2.37352 | True                    |
|              2021 | ST. TAMMANY          | SALMEN HIGH SCHOOL          | ZETA                      |                   11.846   |                    100 |                   2.3069  | True                    |
|              2022 | ST. JOHN THE BAPTIST | EAST ST. JOHN HIGH SCHOOL   | IDA                       |                   20.9722  |                    130 |                   2.28399 | True                    |
|              2009 | ST. MARY             | PATTERSON HIGH SCHOOL       | GUSTAV                    |                   12.7197  |                    135 |                   2.25346 | True                    |