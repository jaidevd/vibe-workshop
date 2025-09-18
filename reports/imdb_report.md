# IMDb Movie Trends Report

## Data Snapshot

* Downloaded from [datasets.imdbws.com](https://datasets.imdbws.com/) on 2025-09-18 07:29 UTC.
* Focused on mainstream feature films released from 1980 onwards (counts capped at 2024).
* 445,353 feature films in scope; 56.7% have audience rating data.
* Median runtime 92 minutes and median IMDb rating 6.3.

## Release Growth

Mainstream film production has accelerated sharply since the 2000s. The five-year view below (1980–2024) shows streaming-era years (2015 onwards) averaging more than twice as many releases per year as the 1980s.

| period    |   total_movies |   annual_avg |
|:----------|---------------:|-------------:|
| 1980-1984 |          19602 |       3920.4 |
| 1985-1989 |          21241 |       4248.2 |
| 1990-1994 |          21537 |       4307.4 |
| 1995-1999 |          23353 |       4670.6 |
| 2000-2004 |          29946 |       5989.2 |
| 2005-2009 |          48170 |       9634   |
| 2010-2014 |          74169 |      14833.8 |
| 2015-2019 |          93851 |      18770.2 |
| 2020-2024 |          97828 |      19565.6 |

## Genres audiences love

Recent big-audience hits (2010 onwards, 25k+ votes) show clear genre patterns. History, Biography, Animation top the list by average rating, and all top genres have more than 100 major releases in the last decade.

| Genre     |   Films |   Avg rating |   Median rating |   % ≥ 7.5 |   Median votes |
|:----------|--------:|-------------:|----------------:|----------:|---------------:|
| History   |     130 |         7.11 |            7.15 |      29.2 |          53428 |
| Biography |     294 |         7.1  |            7.1  |      28.9 |          60350 |
| Animation |     191 |         6.95 |            7    |      29.8 |          82199 |
| Drama     |    1738 |         6.79 |            6.9  |      21.5 |          61294 |
| Crime     |     609 |         6.63 |            6.6  |      14.9 |          66743 |
| Romance   |     413 |         6.53 |            6.5  |      13.1 |          55303 |
| Adventure |     683 |         6.53 |            6.6  |      14.9 |         125757 |
| Comedy    |    1042 |         6.46 |            6.5  |      11.4 |          66627 |
| Action    |    1023 |         6.41 |            6.4  |      14.7 |         101292 |
| Thriller  |     579 |         6.38 |            6.4  |      11.9 |          73361 |

## IMDb rating distribution

Audience scores skew toward the middle of the 10-point scale, leaving both extremes sparsely populated. 6.0-6.9 ratings lead the pack with 71,125 titles (28.1% of rated features), edging past 7.0-7.9 at 53,573 titles (21.2%). Mid-5 scores account for 52,722 titles (20.9%), while the lowest band (1.0-1.9) only captures 994 titles (0.4%). High-end acclaim remains rare: 4,604 titles sit at 9.0 or above (1.8% of the set) yet those releases still attract 10.2M cumulative votes. No rated feature in the dataset falls below 1.0.

| Rating band   |   Titles |   Total votes |   % of titles |   % of votes |   Avg votes |
|:--------------|---------:|--------------:|--------------:|-------------:|------------:|
| 1.0-1.9       |      994 |       1456467 |           0.4 |          0.1 |        1465 |
| 2.0-2.9       |     4654 |       3823064 |           1.8 |          0.3 |         821 |
| 3.0-3.9       |    13698 |       9675319 |           5.4 |          0.9 |         706 |
| 4.0-4.9       |    29439 |      29218662 |          11.7 |          2.6 |         993 |
| 5.0-5.9       |    52722 |     121814980 |          20.9 |         10.9 |        2311 |
| 6.0-6.9       |    71125 |     345485852 |          28.1 |         30.9 |        4857 |
| 7.0-7.9       |    53573 |     407467722 |          21.2 |         36.4 |        7606 |
| 8.0-8.9       |    21862 |     189093382 |           8.7 |         16.9 |        8649 |
| 9.0-9.9       |     4601 |      10161074 |           1.8 |          0.9 |        2208 |
| 10.0          |        3 |            19 |           0   |          0   |           6 |

## Hypothesis fact-checks

We compared the published hypotheses with the 1980+ feature-film aggregates used throughout this report. The modal rating band in this sample is 6.0-6.9. The lowest recorded IMDb score is 1.0. Tables below compare the stated expectations with observed values (Δ = actual − expected).

Key discrepancies:
- **Ratings:** 7.0–7.9 titles dominate: 53,573 vs 534,788; share 21.2% vs 33.1%; votes 407.5M vs 588.0M; 6.0–6.9 ranks second: 71,125 vs 379,310; share 28.1% vs 23.5%; Mid-5 scores remain modest: 52,722 vs 193,466; share 20.9% vs 12.0%; Low-end 1.x ratings are rare: 994 vs 6,509; High-end 9.0–10.0 acclaim is scarce: 4,604 vs 77,073; share 1.8% vs 4.8%; votes 10.2M vs 64.0M

**Rating distribution claims**

| Claim                               | Metric    | Expected   | Actual   | Difference   | Verdict   |
|:------------------------------------|:----------|:-----------|:---------|:-------------|:----------|
| 7.0–7.9 titles dominate             | Titles    | 534,788    | 53,573   | -481,215     | Differs   |
| 7.0–7.9 titles dominate             | Share (%) | 33.1%      | 21.2%    | -11.9pp      | Differs   |
| 7.0–7.9 titles dominate             | Votes     | 588.0M     | 407.5M   | -180.5M      | Differs   |
| 6.0–6.9 ranks second                | Titles    | 379,310    | 71,125   | -308,185     | Differs   |
| 6.0–6.9 ranks second                | Share (%) | 23.5%      | 28.1%    | +4.6pp       | Differs   |
| Mid-5 scores remain modest          | Titles    | 193,466    | 52,722   | -140,744     | Differs   |
| Mid-5 scores remain modest          | Share (%) | 12.0%      | 20.9%    | +8.9pp       | Differs   |
| Low-end 1.x ratings are rare        | Titles    | 6,509      | 994      | -5,515       | Differs   |
| Low-end 1.x ratings are rare        | Share (%) | 0.4%       | 0.4%     | +0.0pp       | Matches   |
| High-end 9.0–10.0 acclaim is scarce | Titles    | 77,073     | 4,604    | -72,469      | Differs   |
| High-end 9.0–10.0 acclaim is scarce | Share (%) | 4.8%       | 1.8%     | -3.0pp       | Differs   |
| High-end 9.0–10.0 acclaim is scarce | Votes     | 64.0M      | 10.2M    | -53.8M       | Differs   |

**Lockdown spike claims**

| Genre      | Metric             | Expected   | Actual   |   Difference | Verdict   |
|:-----------|:-------------------|:-----------|:---------|-------------:|:----------|
| Reality-TV | Peak year          | 2020       | 2020     |            0 | Matches   |
| Reality-TV | Lockdown releases  | 78         | 78       |            0 | Matches   |
| Reality-TV | Avg pre-lockdown   | 37.7       | 37.7     |            0 | Matches   |
| Reality-TV | Avg post-lockdown  | 25.0       | 25.0     |            0 | Matches   |
| Reality-TV | × vs pre-lockdown  | 2.07×      | 2.07×    |            0 | Matches   |
| Reality-TV | × vs post-lockdown | 3.12×      | 3.12×    |            0 | Matches   |
| Talk-Show  | Peak year          | 2020       | 2020     |            0 | Matches   |
| Talk-Show  | Lockdown releases  | 26         | 26       |            0 | Matches   |
| Talk-Show  | Avg pre-lockdown   | 13.0       | 13.0     |            0 | Matches   |
| Talk-Show  | Avg post-lockdown  | 11.7       | 11.7     |            0 | Matches   |
| Talk-Show  | × vs pre-lockdown  | 2.00×      | 2.00×    |            0 | Matches   |
| Talk-Show  | × vs post-lockdown | 2.23×      | 2.23×    |            0 | Matches   |

## Lockdown-era genre spikes

A handful of genres hit their highest release counts during the 2020–2021 lockdown period. Each listed genre released at least 25 feature films in its peak lockdown year and outpaced the 2017-2019 average by 20% or more. Leading spikes include Reality-TV (78 films in 2020, 2.1× pre-lockdown); Talk-Show (26 films in 2020, 2.0× pre-lockdown). The table compares the lockdown surge with the 2017-2019 baseline and the 2022-2024 recovery window.

| Genre      |   Peak year |   Lockdown releases |   Avg 2017-2019 |   Avg 2022-2024 |   × vs pre-lockdown |   × vs post-lockdown |
|:-----------|------------:|--------------------:|----------------:|----------------:|--------------------:|---------------------:|
| Reality-TV |        2020 |                  78 |            37.7 |            25   |                2.07 |                 3.12 |
| Talk-Show  |        2020 |                  26 |            13   |            11.7 |                2    |                 2.23 |

## Runtime and ratings

Longer movies are not dramatically better rated, but the correlation is mildly positive (0.32). Movies over 150 minutes gain the highest average scores, while sub-90-minute releases trail.

| Runtime band   |   Films |   Avg rating |   Avg minutes |
|:---------------|--------:|-------------:|--------------:|
| [0, 90)        |    1826 |         5.89 |          83.9 |
| [90, 120)      |    8918 |         6.29 |         102.6 |
| [120, 150)     |    2722 |         6.84 |         130.9 |
| [150, 1000)    |     872 |         7.03 |         166.2 |
