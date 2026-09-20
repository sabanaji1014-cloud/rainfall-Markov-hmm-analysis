# Rainfall & Drought Analysis with Markov Chains and a Hidden Markov Model

Course project for **Stochastic Processes 1** (Instructor: Dr. Safari).

**Authors:** Mehrnaz Yazdani (610301201), Saba Naji (610301190)

## Overview

This project models the day-to-day and month-to-month behavior of rainfall/drought
conditions as a discrete-time Markov chain, and extends the analysis with a Hidden
Markov Model (HMM). It uses about 12 years (2009–2020) of daily weather observations.

The analysis has three parts:

1. **Part 1 — Binary drought/wet classification.** The Standardized Precipitation
   Index (SPI) is computed from monthly and yearly rainfall averages and converted to
   a binary state (`0` = drought, `1` = wet). Transition matrices are built for lags
   1 through 5 (both months and years), and AIC/BIC are used to compare the
   candidate Markov models. The best-fitting model uses the previous **year** as the
   single lag.
2. **Part 2 — Multi-category rainfall classification.** Mean rainfall is split into
   10 categories, and one-day and two-day transition matrices (with AIC/BIC) are
   computed between categories.
3. **Part 3 — Hidden Markov Model.** An HMM is fit to daily weather features (min
   temperature, wind, humidity, cloud cover, pressure, weather code) to infer hidden
   states, which are then compared against the observed rainy/sunny classification.

The full write-up, methodology and results tables are in [`report/`](report).

## Repository structure

```
.
├── data/
│   ├── raw/
│   │   └── project.xlsx                 # Daily weather data, 2009-2020 (source data)
│   └── processed/
│       ├── rain_data.csv                # Extracted daily rainfall series
│       ├── spi_data.csv                 # Early monthly SPI output
│       ├── SPI_data_monthly_with_categories.xlsx
│       ├── SPI_data_yearly_with_categories.xlsx
│       ├── transition_matrix_Mean_Rainfall_category.csv
│       ├── transition_matrix_Mean_Rainfall_category_one_day.csv
│       └── transition_matrix_Mean_Rainfall_category_two_days.csv
├── src/
│   ├── project.py         # Part 1, v1: SPI + binary Markov chain (lags 1-5)
│   ├── project2.py        # Alternative manual SPI calculation (numerical integration)
│   ├── ST_part_b.py       # Part 2, intermediate: adds the 10-category rainfall classification
│   ├── video_way.py       # Part 1, intermediate: adds AIC/BIC to the binary Markov chain
│   ├── video_way2.py       # Final script: Parts 1 & 2 combined, incl. AIC/BIC and the
│   │                       #   one-day/two-day category transition matrices
│   └── HMM.py              # Part 3: Hidden Markov Model
├── report/
│   ├── Stochastic Processes.docx   # Full report (editable source)
│   └── Stochastic processes.pdf    # Compiled PDF version
├── requirements.txt
└── .gitignore
```

`data/raw/project.xlsx` is the single source dataset; every file in `data/processed/`
was generated from it by the scripts in `src/`.

## Scripts

The `src/` scripts represent the evolution of the analysis rather than independent,
unrelated files — they are kept as separate steps so the original work and its
history are preserved, in the order below:

| File | Role |
|---|---|
| `project.py` | First full version of Part 1: computes SPI, classifies drought/wet, prints binary transition matrices for lags 1–5. |
| `project2.py` | An alternative, from-scratch SPI computation (manual numerical integration instead of `scipy.stats.gamma`). Kept for reference; not the version used for the final results. |
| `ST_part_b.py` | Adds the 10-bin mean-rainfall categorization used in Part 2. |
| `video_way.py` | Adds AIC/BIC model comparison to the Part 1 binary Markov chain. |
| `video_way2.py` | **Final, consolidated script.** Combines Part 1 (binary SPI Markov chain + AIC/BIC) and Part 2 (rainfall categories + one-day/two-day transition matrices + AIC/BIC). This is the script that produced every file in `data/processed/`. |
| `HMM.py` | Part 3, the Hidden Markov Model. Originally written and run in Google Colab against an external, preprocessed dataset (`data_new.xlsx`, with an added `Weather_Code` column) that is not part of this repository, so it will not run as-is — it is included for documentation of the method. |

**Only the file paths inside these scripts were changed**, to match this
repository's `data/raw` / `data/processed` layout (some originally pointed to
`project.xlsx` in the same folder, and one to an absolute path on the author's own
computer). No other code, logic, or variable was modified.

## How to run

```bash
pip install -r requirements.txt
cd src
python video_way2.py
```

This regenerates the contents of `data/processed/`. `HMM.py` needs the external
dataset described above and will not run without it.

## Data

`data/raw/project.xlsx` contains daily observations from 2009 to 2020 with the
columns: `Year` (date), `Weather` (text description), `Max`/`Min` (temperature),
`Wind`, `Rain`, `Humidity`, `Cloud`, `Pressure`.

## License

No license has been added yet; all rights are reserved by the authors.
