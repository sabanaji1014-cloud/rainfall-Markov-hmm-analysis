"""
Rainfall & Drought Analysis with Markov Chains and a Hidden Markov Model
=========================================================================

Stochastic Processes 1 -- course project
Authors: Mehrnaz Yazdani (610301201), Saba Naji (610301190)

This single script reproduces the full analysis described in the project
report, consolidating what used to be five separate draft scripts
(project.py, project2.py, ST_part_b.py, video_way.py, video_way2.py, HMM.py)
into one clean, documented pipeline.

    Part 1 - Binary (drought/wet) classification of rainfall via the
             Standardized Precipitation Index (SPI), modeled as a
             discrete-time Markov chain over 1-5 month/year lags, with
             AIC/BIC used to compare candidate lags.
    Part 2 - A finer, 10-level classification of mean rainfall, modeled
             as one-day and two-day (order-1 and order-2) Markov chains,
             again compared via AIC/BIC.
    Part 3 - A Hidden Markov Model (HMM) fit to monthly-averaged weather
             features, used to infer a hidden 2-state sequence and compare
             it against a simple rainy/sunny classification of the same
             months.

Two additional, self-contained pieces are kept for reference (they were
not used to produce the final results, but were part of the original
exploration and are documented here rather than dropped):

    - `classify_spi_severity`: a 9-level SPI severity scale (from
      "extreme drought" to "extremely wet"), which was not used in the
      final binary classification.
    - `compute_spi_manual_integration`: an alternative, from-scratch SPI
      calculation via direct numerical integration, instead of
      `scipy.stats.gamma`.

Run directly to regenerate every file in ../data/processed/:

    pip install -r ../requirements.txt
    python analysis.py

Requires ../data/raw/project.xlsx (raw daily weather data, 2009-2020).
"""

from __future__ import annotations

import datetime as dt
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy import integrate
from hmmlearn import hmm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
RAW_DATA = SRC_DIR.parent / "data" / "raw" / "project.xlsx"
PROCESSED_DIR = SRC_DIR.parent / "data" / "processed"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_daily_weather(path: Path = RAW_DATA) -> pd.DataFrame:
    """Load the raw daily weather workbook into a tidy, numeric DataFrame
    indexed by date.

    The source file stores some columns as text with units baked in
    (e.g. "3.4 mm", "1022 mb"); those are converted to plain floats here
    so every later step can work with numbers directly instead of
    re-parsing strings.
    """
    df = pd.read_excel(path)
    df = df.rename(columns={"Year": "Date"})

    # The source workbook has one corrupted row (a stray "0" where the date
    # should be, with garbage numeric values in the other columns too) --
    # without this filter it gets parsed as 1970-01-01 and silently corrupts
    # every later resample/rolling calculation. Real rows are always either
    # a datetime or an "MM/DD/YYYY" string, so this drops just that row.
    df = df[df["Date"].apply(lambda v: isinstance(v, (dt.datetime, str)))]
    df["Date"] = pd.to_datetime(df["Date"])

    def _strip_units(series: pd.Series) -> pd.Series:
        if series.dtype != object:
            return series.astype(float)
        return (
            series.astype(str)
            .str.replace(r"[^\d\.\-]", "", regex=True)
            .replace("", np.nan)
            .astype(float)
        )

    for col in ["Rain", "Max", "Min", "Wind", "Pressure"]:
        df[col] = _strip_units(df[col])
    for col in ["Humidity", "Cloud"]:
        # Already stored as 0-1 fractions in the source workbook; the
        # helper only kicks in if a future version stores them as "62%".
        if df[col].dtype == object:
            df[col] = _strip_units(df[col]) / 100.0

    return df.set_index("Date").sort_index()


# ---------------------------------------------------------------------------
# Part 1a -- Standardized Precipitation Index (SPI)
# ---------------------------------------------------------------------------
def compute_spi(series: pd.Series, window: int = 1) -> pd.Series:
    """Standardized Precipitation Index (McKee et al., 1993).

    Fits a two-parameter gamma distribution to a rolling-mean rainfall
    series and returns the corresponding standard-normal quantile for
    each point.
    """
    rolling_mean = series.rolling(window, center=False).mean()
    rolling_mean = rolling_mean.where(rolling_mean > 0)
    log_mean = np.log(rolling_mean).replace([np.inf, -np.inf], np.nan)

    mu = np.nanmean(rolling_mean)
    log_sum = np.nansum(log_mean)
    n = len(log_mean.iloc[window - 1:])

    A = np.log(mu) - (log_sum / n)
    alpha = (1 / (4 * A)) * (1 + np.sqrt(1 + (4 * A) / 3))
    beta = mu / alpha

    cdf = st.gamma.cdf(rolling_mean, a=alpha, scale=beta)
    return pd.Series(st.norm.ppf(cdf, loc=0, scale=1), index=series.index, name="SPI")


# SPI severity scale (McKee et al., 1993), used for reference/reporting only
# -- it is not the classification used in the Markov-chain models below,
# which use the simpler binary drought(0)/wet(1) split instead.
_SPI_SEVERITY_LEVELS = [
    (2.0, math.inf, "extremely wet", 0),
    (1.5, 2.0, "severely wet", 1),
    (1.0, 1.5, "moderately wet", 2),
    (0.5, 1.0, "mildly wet", 3),
    (-0.5, 0.5, "near normal", 4),
    (-1.0, -0.5, "mildly dry", 5),
    (-1.5, -1.0, "moderately dry", 6),
    (-2.0, -1.5, "severely dry", 7),
    (-math.inf, -2.0, "extremely dry", 8),
]


def classify_spi_severity(spi_value: float) -> tuple[str, int]:
    """Map an SPI value to the standard 9-level severity scale.

    Kept for reference (this scale is not what the binary Markov-chain
    models below are built on).
    """
    if pd.isna(spi_value):
        return "unknown", 9
    for lower, upper, label, code in _SPI_SEVERITY_LEVELS:
        if lower <= spi_value < upper:
            return label, code
    return "unknown", 9


def compute_spi_manual_integration(monthly_rain_values: list[float]) -> float:
    """Alternative SPI calculation via direct numerical integration of the
    gamma distribution, instead of `scipy.stats.gamma` (originally
    `project2.py`). Kept for reference as a different approach; it was
    not used to produce the final results tables.

    Note: the inverse-normal approximation below uses the standard
    Abramowitz & Stegun (1964) rational formula
        t - (c0 + c1*t + c2*t^2) / (1 + d1*t + d2*t^2 + d3*t^3)
    The original draft was missing the parentheses around the
    denominator; that arithmetic slip is fixed here.
    """
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308

    values = [v for v in monthly_rain_values]
    n = len(values)
    if n == 0:
        return 0.0

    mean_rain = sum(values) / n
    zero_count = sum(1 for v in values if v == 0)
    log_sum = sum(math.log(v) for v in values if v != 0)

    A = math.log(mean_rain) - (log_sum / n) if n > 0 else 0.0
    if (1 + (4 * A) / 3) < 0:
        alpha = 0.001
    else:
        alpha = (1 / (4 * A)) * (1 + math.sqrt(1 + (4 * A) / 3))
    beta = round(mean_rain / alpha)
    if beta == 0:
        return 0.0

    gamma_alpha, _ = integrate.quad(lambda y: y ** (alpha - 1) * math.exp(-y), 0, np.inf)
    gamma_alpha = round(gamma_alpha)
    if gamma_alpha in (0, 1):
        return 0.0

    def _gamma_pdf(x: float) -> float:
        return (1 / ((beta ** round(alpha)) * gamma_alpha)) * (x ** (alpha - 1)) * math.exp(-x / beta)

    cdf_value, _ = integrate.quad(_gamma_pdf, 0, max(values))
    cdf_value = round(cdf_value, 5)

    q = zero_count / n
    h = q + (1 - q) * cdf_value
    h_rounded = round(h, 2)

    if 0 < h_rounded <= 0.5:
        t = math.sqrt(math.log(1 / (h ** 2)))
        return -(t - (c0 + c1 * t + c2 * t ** 2) / (1 + d1 * t + d2 * t ** 2 + d3 * t ** 3))
    elif 0.5 < h_rounded <= 1:
        t = math.sqrt(math.log(1 / ((1 - h) ** 2)))
        return t - (c0 + c1 * t + c2 * t ** 2) / (1 + d1 * t + d2 * t ** 2 + d3 * t ** 3)
    return 0.0


# ---------------------------------------------------------------------------
# Part 1b -- Binary drought/wet Markov chain, lags 1-5, with AIC/BIC
# ---------------------------------------------------------------------------
def _binary_patterns(length: int) -> list[tuple[int, ...]]:
    return list(itertools.product([0, 1], repeat=length))


def binary_transition_matrix(binary_series: list[int], lag: int) -> tuple[pd.DataFrame, float, float]:
    """Transition matrix, AIC and BIC for a binary (drought=0/wet=1) Markov
    chain of the given lag, i.e. P(next state | previous `lag` states).
    """
    patterns = _binary_patterns(lag)
    rows = []
    log_likelihood = 0.0

    for pattern in patterns:
        matches = sum(
            1 for i in range(len(binary_series) - lag)
            if tuple(binary_series[i:i + lag]) == pattern
        )
        wet_next = sum(
            1 for i in range(len(binary_series) - lag)
            if tuple(binary_series[i:i + lag]) == pattern and binary_series[i + lag] == 1
        )
        p_wet = wet_next / matches if matches else 0.0
        p_drought = 1 - p_wet
        rows.append([p_drought, p_wet])

        drought_next = matches - wet_next
        if p_drought > 0 and drought_next > 0:
            log_likelihood += drought_next * np.log(p_drought)
        if p_wet > 0 and wet_next > 0:
            log_likelihood += wet_next * np.log(p_wet)

    table = pd.DataFrame(rows, index=patterns, columns=["P(0)", "P(1)"])
    n = len(binary_series)
    k = (2 ** lag) * (2 ** lag) - (2 ** lag)
    aic = 2 * k - 2 * log_likelihood
    bic = np.log(n) * k - 2 * log_likelihood
    return table, aic, bic


def run_binary_markov_analysis(spi_yearly: pd.DataFrame, spi_monthly: pd.DataFrame) -> pd.DataFrame:
    """Part 1: build the binary drought/wet transition matrix for lags 1-5,
    for both the yearly and monthly SPI series, and collect the AIC/BIC of
    every candidate model into one summary table.
    """
    series_by_label = {
        "yearly": spi_yearly["SPI_binary"].tolist(),
        "monthly": spi_monthly["SPI_binary"].tolist(),
    }

    summary_rows = []
    for label, binary_series in series_by_label.items():
        for lag in range(1, 6):
            table, aic, bic = binary_transition_matrix(binary_series, lag)
            table.to_csv(PROCESSED_DIR / f"transition_matrix_binary_{label}_lag{lag}.csv")
            summary_rows.append({"series": label, "lag": lag, "AIC": aic, "BIC": bic})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(PROCESSED_DIR / "aic_bic_binary_markov_summary.csv", index=False)
    return summary


# ---------------------------------------------------------------------------
# Part 2 -- 10-level rainfall categories, one-day / two-day Markov chains
# ---------------------------------------------------------------------------
def categorize_mean_rainfall(value: float) -> float:
    """Bin mean rainfall (mm) into 10 categories: 0 for [0,1), 1 for
    [1,2), ..., 9 for [9,10). Values outside [0, 10) are left unclassified.
    """
    if pd.isna(value):
        return np.nan
    for lower in range(10):
        if lower <= value < lower + 1:
            return float(lower)
    return np.nan


def _log_likelihood_categorical(transition_matrix: pd.DataFrame, categories: pd.Series, order: int) -> float:
    log_likelihood = 0.0
    for i in range(order, len(categories)):
        prev = tuple(categories.iloc[i - order:i]) if order > 1 else categories.iloc[i - 1]
        current = categories.iloc[i]
        if pd.isna(current) or (isinstance(prev, tuple) and any(pd.isna(p) for p in prev)) or pd.isna(prev if not isinstance(prev, tuple) else 0):
            continue
        prob = transition_matrix.at[prev, current]
        if prob > 0:
            log_likelihood += np.log(prob)
    return log_likelihood


def one_day_category_transition(categories: pd.Series) -> tuple[pd.DataFrame, float, float]:
    """One-day (order-1) transition matrix between rainfall categories,
    with AIC/BIC."""
    levels = sorted(categories.dropna().unique())
    matrix = pd.DataFrame(0.0, index=levels, columns=levels)

    for i in range(1, len(categories)):
        prev, current = categories.iloc[i - 1], categories.iloc[i]
        if pd.notna(prev) and pd.notna(current):
            matrix.at[prev, current] += 1

    matrix = matrix.div(matrix.sum(axis=1), axis=0).fillna(0)

    n = len(categories) - 1
    k = len(levels) * (len(levels) - 1)
    log_likelihood = _log_likelihood_categorical(matrix, categories, order=1)
    aic = 2 * k - 2 * log_likelihood
    bic = np.log(n) * k - 2 * log_likelihood
    return matrix, aic, bic


def two_day_category_transition(categories: pd.Series) -> tuple[pd.DataFrame, float, float]:
    """Two-day (order-2) transition matrix between rainfall categories,
    with AIC/BIC."""
    levels = sorted(categories.dropna().unique())
    index = pd.MultiIndex.from_product([levels, levels])
    matrix = pd.DataFrame(0.0, index=index, columns=levels)

    for i in range(2, len(categories)):
        prev2, prev1, current = categories.iloc[i - 2], categories.iloc[i - 1], categories.iloc[i]
        if pd.notna(prev2) and pd.notna(prev1) and pd.notna(current):
            matrix.at[(prev2, prev1), current] += 1

    matrix = matrix.div(matrix.sum(axis=1), axis=0).fillna(0)

    n = len(categories) - 2
    k = len(levels) ** 3 - len(levels) ** 2
    log_likelihood = 0.0
    for i in range(2, len(categories)):
        prev2, prev1, current = categories.iloc[i - 2], categories.iloc[i - 1], categories.iloc[i]
        if pd.notna(prev2) and pd.notna(prev1) and pd.notna(current):
            prob = matrix.at[(prev2, prev1), current]
            if prob > 0:
                log_likelihood += np.log(prob)
    aic = 2 * k - 2 * log_likelihood
    bic = np.log(n) * k - 2 * log_likelihood
    return matrix, aic, bic


def run_category_markov_analysis(spi_monthly: pd.DataFrame) -> None:
    """Part 2: one-day and two-day transition matrices between the 10
    rainfall categories, with AIC/BIC, saved to data/processed/."""
    categories = spi_monthly["Mean_Rainfall_category"]

    one_day, aic_1, bic_1 = one_day_category_transition(categories)
    two_day, aic_2, bic_2 = two_day_category_transition(categories)

    one_day.to_csv(PROCESSED_DIR / "transition_matrix_Mean_Rainfall_category_one_day.csv")
    two_day.to_csv(PROCESSED_DIR / "transition_matrix_Mean_Rainfall_category_two_days.csv")

    print("Transition matrix for Mean_Rainfall_category (one day):")
    print(one_day)
    print(f"AIC: {aic_1:.2f}  BIC: {bic_1:.2f}\n")

    print("Transition matrix for Mean_Rainfall_category (two days):")
    print(two_day)
    print(f"AIC: {aic_2:.2f}  BIC: {bic_2:.2f}\n")


# ---------------------------------------------------------------------------
# Part 3 -- Hidden Markov Model on monthly weather features
# ---------------------------------------------------------------------------
def build_monthly_hmm_features(weather: pd.DataFrame) -> pd.DataFrame:
    """Monthly-averaged weather features used to fit the HMM, following the
    "compute monthly averages" step described in the report. `Weather`
    (a text description, e.g. "Partly cloudy") is numerically encoded as
    `Weather_Code` since the HMM needs numeric input.
    """
    df = weather.copy()
    df["Weather_Code"], _ = pd.factorize(df["Weather"])
    monthly = df.resample("ME").mean(numeric_only=True)
    return monthly[["Min", "Wind", "Humidity", "Cloud", "Pressure", "Weather_Code", "Rain"]]


def run_hmm_analysis(weather: pd.DataFrame) -> pd.DataFrame:
    """Part 3: fit a 2-state Gaussian HMM on standardized monthly weather
    features and compare the inferred hidden states against a simple
    rainy/sunny classification of the same months (median-of-monthly-mean
    rainfall threshold, as described in the report).

    This replaces the original HMM.py, which depended on an external,
    already-preprocessed dataset (data_new.xlsx) that is not included in
    this repository, and still referenced a couple of undefined variables
    left over from that notebook. This version runs end-to-end on
    data/raw/project.xlsx alone.
    """
    monthly = build_monthly_hmm_features(weather)
    feature_cols = ["Min", "Wind", "Humidity", "Cloud", "Pressure", "Weather_Code"]
    standardized = (monthly[feature_cols] - monthly[feature_cols].mean()) / monthly[feature_cols].std()

    model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=1000, random_state=0)
    model.fit(standardized.to_numpy())
    hidden_states = model.predict(standardized.to_numpy())

    threshold = monthly["Rain"].median()
    rain_state = (monthly["Rain"] > threshold).astype(int)

    result = monthly.copy()
    result["hidden_state"] = hidden_states
    result["rain_state"] = rain_state.to_numpy()
    result.to_csv(PROCESSED_DIR / "hmm_monthly_states.csv")

    # The HMM's two labels (0/1) are arbitrary (label-switching), so the
    # hidden state is compared against the rain classification both ways
    # and the better-matching orientation is reported.
    direct_agreement = (result["hidden_state"] == result["rain_state"]).mean()
    swapped_agreement = (result["hidden_state"] == (1 - result["rain_state"])).mean()
    agreement = max(direct_agreement, swapped_agreement)

    print(f"HMM hidden-state vs. rainy/sunny agreement: {agreement:.1%}")
    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    weather = load_daily_weather()

    # --- Part 1: SPI, binary classification, severity scale -------------
    yearly_rain = weather["Rain"].resample("YE").mean()
    monthly_rain = weather["Rain"].resample("ME").mean()

    spi_yearly = pd.DataFrame({"SPI": compute_spi(yearly_rain, 1)})
    spi_yearly["SPI_binary"] = np.where(spi_yearly["SPI"] <= 0, 0, 1)
    spi_yearly["Severity"], spi_yearly["Severity_code"] = zip(
        *spi_yearly["SPI"].map(classify_spi_severity)
    )
    spi_yearly["Mean_Rainfall"] = yearly_rain.to_numpy()
    spi_yearly["Mean_Rainfall_category"] = spi_yearly["Mean_Rainfall"].map(categorize_mean_rainfall)

    spi_monthly = pd.DataFrame({"SPI": compute_spi(monthly_rain, 1)})
    spi_monthly["SPI_binary"] = np.where(spi_monthly["SPI"] <= 0, 0, 1)
    spi_monthly["Severity"], spi_monthly["Severity_code"] = zip(
        *spi_monthly["SPI"].map(classify_spi_severity)
    )
    spi_monthly["Mean_Rainfall"] = monthly_rain.to_numpy()
    spi_monthly["Mean_Rainfall_category"] = spi_monthly["Mean_Rainfall"].map(categorize_mean_rainfall)

    for df in (spi_yearly, spi_monthly):
        df.index.name = "Date"
        df.reset_index(inplace=True)
        df["Date"] = df["Date"].dt.strftime("%m/%d/%Y")

    with pd.ExcelWriter(PROCESSED_DIR / "SPI_data_yearly_with_categories.xlsx", date_format="mm/dd/yyyy") as writer:
        spi_yearly.to_excel(writer, sheet_name="Yearly", index=False)
    with pd.ExcelWriter(PROCESSED_DIR / "SPI_data_monthly_with_categories.xlsx", date_format="mm/dd/yyyy") as writer:
        spi_monthly.to_excel(writer, sheet_name="Monthly", index=False)

    binary_summary = run_binary_markov_analysis(spi_yearly, spi_monthly)
    print("Binary (drought/wet) Markov chain -- AIC/BIC by lag:")
    print(binary_summary.to_string(index=False))
    print()

    # --- Part 2: rainfall categories, one-day / two-day transitions -----
    run_category_markov_analysis(spi_monthly)

    # --- Part 3: Hidden Markov Model -------------------------------------
    run_hmm_analysis(weather)


if __name__ == "__main__":
    main()
