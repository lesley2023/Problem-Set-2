"""Question 2(e): restricted steady-state out-of-sample forecasts."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


# Read and inspect the monthly observations of annual variables.
df = pd.read_csv("EQ Dataset.csv")
print(df.head())

df["date"] = pd.to_datetime(
    {"year": df["YEAR"], "month": df["MONTH"], "day": 1}
)
df = df.sort_values("date").reset_index(drop=True)

print("\nMissing values by column:")
print(df.isna().sum())

expected_dates = pd.date_range(df["date"].iloc[0], df["date"].iloc[-1], freq="MS")
if not df["date"].equals(pd.Series(expected_dates, name="date")):
    raise ValueError("The dataset is not a complete sequence of monthly observations.")

# Construct annual simple excess returns, dividend-price ratios, and growth.
df["xRe"] = np.exp(df["re"]) - np.exp(df["rf"])
df["DP"] = np.exp(df["dp"])
df["g"] = np.exp(df["dg"])

# Pair current DP with xRe exactly 12 months ahead.
pairs = pd.DataFrame(
    {
        "predictor_date": df["date"],
        "forecast_date": df["date"].shift(-12),
        "DP": df["DP"],
        "realized_xRe": df["xRe"].shift(-12),
    }
).dropna()

# Full-sample in-sample regression and fitted values.
X_full = sm.add_constant(pairs["DP"])
full_model = sm.OLS(pairs["realized_xRe"], X_full).fit()
pairs["IS_forecast"] = full_model.predict(X_full)

# Evaluate restricted forecasts from December 1940 through December 2021.
first_forecast_date = pd.Timestamp("1940-12-01")
evaluation = pairs.loc[pairs["forecast_date"] >= first_forecast_date].copy()
for column in ["G", "a_restricted", "b_restricted", "historical_mean", "OS_forecast"]:
    evaluation[column] = np.nan

first_training_dates = None
for row_index, row in evaluation.iterrows():
    # Historical return observations come from the expanding regression sample.
    training = pairs.loc[pairs["forecast_date"] <= row["predictor_date"]]

    # Use raw exp(dg_s) from exactly the same calendar months as the historical
    # return outcomes. The dg series is not shifted by 12 months.
    historical_growth = df.loc[df["date"].isin(training["forecast_date"]), "g"]
    if len(historical_growth) != len(training):
        raise ValueError("Return and dividend-growth historical samples do not align.")

    G = historical_growth.mean()
    historical_mean = training["realized_xRe"].mean()
    a_restricted = G - 1
    b_restricted = G
    os_forecast = a_restricted + b_restricted * row["DP"]

    evaluation.loc[row_index, "G"] = G
    evaluation.loc[row_index, "a_restricted"] = a_restricted
    evaluation.loc[row_index, "b_restricted"] = b_restricted
    evaluation.loc[row_index, "historical_mean"] = historical_mean
    evaluation.loc[row_index, "OS_forecast"] = os_forecast

    if first_training_dates is None:
        first_training_dates = (
            training["forecast_date"].iloc[0],
            training["forecast_date"].iloc[-1],
            len(training),
        )

# Full evaluation-period out-of-sample R-squared.
evaluation["OS_squared_error"] = (
    evaluation["realized_xRe"] - evaluation["OS_forecast"]
) ** 2
evaluation["mean_squared_error"] = (
    evaluation["realized_xRe"] - evaluation["historical_mean"]
) ** 2
r2_os = 1 - (
    evaluation["OS_squared_error"].sum()
    / evaluation["mean_squared_error"].sum()
)

# Plot the benchmark, full-sample fitted value, and restricted OS forecast.
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(
    evaluation["forecast_date"],
    evaluation["historical_mean"],
    label=r"Historical mean $\overline{xR}_{e,\tau}$",
    linewidth=1.8,
)
ax.plot(
    evaluation["forecast_date"],
    evaluation["IS_forecast"],
    label=r"In-sample $\widehat{E}^{IS}_{\tau}[xR_e]$",
    linewidth=1.4,
)
ax.plot(
    evaluation["forecast_date"],
    evaluation["OS_forecast"],
    label=r"Restricted OS $\widehat{E}^{OS}_{\tau}[xR_e]$",
    linewidth=1.6,
)
ax.set_xlabel("Forecast date")
ax.set_ylabel("Annual simple excess return")
ax.set_title("Restricted Steady-State Equity Excess-Return Forecasts")
ax.legend(frameon=False)
ax.grid(True, alpha=0.3)
ax.text(
    0.015,
    0.035,
    rf"Full-sample $R^2_{{OS}}={r2_os:.4f}$",
    transform=ax.transAxes,
    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
)
fig.tight_layout()
fig.savefig("figures/q2e_restricted_forecasts.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Rolling 50-year R-squared uses 600 forecast errors; G remains expanding.
window = 600
evaluation["rolling_OS_SSE"] = evaluation["OS_squared_error"].rolling(window).sum()
evaluation["rolling_mean_SSE"] = evaluation["mean_squared_error"].rolling(window).sum()
evaluation["rolling_R2_OS"] = 1 - (
    evaluation["rolling_OS_SSE"] / evaluation["rolling_mean_SSE"]
)
rolling = evaluation.loc[
    (evaluation["forecast_date"] >= pd.Timestamp("1990-12-01"))
    & evaluation["rolling_R2_OS"].notna(),
    ["forecast_date", "rolling_R2_OS"],
].copy()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(rolling["forecast_date"], rolling["rolling_R2_OS"], linewidth=2)
ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("End date of 50-year window")
ax.set_ylabel(r"50-year rolling $R^2_{OS}$")
ax.set_title(r"Restricted Forecast: Rolling 50-Year Out-of-Sample $R^2$")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("figures/q2e_rolling_r2_os.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Requested diagnostics.
print("\nQ2(e) diagnostics")
print(f"First OS forecast date: {evaluation['forecast_date'].iloc[0].date()}")
print(f"Last OS forecast date: {evaluation['forecast_date'].iloc[-1].date()}")
print(f"Number of OS forecasts: {len(evaluation)}")
print(
    "December 1940 G and historical-mean sample dates: "
    f"{first_training_dates[0].date()} to {first_training_dates[1].date()}"
)
print(f"December 1940 common historical sample size: {first_training_dates[2]}")
print("\nFirst restricted coefficients:")
print(evaluation[["forecast_date", "G", "a_restricted", "b_restricted"]].head())
print("\nFirst forecast observations:")
print(
    evaluation[
        [
            "forecast_date",
            "realized_xRe",
            "historical_mean",
            "IS_forecast",
            "OS_forecast",
        ]
    ].head()
)
print(f"\nFull-sample R2_OS: {r2_os:.4f}")
print(f"First rolling R2_OS date: {rolling['forecast_date'].iloc[0].date()}")
print(f"Last rolling R2_OS date: {rolling['forecast_date'].iloc[-1].date()}")
