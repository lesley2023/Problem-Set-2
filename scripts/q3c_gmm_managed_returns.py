"""Implement Prompts/Q3c_prompt.txt. Run: python3 scripts/q3c_gmm_managed_returns.py.

The forecast loop reproduces scripts/Pset1_q2e.py's expanding historical
sample and 12-month predictor/outcome pairing, without its unrelated outputs.
"""

from q3b_gmm_risk_aversion import ROOT, newey_west_1994

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import chi2


def reproduce_forecasts(data):
    """Port the supplied Pset 1 Q2(e) forecast construction and training rule."""
    pairs = pd.DataFrame({
        "predictor_date": data.index,
        "forecast_date": pd.Series(data.index).shift(-12).to_numpy(),
        "DP": np.exp(data["dp"].to_numpy()),
        "realized_xRe": data["xRe"].shift(-12).to_numpy(),
    }).dropna()
    evaluation = pairs.loc[pairs["forecast_date"].between(
        pd.Timestamp("1940-12-01"), pd.Timestamp("2021-12-01"),
    )].copy()
    audit = []
    for _, row in evaluation.iterrows():
        training = pairs.loc[pairs["forecast_date"] <= row["predictor_date"]]
        historical_growth = np.exp(data.loc[data.index.isin(training["forecast_date"]), "dg"])
        if len(training) == 0 or len(historical_growth) != len(training):
            raise ValueError("Forecast training and historical growth samples do not align.")
        growth = historical_growth.mean()
        forecast = (growth - 1) + growth * row["DP"]
        audit.append({
            "predictor_date": row["predictor_date"],
            "forecast_date": row["forecast_date"],
            "training_start": training["forecast_date"].iloc[0],
            "training_end": training["forecast_date"].iloc[-1],
            "training_observations": len(training),
            "G": growth, "DP": row["DP"], "OS_forecast": forecast,
            "realized_xRe": row["realized_xRe"],
        })
    result = pd.DataFrame(audit)
    expected = pd.date_range("1940-12-01", "2021-12-01", freq="MS")
    if not pd.DatetimeIndex(result["forecast_date"]).equals(expected):
        raise ValueError("Forecasts must cover every month from December 1940 to December 2021.")
    return result


def moment_hac(moment_matrix):
    """One NW94 lag from equal-weight centered moments, used for the whole S."""
    centered = moment_matrix - moment_matrix.mean(axis=0)
    _, lag, details = newey_west_1994(centered.sum(axis=1))
    n = len(centered)
    covariance = np.einsum("ti,tj->ij", centered, centered) / n
    for ell in range(1, lag + 1):
        lag_cov = np.einsum("ti,tj->ij", centered[ell:], centered[:-ell]) / (n - ell)
        weight = 1 - ell / (lag + 1)
        covariance += weight * (lag_cov + lag_cov.T)
    if not np.isfinite(covariance).all() or np.linalg.eigvalsh(covariance).min() <= 0:
        raise ValueError("The moment HAC matrix must be finite and positive definite.")
    return covariance, lag, details


def load_estimation_sample(reuse_saved_forecasts=False):
    """Shared Q3(c)/(d) inputs; saved forecasts already name outcome dates."""
    data = pd.read_csv(ROOT / "datasets/EQ Dataset.csv")
    data.index = pd.DatetimeIndex(pd.to_datetime(dict(
        year=data["YEAR"], month=data["MONTH"], day=1,
    )), name="forecast_date")
    data = data.sort_index()
    required = data[["re", "rf", "dp", "dg"]].to_numpy()
    if not np.isfinite(required).all() or not data.index.is_unique:
        raise ValueError("The EQ input must have valid, finite required variables and unique dates.")
    if not data.index.equals(pd.date_range(data.index[0], data.index[-1], freq="MS")):
        raise ValueError("The EQ input must form a complete monthly sequence.")
    data["R_w"] = np.exp(data["re"])
    data["R_f"] = np.exp(data["rf"])
    data["xRe"] = data["R_w"] - data["R_f"]
    forecasts = reproduce_forecasts(data)
    if reuse_saved_forecasts:
        saved = pd.read_csv(ROOT / "datasets/q2e_os_forecasts.csv", parse_dates=["forecast_date"])
        if not pd.DatetimeIndex(saved["forecast_date"]).equals(
            pd.DatetimeIndex(forecasts["forecast_date"])
        ) or not np.allclose(saved["OS_forecast"], forecasts["OS_forecast"], rtol=0, atol=1e-14):
            raise ValueError("Saved forecasts do not reproduce the specified Pset 1 construction.")
        forecasts["OS_forecast"] = saved["OS_forecast"].to_numpy()
    # forecast_date already names the predicted annual outcome's date.
    # Use the forecast for that outcome directly: do not shift a second time.
    sample = data.join(forecasts.set_index("forecast_date")[["OS_forecast"]],
                       how="inner", validate="one_to_one")
    # Revised prompt: estimate the variance only on the aligned Q3(c) sample.
    variance = sample["R_w"].var(ddof=0)
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("Estimation-sample gross-equity variance must be positive and finite.")
    sample["z_lagged"] = sample["OS_forecast"] / variance
    return data, forecasts, sample, variance


def main():
    data, forecasts, sample, variance = load_estimation_sample()
    result_dir = ROOT / "results"
    figure_dir = ROOT / "figures"
    result_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)
    forecasts[["forecast_date", "OS_forecast"]].to_csv(
        ROOT / "datasets/q2e_os_forecasts.csv", index=False, date_format="%Y-%m-%d",
    )
    forecasts.to_csv(result_dir / "q3c_forecast_alignment.csv", index=False, date_format="%Y-%m-%d")
    re = sample["re"].to_numpy()
    base_returns = np.column_stack([sample["xRe"], sample["z_lagged"] * sample["xRe"]])
    n = len(sample)

    def moments(gamma):
        return np.exp(-gamma * re)[:, None] * base_returns

    def objective(gamma):
        with np.errstate(over="ignore", invalid="ignore"):
            means = moments(gamma).mean(axis=0)
            value = np.sum(means ** 2)
        return float(value) if np.isfinite(value) else np.inf

    optimized = minimize_scalar(objective, method="brent", options={"xtol": 1e-13, "maxiter": 1000})
    if not optimized.success or not np.isfinite(optimized.fun):
        raise RuntimeError(f"GMM optimization failed: {optimized.message}")
    gamma = float(optimized.x)
    g = moments(gamma)
    errors = g.mean(axis=0)
    jacobian = np.mean(-re[:, None] * g, axis=0)
    gg = np.sum(jacobian ** 2)
    foc = np.sum(jacobian * errors)
    if gg < 1e-20 or abs(foc) > 1e-9:
        raise ValueError("GMM Jacobian or first-order-condition check failed.")
    hac, lag, bw = moment_hac(g)
    gamma_variance = np.sum(np.outer(jacobian, jacobian) * hac) / (n * gg ** 2)
    gamma_se = np.sqrt(gamma_variance)
    # Account for estimating gamma in the pricing-error covariance matrix.
    projection = np.eye(2) - np.outer(jacobian, jacobian) / gg
    error_covariance = projection @ hac @ projection.T / n
    error_covariance = (error_covariance + error_covariance.T) / 2
    if np.linalg.matrix_rank(error_covariance, tol=1e-12) != 1:
        raise ValueError("The pricing-error covariance should have rank one.")
    error_se = np.sqrt(np.diag(error_covariance))
    t_statistics = errors / error_se
    wald = float(errors @ np.linalg.pinv(error_covariance, hermitian=True, rcond=1e-10) @ errors)
    p_value = float(chi2.sf(wald, df=1))

    summary = pd.DataFrame({
        "Statistic": ["gamma_hat", "SE(gamma_hat)", "Pricing error g1", "t-statistic g1",
                      "Pricing error g2", "t-statistic g2", "Joint Wald chi-square",
                      "Joint p-value (df=1)", "T", "L"],
        "Value": [gamma, gamma_se, errors[0], t_statistics[0], errors[1], t_statistics[1],
                  wald, p_value, n, lag],
    })
    summary.to_csv(result_dir / "q3c_gmm_summary.csv", index=False, float_format="%.4f")
    summary.to_csv(result_dir / "q3c_gmm_summary_full_precision.csv", index=False)
    labels = ["g1", "g2"]
    pd.DataFrame(hac, index=labels, columns=labels).to_csv(result_dir / "q3c_hac_matrix.csv")
    pd.DataFrame(error_covariance, index=labels, columns=labels).to_csv(result_dir / "q3c_pricing_error_covariance.csv")
    pd.DataFrame({"Moment": labels, "Jacobian": jacobian, "Pricing error": errors,
                  "Pricing-error SE": error_se, "t-statistic": t_statistics}).to_csv(
        result_dir / "q3c_moment_statistics.csv", index=False,
    )
    sample.assign(g1=g[:, 0], g2=g[:, 1]).to_csv(
        result_dir / "q3c_estimation_sample.csv", date_format="%Y-%m-%d",
    )

    tex_labels = [r"Risk aversion $\hat\gamma$", r"Standard error $SE(\hat\gamma)$",
                  r"Pricing error $\bar g_1$", r"$t$-statistic for $\bar g_1$",
                  r"Pricing error $\bar g_2$", r"$t$-statistic for $\bar g_2$",
                  r"Joint Wald $\chi^2$ statistic", r"Joint-test $p$-value ($df=1$)",
                  r"Observations $T$", r"Common HAC lag $L$"]
    table = [r"\begin{table}[!htbp]", r"  \centering",
             r"  \caption{Identity-weighting two-moment GMM estimates, December 1940",
             r"  through December 2021. Results are rounded to four decimal places.}",
             r"  \label{tab:q3c-gmm-summary}", r"  \small", r"  \begin{tabular}{lr}",
             r"    \toprule", r"    Statistic & Value \\", r"    \midrule"]
    for i, (label, value) in enumerate(zip(tex_labels, summary["Value"])):
        formatted = str(int(value)) if i >= 8 else f"{value:.4f}"
        table.append(f"    {label} & {formatted}" + r" \\")
    table += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    (result_dir / "q3c_gmm_summary.tex").write_text("\n".join(table))

    details = "\n".join([
        r"The instrument uses the supplied Pset 1 Q2(e) expanding-window",
        r"forecasts, aligned to the annual return dated 12 months after the predictor.",
        r"No additional forecast shift is applied. The denominator is the Q3(c) estimation-sample",
        rf"gross-equity variance, $\hat\sigma_e^2={variance:.6f}$ (divisor $T={n}$),",
        r"using December 1940 through December 2021.",
        "",
        r"For common-lag selection, apply the NW94 Bartlett plug-in to",
        r"$u_t=(g_{1,t}-\bar g_1)+(g_{2,t}-\bar g_2)$, using equal weights $(1,1)$.",
        rf"The ceiling-rounded pilot is $m={bw['pilot']}$, the estimated constant is",
        rf"$C={bw['constant']:.4f}$, and $L=\lceil CT^{{1/3}}\rceil=\lceil {bw['bandwidth']:.4f}\rceil={lag}$.",
        r"Use this same $L$ for every entry of $\hat S$, with centered moment vectors,",
        r"autocovariance denominator $T-\ell$, and weights $1-\ell/(L+1)$.",
        r"There is no prewhitening or separate degrees-of-freedom adjustment.",
        r"\[",
        r"\hat S=\begin{pmatrix} " + f"{hac[0,0]:.6f} & {hac[0,1]:.6f}" + r" \\ "
        + f"{hac[1,0]:.6f} & {hac[1,1]:.6f}" + r" \end{pmatrix}.",
        r"\]",
        r"Pricing-error standard errors use $\hat V_\alpha=P\hat S P'/T$,",
        r"where $P=I-\hat G(\hat G'\hat G)^{-1}\hat G'$.",
        r"The joint test uses $\bar g'\hat V_\alpha^+\bar g$ and one degree of freedom.",
        "",
    ])
    (result_dir / "q3c_details.tex").write_text(details)

    grid = np.linspace(gamma - 2, gamma + 2, 501)
    grid_moments = np.array([moments(value).mean(axis=0) for value in grid])
    grid_objective = np.sum(grid_moments ** 2, axis=1)
    pd.DataFrame({"gamma": grid, "Objective": grid_objective, "g1 mean": grid_moments[:, 0],
                  "g2 mean": grid_moments[:, 1]}).to_csv(result_dir / "q3c_objective_grid.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8), layout="constrained")
    axes[0].plot(grid, grid_objective, color="#245b85", linewidth=1.8)
    axes[0].set_ylabel(r"GMM objective $\bar g_1^2+\bar g_2^2$")
    axes[0].set_ylim(bottom=0)
    axes[1].plot(grid, grid_moments[:, 0], label=r"$\bar g_1$", linewidth=1.7)
    axes[1].plot(grid, grid_moments[:, 1], label=r"$\bar g_2$", linewidth=1.7)
    axes[1].axhline(0, color="0.45", linewidth=0.8)
    axes[1].set_ylabel("Sample moment")
    for ax in axes:
        ax.axvline(gamma, color="tab:red", linestyle="--", linewidth=1.1,
                   label=rf"$\hat\gamma={gamma:.4f}$")
        ax.set_xlabel(r"Risk aversion $\gamma$")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(figure_dir / "q3c_gmm_objective.png", dpi=300)
    plt.close(fig)

    first = forecasts.iloc[0]
    diagnostics = (
        f"Full EQ sample: {data.index[0].date()} to {data.index[-1].date()}; T_full = {len(data)}\n"
        f"Estimation sample: {sample.index[0].date()} to {sample.index[-1].date()}; T = {n}\n"
        f"Estimation-sample gross-equity variance (ddof=0) = {variance:.15g}\n"
        "Variance sample: December 1940 through December 2021, after forecast alignment.\n"
        f"First predictor date: {first['predictor_date'].date()}\n"
        f"First growth training sample: {first['training_start'].date()} to "
        f"{first['training_end'].date()}; n = {first['training_observations']}\n"
        "Forecast target dates are aligned once; no additional forecast shift.\n"
        f"gamma_hat = {gamma:.15g}\nSE = {gamma_se:.15g}\nG = {jacobian}\n"
        f"Sample moments = {errors}\nG prime mean(g) = {foc:.15e}\n"
        f"Objective = {optimized.fun:.15g}\nPilot m = {bw['pilot']}\n"
        f"Common-lag weights = (1, 1); C = {bw['constant']:.15g}; "
        f"unrounded = {bw['bandwidth']:.15g}; L = {lag}\n"
        f"S_hat =\n{hac}\nPricing-error covariance =\n{error_covariance}\n"
        f"Wald chi-square(1) = {wald:.15g}; p-value = {p_value:.15g}\n"
    )
    (result_dir / "q3c_diagnostics.txt").write_text(diagnostics)
    print(diagnostics)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
