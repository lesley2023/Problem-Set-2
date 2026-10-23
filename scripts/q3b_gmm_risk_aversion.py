"""Implement Prompts/Q3b_prompt.txt.

Run: python3 scripts/q3b_gmm_risk_aversion.py

NW94 Bartlett plug-in (centered moments, no prewhitening/df adjustment):
https://raw.githubusercontent.com/cran/sandwich/master/R/vcovHAC.R
See bwNeweyWest() and NeweyWest(), with prewhite=False and adjust=False.
"""

from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "q3b-matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


ROOT = Path(__file__).resolve().parents[1]


def newey_west_1994(moment_series):
    """Centered Bartlett HAC with a data-dependent NW94 plug-in lag."""
    u = np.asarray(moment_series) - np.mean(moment_series)
    n = len(u)

    def autocovariance(lag):
        # Divisor T at every lag, not T - lag.
        return np.sum(u[lag:] * u[:n-lag]) / n

    # NW94 Table II C pilot truncation for Bartlett without prewhitening.
    pilot = min(int(np.floor(4 * (n / 100.0) ** (2 / 9))), n - 1)
    pilot_cov = np.array([autocovariance(lag) for lag in range(pilot + 1)])
    s0 = pilot_cov[0] + 2 * np.sum(pilot_cov[1:])
    s1 = 2 * np.sum(np.arange(1, pilot + 1) * pilot_cov[1:])
    if not np.isfinite(s0) or abs(s0) < 1e-14:
        raise ValueError("NW94 plug-in bandwidth is undefined: pilot s0 is zero or nonfinite.")
    constant = 1.1447 * ((s1 / s0) ** 2) ** (1 / 3)
    bandwidth = constant * n ** (1 / 3)
    lag = min(int(np.floor(bandwidth)), n - 1)
    covariance = np.array([autocovariance(j) for j in range(lag + 1)])
    weights = 1 - np.arange(1, lag + 1) / (lag + 1)
    long_run_variance = covariance[0] + 2 * np.sum(weights * covariance[1:])
    if not np.isfinite(long_run_variance) or long_run_variance <= 0:
        raise ValueError("The estimated long-run variance must be positive and finite.")
    details = dict(pilot=pilot, s0=s0, s1=s1, constant=constant, bandwidth=bandwidth)
    return long_run_variance, lag, details


def main():
    frame = pd.read_csv(ROOT / "datasets/EQ Dataset.csv")
    # Follow the prompt's valid-pair rule; keep monthly dates for time ordering.
    returns = frame[["re", "rf"]].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(returns.to_numpy()).all(axis=1)
    data = frame.loc[valid, ["YEAR", "MONTH"]].join(returns.loc[valid])
    data.index = pd.PeriodIndex(
        pd.to_datetime(dict(year=data["YEAR"], month=data["MONTH"], day=1)), freq="M",
    )
    data.index.name = "Month"
    data = data.sort_index()
    if len(data) < 3 or not data.index.is_unique:
        raise ValueError("Need at least three valid observations with unique monthly dates.")
    if not np.all(np.diff(data.index.asi8) == 1):
        raise ValueError("Valid observations have calendar gaps; specify HAC treatment of gaps.")
    re = data["re"].to_numpy()
    # The supplied logs are used as recorded, without dividing by 100 or 12.
    # The assignment describes these as annual returns observed monthly.
    rw = np.exp(re)
    rf = np.exp(data["rf"].to_numpy())
    excess = rw - rf
    if not np.isfinite(rw).all() or not np.isfinite(rf).all():
        raise ValueError("Nonfinite gross returns after exponentiation.")

    def moments(gamma):
        return np.exp(-gamma * re) * excess

    def objective(gamma):
        # This only guards numerical overflow; it imposes no parameter bounds.
        with np.errstate(over="ignore", invalid="ignore"):
            value = np.mean(moments(gamma)) ** 2
        return float(value) if np.isfinite(value) else np.inf

    result = minimize_scalar(objective, method="brent", options={"xtol": 1e-13, "maxiter": 1000})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"GMM optimization failed: {result.message}")
    gamma = float(result.x)
    g = moments(gamma)
    mean_g = g.mean()
    jacobian = np.mean(-re * g)
    foc = jacobian * mean_g
    if abs(foc) > 1e-10:
        raise ValueError("GMM first-order condition did not pass the numerical tolerance.")
    if not np.isfinite(jacobian) or abs(jacobian) < 1e-12:
        raise ValueError("The Jacobian is zero or too small for the requested SE formula.")
    long_run_variance, lag, bw = newey_west_1994(g)
    standard_error = np.sqrt(long_run_variance / (len(data) * jacobian ** 2))

    result_dir = ROOT / "results"
    figure_dir = ROOT / "figures"
    result_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)
    summary = pd.DataFrame({
        "Statistic": ["gamma_hat", "SE(gamma_hat)", "G_hat", "S_hat", "Sample moment", "T", "L"],
        "Value": [gamma, standard_error, jacobian, long_run_variance, mean_g, len(data), lag],
    })
    summary.to_csv(result_dir / "q3b_gmm_summary.csv", index=False, float_format="%.4f")
    summary.to_csv(result_dir / "q3b_gmm_summary_full_precision.csv", index=False)
    pd.DataFrame({
        "R_w": rw, "R_f": rf, "xR_w": excess, "g_t": g, "Centered g_t": g - mean_g,
    }, index=data.index).to_csv(result_dir / "q3b_moment_series.csv")

    labels = [
        r"Risk aversion $\hat\gamma$", r"Standard error $SE(\hat\gamma)$",
        r"Jacobian $\hat G$", r"Long-run variance $\hat S$",
        r"Sample moment $\bar g_T(\hat\gamma)$", r"Observations $T$", r"HAC lag $L$",
    ]
    table = [
        r"\begin{table}[!htbp]", r"  \centering",
        r"  \caption{GMM estimates using the supplied equity/risk-free log returns,",
        f"  {data.index[0].strftime('%B %Y')} through {data.index[-1].strftime('%B %Y')}. "
        r"Values are rounded to four decimal places.}",
        r"  \label{tab:q3b-gmm-summary}", r"  \small", r"  \begin{tabular}{lr}",
        r"    \toprule", r"    Statistic & Value \\", r"    \midrule",
    ]
    for i, (label, value) in enumerate(zip(labels, summary["Value"])):
        formatted = str(int(value)) if i >= 5 else f"{value:.4f}"
        table.append(f"    {label} & {formatted}" + r" \\")
    table += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    (result_dir / "q3b_gmm_summary.tex").write_text("\n".join(table))

    # This interval is for the figure only, not a restriction on optimization.
    gamma_grid = np.linspace(gamma - 2, gamma + 2, 501)
    mean_grid = np.array([moments(value).mean() for value in gamma_grid])
    pd.DataFrame({"gamma": gamma_grid, "Sample moment": mean_grid, "Objective": mean_grid ** 2}).to_csv(
        result_dir / "q3b_objective_grid.csv", index=False,
    )
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), layout="constrained")
    for ax, values, ylabel in zip(
        axes, [mean_grid ** 2, mean_grid],
        [r"GMM objective $[\bar g_T(\gamma)]^2$", r"Sample moment $\bar g_T(\gamma)$"],
    ):
        ax.plot(gamma_grid, values, color="#245b85", linewidth=1.8)
        ax.axvline(gamma, color="tab:red", linestyle="--", linewidth=1.1,
                   label=rf"$\hat\gamma={gamma:.4f}$")
        ax.set_xlabel(r"Risk aversion $\gamma$")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    axes[1].axhline(0, color="0.45", linewidth=0.8)
    axes[0].set_ylim(bottom=0)
    fig.savefig(figure_dir / "q3b_gmm_objective.png", dpi=300)
    plt.close(fig)

    methodology = "\n".join([
        r"For the centered moments $u_t=g_t(\hat\gamma)-\bar g_T(\hat\gamma)$,",
        r"define $\hat c_\ell=T^{-1}\sum_{t=\ell+1}^T u_tu_{t-\ell}$.",
        r"The Newey--West (1994) Bartlett plug-in procedure uses",
        r"\[",
        r"  m=\left\lfloor4(T/100)^{2/9}\right\rfloor,\quad",
        r"  s_0=\hat c_0+2\sum_{\ell=1}^{m}\hat c_\ell,\quad",
        r"  s_1=2\sum_{\ell=1}^{m}\ell\hat c_\ell,",
        r"\]",
        r"\[",
        r"  C=1.1447\left[(s_1/s_0)^2\right]^{1/3},\qquad",
        r"  L=\left\lfloor CT^{1/3}\right\rfloor.",
        r"\]",
        f"Here $m={bw['pilot']}$, $C={bw['constant']:.4f}$, and "
        f"$CT^{{1/3}}={bw['bandwidth']:.4f}$, giving $L={lag}$.",
        r"Thus the lag is data-dependent, with $m$ serving only as the pilot truncation.",
        r"With Bartlett weights $w_\ell=1-\ell/(L+1)$,",
        r"$\hat S=\hat c_0+2\sum_{\ell=1}^{L}w_\ell\hat c_\ell$.",
        r"The calculation uses no prewhitening or degrees-of-freedom adjustment,",
        r"matching the centered-moment HAC requested in the prompt. The plug-in",
        r"implementation follows \href{https://zeileis.codeberg.page/sandwich/reference/NeweyWest.html}{the documented Newey--West procedure}.",
        "",
    ])
    (result_dir / "q3b_bandwidth.tex").write_text(methodology)
    diagnostics = (
        f"Sample: {data.index[0]} to {data.index[-1]}; T = {len(data)}\n"
        f"Invalid re/rf pairs removed: {len(frame) - len(data)}\n"
        "Logs used as supplied; annual return observations sampled monthly, per assignment.\n"
        "Optimizer: scipy.optimize.minimize_scalar, unconstrained Brent; weighting scalar 1.\n"
        f"gamma_hat = {gamma:.15g}\nSE = {standard_error:.15g}\n"
        f"Sample moment = {mean_g:.15e}\nJacobian G_hat = {jacobian:.15e}\n"
        f"First-order condition G_hat * mean_g = {foc:.15e}\nObjective = {result.fun:.15e}\n"
        f"Pilot m = {bw['pilot']}\ns0 = {bw['s0']:.15g}\ns1 = {bw['s1']:.15g}\n"
        f"NW94 constant C = {bw['constant']:.15g}\nUnrounded lag = {bw['bandwidth']:.15g}\n"
        f"Selected L = {lag}\nS_hat = {long_run_variance:.15g}\n"
        "HAC uses centered moments, divisor T at each lag, no prewhitening or df adjustment.\n"
        "Bandwidth reference: https://raw.githubusercontent.com/cran/sandwich/master/R/vcovHAC.R\n"
    )
    (result_dir / "q3b_diagnostics.txt").write_text(diagnostics)
    print(diagnostics)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
