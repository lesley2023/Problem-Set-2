"""Implement Prompts/Q3e_prompt.txt. Run: python3 scripts/q3e_21_moment_gmm.py.

Compound monthly nominal returns before subtracting the compounded risk-free
return and dividing by CPI inflation. Reuse Q3(c)'s dimension-independent HAC.
"""

from q3c_gmm_managed_returns import ROOT, moment_hac
from q2a_bm_sdf import BM_COLUMNS, read_monthly_block
from q2b_mom_sdf import MOM_COLUMNS

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import chi2


LABELS = ["Equity"] + [f"BM{j}" for j in range(1, 11)] + [f"MOM{j}" for j in range(1, 11)]


def load_annual_returns():
    """Monthly end-date alignment, rolling 12-month products, CPI_t/CPI_t-12."""
    # Retain the value-weighted deciles used elsewhere in this project.
    bm = read_monthly_block(ROOT / "datasets/BM_Dataset.csv", ",<= 0,Lo 30")[BM_COLUMNS]
    bm.columns = LABELS[1:11]
    mom = read_monthly_block(ROOT / "datasets/MOM_Dataset.csv", ",Lo PRIOR,PRIOR 2")[MOM_COLUMNS]
    mom.columns = LABELS[11:]
    rf = read_monthly_block(ROOT / "datasets/Fama_French_3_Factors.csv", ",Mkt-RF,SMB,HML,RF")[["RF"]]
    monthly = pd.concat([bm, mom, rf], axis=1).sort_index()
    monthly = monthly.reindex(pd.period_range(monthly.index.min(), monthly.index.max(), freq="M"))
    # The supplied files report percentages; missing sentinels are unavailable.
    monthly = monthly.replace([-99.99, -999], np.nan) / 100.0
    monthly = monthly.where(np.isfinite(monthly))
    if (monthly < -1).any().any():
        raise ValueError("A monthly simple return is below -100 percent.")
    annual_nominal = (1 + monthly).rolling(12, min_periods=12).apply(np.prod, raw=True)

    cpi = pd.read_csv(ROOT / "datasets/CPIAUCNS.csv", parse_dates=["observation_date"])
    cpi.index = pd.PeriodIndex(cpi["observation_date"], freq="M")
    if not cpi.index.is_unique:
        raise ValueError("CPI has duplicate monthly observations.")
    cpi = pd.to_numeric(cpi["CPIAUCNS"], errors="coerce").sort_index()
    cpi = cpi.reindex(pd.period_range(cpi.index.min(), cpi.index.max(), freq="M"))
    cpi = cpi.where(np.isfinite(cpi) & (cpi > 0))
    inflation = (cpi / cpi.shift(12)).rename("inflation_factor")
    annual_real_excess = annual_nominal.drop(columns="RF").sub(annual_nominal["RF"], axis=0).div(
        inflation, axis=0,
    )

    eq = pd.read_csv(ROOT / "datasets/EQ Dataset.csv")
    eq.index = pd.PeriodIndex(pd.to_datetime(dict(year=eq["YEAR"], month=eq["MONTH"], day=1)), freq="M")
    if not eq.index.is_unique:
        raise ValueError("EQ has duplicate monthly observations.")
    eq = eq.sort_index()
    eq["R_w"] = np.exp(eq["re"])
    eq["R_f_EQ"] = np.exp(eq["rf"])
    eq["Equity"] = eq["R_w"] - eq["R_f_EQ"]
    sample = eq[["re", "R_w", "R_f_EQ", "Equity"]].join(annual_real_excess, how="left")
    sample = sample.replace([np.inf, -np.inf], np.nan).dropna(subset=["R_w"] + LABELS)
    if len(sample) < 22 or not np.all(np.diff(sample.index.asi8) == 1):
        raise ValueError("Need at least 22 complete, consecutive monthly observations for HAC.")
    sample.index.name = "Month"
    audit = annual_nominal.add_prefix("nominal_gross_").join(inflation, how="left").loc[sample.index]
    audit["R_f_real_reconstructed"] = audit["nominal_gross_RF"] / audit["inflation_factor"]
    audit["R_f_EQ"] = sample["R_f_EQ"]
    audit["RF_difference"] = audit["R_f_real_reconstructed"] - audit["R_f_EQ"]
    return sample, audit, len(eq) - len(sample)


def main():
    sample, audit, excluded = load_annual_returns()
    re = sample["re"].to_numpy()
    excess = sample[LABELS].to_numpy()
    n, k = excess.shape

    def moments(gamma):
        return np.exp(-gamma * re)[:, None] * excess

    def objective(gamma):
        means = moments(gamma).mean(axis=0)
        return float(means @ means)

    optimized = minimize_scalar(objective, method="bounded", bounds=(0, 15),
                                options={"xatol": 1e-12, "maxiter": 1000})
    if not optimized.success or not np.isfinite(optimized.fun):
        raise RuntimeError(f"GMM optimization failed: {optimized.message}")
    gamma = float(optimized.x)
    # Explicitly evaluate both endpoints, which bounded minimization omits.
    candidates = [(gamma, objective(gamma)), (0.0, objective(0.0)), (15.0, objective(15.0))]
    gamma, minimum = min(candidates, key=lambda pair: pair[1])
    boundary = gamma <= 1e-6 or gamma >= 15 - 1e-6
    if boundary:
        raise ValueError(f"Boundary optimum at gamma={gamma}; interior inference needs review.")
    g = moments(gamma)
    errors = g.mean(axis=0)
    jacobian = (-re[:, None] * g).mean(axis=0)
    gg = float(jacobian @ jacobian)
    foc = float(jacobian @ errors)
    if not np.isfinite(gg) or gg < 1e-20 or abs(foc) > 1e-8:
        raise ValueError("GMM Jacobian or interior first-order-condition check failed.")
    hac, lag, bw = moment_hac(g)
    gamma_se = np.sqrt((jacobian @ hac @ jacobian) / (n * gg ** 2))
    projection = np.eye(k) - np.outer(jacobian, jacobian) / gg
    # Explicit contractions avoid spurious matrix-multiply warnings from the
    # local BLAS backend on these small matrices; the formulas are unchanged.
    error_covariance = np.einsum("ij,jk,lk->il", projection, hac, projection) / n
    error_covariance = (error_covariance + error_covariance.T) / 2
    rank = np.linalg.matrix_rank(error_covariance, tol=1e-12)
    if rank != k - 1 or np.any(np.diag(error_covariance) <= 0):
        raise ValueError("Pricing-error covariance must have rank 20 and positive diagonals.")
    error_se = np.sqrt(np.diag(error_covariance))
    t_statistics = errors / error_se
    eigenvalues, eigenvectors = np.linalg.eigh(error_covariance)
    keep = np.abs(eigenvalues) > 1e-10 * np.max(np.abs(eigenvalues))
    error_pseudoinverse = np.einsum("ij,j,kj->ik", eigenvectors[:, keep],
                                  1 / eigenvalues[keep], eigenvectors[:, keep])
    wald = float(errors @ error_pseudoinverse @ errors)
    p_value = float(chi2.sf(wald, df=k - 1))
    rf_difference = audit["RF_difference"]
    rf_metrics = {
        "Mean signed difference": rf_difference.mean(),
        "Mean absolute difference": rf_difference.abs().mean(),
        "Root mean squared difference": np.sqrt(np.mean(rf_difference ** 2)),
        "Maximum absolute difference": rf_difference.abs().max(),
        "Correlation": audit["R_f_real_reconstructed"].corr(audit["R_f_EQ"]),
    }

    result_dir = ROOT / "results"
    figure_dir = ROOT / "figures"
    result_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)
    summary = pd.DataFrame({
        "Statistic": ["gamma_hat", "SE(gamma_hat)", "Joint Wald chi-square", "Joint p-value (df=20)",
                      "T", "L", "Moment conditions", "Boundary optimum"],
        "Value": [gamma, gamma_se, wald, p_value, n, lag, k, int(boundary)],
    })
    summary.to_csv(result_dir / "q3e_gmm_summary.csv", index=False, float_format="%.4f")
    summary.to_csv(result_dir / "q3e_gmm_summary_full_precision.csv", index=False)
    moment_stats = pd.DataFrame({
        "Portfolio": LABELS, "Jacobian": jacobian, "Pricing error": errors,
        "Pricing-error SE": error_se, "t-statistic": t_statistics,
    })
    moment_stats.to_csv(result_dir / "q3e_moment_statistics.csv", index=False)
    moment_stats.to_csv(result_dir / "q3e_pricing_errors.csv", index=False, float_format="%.4f")
    for filename, matrix in [("q3e_hac_matrix.csv", hac),
                             ("q3e_pricing_error_covariance.csv", error_covariance),
                             ("q3e_pricing_error_projection.csv", projection)]:
        pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(result_dir / filename)
    sample.assign(**{f"g_{label}": g[:, j] for j, label in enumerate(LABELS)}).to_csv(
        result_dir / "q3e_estimation_sample.csv",
    )
    audit.to_csv(result_dir / "q3e_annual_return_audit.csv")
    audit[["R_f_real_reconstructed", "R_f_EQ", "RF_difference"]].to_csv(
        result_dir / "q3e_risk_free_comparison.csv",
    )
    pd.Series(rf_metrics, name="Value").rename_axis("Statistic").to_csv(
        result_dir / "q3e_risk_free_diagnostics.csv",
    )
    pd.DataFrame([dict(L=lag, **bw)]).to_csv(result_dir / "q3e_bandwidth.csv", index=False)

    table = [r"\begin{table}[!htbp]", r"  \centering", r"  \small",
             r"  \caption{Identity-weighting GMM with 21 moments. Annual real returns",
             f"  observed monthly, {sample.index[0].strftime('%B %Y')} through "
             f"{sample.index[-1].strftime('%B %Y')}. Results are rounded to four decimal places.",
             r"  BM and MOM numbers denote deciles from low to high.}",
             r"  \label{tab:q3e-gmm-results}", r"  \begin{tabular}{lrr}", r"    \toprule",
             r"    Statistic & \multicolumn{2}{r}{Value} \\", r"    \midrule"]
    tex_labels = [r"Risk aversion $\hat\gamma$", r"Standard error $SE(\hat\gamma)$",
                  r"Joint Wald $\chi^2(20)$ statistic", r"Joint-test $p$-value",
                  r"Observations $T$", r"Common HAC lag $L$"]
    for i, (label, value) in enumerate(zip(tex_labels, summary["Value"])):
        formatted = str(int(value)) if i >= 4 else f"{value:.4f}"
        table.append(f"    {label} & " + r"\multicolumn{2}{r}{" + formatted + r"} \\")
    table += [r"    \midrule", r"    Portfolio & Pricing error & $t$-statistic \\", r"    \midrule"]
    for j, label in enumerate(LABELS):
        if j in (1, 11):
            table.append(r"    \addlinespace")
        table.append(f"    {label} & {errors[j]:.4f} & {t_statistics[j]:.4f}" + r" \\")
    table += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    (result_dir / "q3e_gmm_results.tex").write_text("\n".join(table))

    details = [
        f"The common estimation sample contains {n:,} monthly observations from "
        f"{sample.index[0].strftime('%B %Y')} through {sample.index[-1].strftime('%B %Y')}.",
        r"Monthly value-weighted BM and MOM decile returns and Fama--French RF are",
        r"converted from percentages to decimals before compounding 12 monthly gross returns.",
        r"Each annual nominal excess return is divided by $CPI_t/CPI_{t-12}$ and",
        r"aligned with the EQ annual return ending in the same month.",
        rf"The bounded optimization on $[0,15]$ converged to an interior estimate; $\hat\gamma={gamma:.4f}$.",
        "",
        r"The common HAC lag uses the equal-weight sum of all 21 centered moments.",
        rf"The pilot truncation is $m={bw['pilot']}$, the NW94 constant is $C={bw['constant']:.4f}$,",
        rf"and $L=\lceil CT^{{1/3}}\rceil=\lceil {bw['bandwidth']:.4f}\rceil={lag}$.",
        r"Centered Bartlett autocovariances use divisor $T-\ell$.",
        r"The full $21\times21$ HAC matrix is saved in \texttt{results/q3e\_hac\_matrix.csv}.",
        r"The pricing-error covariance accounts for estimation of $\gamma$; the joint test uses 20 degrees of freedom.",
        rf"The unrounded joint-test $p$-value is approximately ${p_value / 10 ** np.floor(np.log10(p_value)):.3f}\times10^{{{int(np.floor(np.log10(p_value)))}}}$.",
        "",
        r"The reconstructed annual real risk-free gross return and $e^{rf_t}$ are compared on the same sample.",
        rf"Their mean signed difference (reconstructed minus EQ) is {rf_metrics['Mean signed difference']:.6f},",
        rf"mean absolute difference is {rf_metrics['Mean absolute difference']:.6f},",
        rf"root mean squared difference is {rf_metrics['Root mean squared difference']:.6f},",
        rf"and maximum absolute difference is {rf_metrics['Maximum absolute difference']:.6f}.",
        rf"Their correlation is {rf_metrics['Correlation']:.9f}.",
        r"No identity of risk-free benchmarks or inflation deflators is imposed.", "",
    ]
    (result_dir / "q3e_details.tex").write_text("\n".join(details))

    grid = np.linspace(0, 15, 601)
    values = np.array([objective(value) for value in grid])
    pd.DataFrame({"gamma": grid, "Objective": values}).to_csv(result_dir / "q3e_objective_grid.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained", gridspec_kw={"width_ratios": [1, 1.15]})
    axes[0].plot(grid, values, color="#245b85", linewidth=1.8)
    axes[0].axvline(gamma, color="tab:red", linestyle="--", linewidth=1.1,
                    label=rf"$\hat\gamma={gamma:.4f}$")
    axes[0].scatter([gamma], [minimum], color="tab:red", s=20, zorder=3)
    axes[0].set(xlabel=r"Risk aversion $\gamma$", ylabel=r"GMM objective $\bar g'\bar g$",
                title="Identity-weighting objective", xlim=(0, 15))
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=9)
    positions = np.arange(k)
    colors = ["#333333"] + ["#245b85"] * 10 + ["#be6a2e"] * 10
    axes[1].scatter(t_statistics, positions, c=colors, s=23, zorder=3)
    axes[1].axvline(0, color="0.45", linewidth=0.8)
    axes[1].set_yticks(positions, LABELS, fontsize=8)
    axes[1].set(xlabel="Pricing-error t-statistic", title="Adjusted pricing-error tests")
    axes[1].invert_yaxis()
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(figure_dir / "q3e_gmm_results.png", dpi=300)
    plt.close(fig)

    diagnostics = (
        f"Common sample: {sample.index[0]} through {sample.index[-1]}; T = {n}; moments = {k}\n"
        f"EQ dates excluded for unavailable complete inputs: {excluded}\n"
        "BM/MOM monthly value-weighted deciles; all monthly input returns divided by 100.\n"
        "Nominal annual gross returns: product of 12 monthly gross returns, ending-month alignment.\n"
        "Inflation: CPI_t / CPI_(t-12). Real portfolio excess: (nominal portfolio - nominal RF)/inflation.\n"
        f"Bounded optimizer: success={optimized.success}; bounds=[0,15]; boundary={boundary}\n"
        f"gamma_hat = {gamma:.15g}; SE = {gamma_se:.15g}; objective = {minimum:.15g}\n"
        f"Endpoint objectives: gamma=0: {objective(0):.15g}; gamma=15: {objective(15):.15g}\n"
        f"G' gbar first-order-condition residual: {foc:.15e}\n"
        f"Bandwidth details: {bw}; common lag = {lag}\n"
        f"Smallest HAC eigenvalue: {np.linalg.eigvalsh(hac).min():.15g}\n"
        f"Pricing-error covariance rank: {rank}; joint df = {k-1}\n"
        f"Joint Wald chi-square(20) = {wald:.15g}; p-value = {p_value:.15g}\n"
        "Risk-free differences are reconstructed annual real gross return minus exp(EQ rf).\n"
        + "\n".join(f"{name}: {value:.15g}" for name, value in rf_metrics.items()) + "\n"
    )
    (result_dir / "q3e_diagnostics.txt").write_text(diagnostics)
    print(diagnostics)
    print(moment_stats.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
