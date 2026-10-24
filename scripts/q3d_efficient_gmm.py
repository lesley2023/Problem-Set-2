"""Implement Prompts/Q3d_prompt.txt. Run: python3 scripts/q3d_efficient_gmm.py.

Reuse Q3(c)'s sample, verified saved forecasts, and common-lag HAC estimator.
The second-step weight is fixed at the inverse of the first-step HAC matrix.
"""

from q3c_gmm_managed_returns import ROOT, load_estimation_sample, moment_hac

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import chi2


def fit_gmm(re, base_returns, weight):
    """Use Q3(c)'s unconstrained Brent method, tolerances, and no bounds."""
    def moments(gamma):
        return np.exp(-gamma * re)[:, None] * base_returns

    def objective(gamma):
        with np.errstate(over="ignore", invalid="ignore"):
            means = moments(gamma).mean(axis=0)
            value = means @ weight @ means
        return float(value) if np.isfinite(value) else np.inf

    optimized = minimize_scalar(objective, method="brent", options={"xtol": 1e-13, "maxiter": 1000})
    if not optimized.success or not np.isfinite(optimized.fun):
        raise RuntimeError(f"GMM optimization failed: {optimized.message}")
    gamma = float(optimized.x)
    g = moments(gamma)
    errors = g.mean(axis=0)
    jacobian = np.mean(-re[:, None] * g, axis=0)
    curvature = float(jacobian @ weight @ jacobian)
    foc = float(jacobian @ weight @ errors)
    if not np.isfinite(curvature) or curvature < 1e-20 or abs(foc) > 1e-9:
        raise ValueError("GMM Jacobian or first-order-condition check failed.")
    return gamma, g, errors, jacobian, curvature, foc, objective


def tex_matrix(matrix):
    return (r"\begin{pmatrix} " + f"{matrix[0, 0]:.4f} & {matrix[0, 1]:.4f}"
            + r" \\ " + f"{matrix[1, 0]:.4f} & {matrix[1, 1]:.4f}"
            + r" \end{pmatrix}")


def main():
    _, forecasts, sample, variance = load_estimation_sample(reuse_saved_forecasts=True)
    re = sample["re"].to_numpy()
    base_returns = np.column_stack([sample["xRe"], sample["z_lagged"] * sample["xRe"]])
    n = len(sample)
    gamma1, g1, errors1, jacobian1, _, foc1, objective1 = fit_gmm(re, base_returns, np.eye(2))
    hac1, lag1, bw1 = moment_hac(g1)
    # Compute once. This same matrix is captured by the second-step objective.
    weight2 = np.linalg.inv(hac1)
    gamma2, g2, errors2, jacobian2, curvature, foc2, objective2 = fit_gmm(re, base_returns, weight2)
    hac2, lag2, bw2 = moment_hac(g2)
    weighted_jacobian = weight2 @ jacobian2
    gamma_variance = float(weighted_jacobian @ hac2 @ weighted_jacobian / (n * curvature ** 2))
    if not np.isfinite(gamma_variance) or gamma_variance <= 0:
        raise ValueError("The sandwich variance must be positive and finite.")
    gamma_se = np.sqrt(gamma_variance)
    projection = np.eye(2) - np.outer(jacobian2, jacobian2 @ weight2) / curvature
    error_covariance = projection @ hac2 @ projection.T / n
    error_covariance = (error_covariance + error_covariance.T) / 2
    if np.linalg.matrix_rank(error_covariance, tol=1e-12) != 1 or np.any(np.diag(error_covariance) <= 0):
        raise ValueError("The pricing-error covariance must have rank one and positive diagonals.")
    error_se = np.sqrt(np.diag(error_covariance))
    t_statistics = errors2 / error_se
    wald = float(errors2 @ np.linalg.pinv(error_covariance, hermitian=True, rcond=1e-10) @ errors2)
    p_value = float(chi2.sf(wald, df=1))

    result_dir = ROOT / "results"
    figure_dir = ROOT / "figures"
    result_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)
    summary = pd.DataFrame({
        "Statistic": ["gamma_hat_2", "SE(gamma_hat_2)", "Pricing error g1", "t-statistic g1",
                      "Pricing error g2", "t-statistic g2", "Joint Wald chi-square",
                      "Joint p-value (df=1)", "T", "L1", "L2", "gamma_hat_1"],
        "Value": [gamma2, gamma_se, errors2[0], t_statistics[0], errors2[1], t_statistics[1],
                  wald, p_value, n, lag1, lag2, gamma1],
    })
    summary.to_csv(result_dir / "q3d_gmm_summary.csv", index=False, float_format="%.4f")
    summary.to_csv(result_dir / "q3d_gmm_summary_full_precision.csv", index=False)
    labels = ["g1", "g2"]
    for filename, matrix in [("q3d_hac_step1.csv", hac1), ("q3d_weight_step2.csv", weight2),
                             ("q3d_hac_step2.csv", hac2),
                             ("q3d_pricing_error_covariance.csv", error_covariance),
                             ("q3d_pricing_error_projection.csv", projection)]:
        pd.DataFrame(matrix, index=labels, columns=labels).to_csv(result_dir / filename)
    pd.DataFrame({"Moment": labels, "Jacobian step 1": jacobian1, "Pricing error step 1": errors1,
                  "Jacobian step 2": jacobian2, "Pricing error step 2": errors2,
                  "Pricing-error SE step 2": error_se, "t-statistic step 2": t_statistics}).to_csv(
        result_dir / "q3d_moment_statistics.csv", index=False,
    )
    sample.assign(g1_step1=g1[:, 0], g2_step1=g1[:, 1], g1_step2=g2[:, 0], g2_step2=g2[:, 1]).to_csv(
        result_dir / "q3d_estimation_sample.csv", date_format="%Y-%m-%d",
    )
    pd.DataFrame([
        dict(step=step, gamma=gamma, lag=lag, **bw)
        for step, gamma, lag, bw in [(1, gamma1, lag1, bw1), (2, gamma2, lag2, bw2)]
    ]).to_csv(result_dir / "q3d_bandwidths.csv", index=False)

    tex_labels = [r"Risk aversion $\hat\gamma_2$", r"Standard error $SE(\hat\gamma_2)$",
                  r"Pricing error $\bar g_1(\hat\gamma_2)$", r"$t$-statistic for $\bar g_1$",
                  r"Pricing error $\bar g_2(\hat\gamma_2)$", r"$t$-statistic for $\bar g_2$",
                  r"Joint Wald $\chi^2$ statistic", r"Joint-test $p$-value ($df=1$)",
                  r"Observations $T$", r"First-step HAC lag $L_1$", r"Second-step HAC lag $L_2$",
                  r"First-step risk aversion $\hat\gamma_1$"]
    table = [r"\begin{table}[!htbp]", r"  \centering",
             r"  \caption{Two-step efficient GMM, December 1940 through December 2021.",
             r"  Results are rounded to four decimal places.}",
             r"  \label{tab:q3d-gmm-summary}", r"  \small", r"  \begin{tabular}{lr}",
             r"    \toprule", r"    Statistic & Value \\", r"    \midrule"]
    for i, (label, value) in enumerate(zip(tex_labels, summary["Value"])):
        formatted = str(int(value)) if 8 <= i <= 10 else f"{value:.4f}"
        table.append(f"    {label} & {formatted}" + r" \\")
    table += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    (result_dir / "q3d_gmm_summary.tex").write_text("\n".join(table))
    details = [
        r"The sample and forecast alignment are identical to Question 3(c).",
        rf"The constant instrument denominator is $\hat\sigma_e^2={variance:.8f}$ (divisor $T={n}$).",
        r"The second step holds $W_2=\hat S_1^{-1}$ fixed during optimization.",
        r"Both steps use the unconstrained Brent method, as in Question 3(c).",
        r"The common lag is selected separately at each estimate from the equal-weight sum",
        r"of the two centered moments, using the specified NW94 ceiling rules.",
        r"The centered Bartlett HAC autocovariances use divisor $T-\ell$.",
        r"\[",
        r"\hat S_1=" + tex_matrix(hac1) + r",\qquad \hat S_2=" + tex_matrix(hac2) + ".",
        r"\]",
        r"The reported standard error uses the two-step sandwich with $W_2$ and $\hat S_2$.",
        r"Pricing-error tests account for estimation of $\gamma$ using the specified",
        r"$\hat P_2\hat S_2\hat P_2'/T$ covariance and a one-degree-of-freedom joint Wald test.", "",
    ]
    (result_dir / "q3d_details.tex").write_text("\n".join(details))

    grid = np.linspace(min(gamma1, gamma2) - 2, max(gamma1, gamma2) + 2, 501)
    values1 = np.array([objective1(value) for value in grid])
    values2 = np.array([objective2(value) for value in grid])
    pd.DataFrame({"gamma": grid, "Step 1 objective": values1,
                  "Step 2 objective (fixed W2)": values2}).to_csv(
        result_dir / "q3d_objective_grid.csv", index=False,
    )
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8), layout="constrained")
    for ax, values, gamma, title, ylabel in [
        (axes[0], values1, gamma1, "Step 1: identity weighting", r"$\bar g'\bar g$"),
        (axes[1], values2, gamma2, "Step 2: fixed efficient weighting", r"$\bar g' W_2\bar g$"),
    ]:
        ax.plot(grid, values, color="#245b85", linewidth=1.8)
        ax.axvline(gamma, color="tab:red", linestyle="--", linewidth=1.1,
                   label=rf"$\hat\gamma={gamma:.4f}$")
        ax.scatter([gamma], [objective1(gamma) if ax is axes[0] else objective2(gamma)],
                   color="tab:red", s=18, zorder=3)
        ax.set(title=title, xlabel=r"Risk aversion $\gamma$", ylabel=ylabel, ylim=(0, None))
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(figure_dir / "q3d_gmm_objectives.png", dpi=300)
    plt.close(fig)

    first = forecasts.iloc[0]
    diagnostics = (
        f"Estimation sample: {sample.index[0].date()} to {sample.index[-1].date()}; T = {n}\n"
        f"Estimation-sample gross-equity variance (ddof=0): {variance:.15g}\n"
        "Saved forecasts verified against the Q3(c)/Pset 1 construction; no additional shift.\n"
        f"First predictor: {first['predictor_date'].date()}; first outcome: {first['forecast_date'].date()}\n"
        "Optimization: unconstrained Brent, xtol=1e-13, maxiter=1000, same as Q3(c).\n"
        f"Step 1 gamma: {gamma1:.15g}; FOC G1' gbar1: {foc1:.15e}\n"
        f"Step 1 bandwidth: {bw1}; L1 = {lag1}\nS1:\n{hac1}\n"
        f"W2 = inverse(S1), fixed throughout step 2:\n{weight2}\n"
        f"Step 2 gamma: {gamma2:.15g}; FOC G2' W2 gbar2: {foc2:.15e}\n"
        f"Step 2 bandwidth: {bw2}; L2 = {lag2}\nS2:\n{hac2}\n"
        f"G2: {jacobian2}\nGamma sandwich variance: {gamma_variance:.15g}\n"
        f"SE(gamma2): {gamma_se:.15g}\nPricing errors: {errors2}\n"
        f"Pricing-error covariance (rank one):\n{error_covariance}\n"
        f"Pricing-error t-statistics: {t_statistics}\n"
        f"Joint Wald chi-square(1): {wald:.15g}\np-value: {p_value:.15g}\n"
    )
    (result_dir / "q3d_diagnostics.txt").write_text(diagnostics)
    print(diagnostics)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
