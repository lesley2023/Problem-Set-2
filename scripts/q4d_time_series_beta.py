"""Implement Prompts/Q4d_prompt.txt. Run: python3 scripts/q4d_time_series_beta.py.

Use gross R_w as explicitly requested. Keep its regression intercepts distinct
from SDF pricing errors, and conventional GRS separate from individual HAC tests.
"""

from q4b_linear_sdf_gmm import ROOT, LABELS, load_annual_returns, moment_hac, estimate_linear_sdf
from q3b_gmm_risk_aversion import newey_west_1994

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import betaln, hyp2f1
from scipy.stats import f, norm


def grs_tail(statistic, numerator_df, denominator_df):
    """Conventional F tail; retain a finite log-tail if float probability underflows."""
    probability = float(f.sf(statistic, numerator_df, denominator_df))
    log_probability = float(f.logsf(statistic, numerator_df, denominator_df))
    if np.isneginf(log_probability):
        # F survival = I_x(a,b); the incomplete-beta identity avoids x**a underflow.
        x = denominator_df / (denominator_df + numerator_df * statistic)
        a, b = denominator_df / 2, numerator_df / 2
        log_probability = float(a * np.log(x) - np.log(a) - betaln(a, b)
                                + np.log(hyp2f1(a, 1 - b, a + 1, x)))
    return probability, log_probability


def latex_log_probability(log_probability):
    exponent = int(np.floor(log_probability / np.log(10)))
    mantissa = np.exp(log_probability - exponent * np.log(10))
    return rf"{mantissa:.3f}\times10^{{{exponent}}}"


def estimate_time_series(sample):
    rw = sample["R_w"].to_numpy()
    excess = sample[LABELS].to_numpy()
    n, assets = excess.shape
    z = np.column_stack([np.ones(n), rw])
    coefficient = np.linalg.lstsq(z, excess, rcond=None)[0]
    fitted = np.einsum("ti,ij->tj", z, coefficient)
    residual = excess - fitted
    np.testing.assert_allclose(np.einsum("ti,tj->ij", z, residual) / n, 0, atol=1e-12, rtol=0)
    lambda_hat = float(rw.mean())
    lambda_hac, lambda_lag, lambda_bw = newey_west_1994(rw - lambda_hat)
    lambda_se = np.sqrt(lambda_hac / n)
    q = np.einsum("ti,tj->ij", z, z) / n
    q_inverse = np.linalg.inv(q)
    covariance, score_hac, lags, bandwidths = [], [], [], []
    for j, label in enumerate(LABELS):
        score = z * residual[:, j, None]
        hac, lag, bw = moment_hac(score)
        vcov = np.einsum("ij,jk,lk->il", q_inverse, hac, q_inverse) / n
        vcov = (vcov + vcov.T) / 2
        if not np.isfinite(vcov).all() or np.any(np.diag(vcov) <= 0):
            raise ValueError(f"Invalid coefficient covariance for {label}.")
        covariance.append(vcov)
        score_hac.append(hac)
        lags.append(lag)
        bandwidths.append(dict(Series=label, L=lag, **bw))
    covariance = np.array(covariance)
    alpha, beta = coefficient
    alpha_se = np.sqrt(covariance[:, 0, 0])
    alpha_t = alpha / alpha_se
    alpha_p = 2 * norm.sf(np.abs(alpha_t))
    residual_covariance = np.einsum("ti,tj->ij", residual, residual) / n
    rank = int(np.linalg.matrix_rank(residual_covariance))
    condition = float(np.linalg.cond(residual_covariance))
    if rank != assets or not np.isfinite(condition) or condition >= 1 / np.finfo(float).eps:
        raise ValueError("GRS residual covariance is singular or numerically noninvertible; specify any asset-set change.")
    if n - assets - 1 <= 0:
        raise ValueError("The GRS denominator degrees of freedom must be positive.")
    variance_rw = float(np.var(rw, ddof=0))
    if variance_rw <= 0:
        raise ValueError("The factor variance must be positive.")
    quadratic = float(alpha @ np.linalg.solve(residual_covariance, alpha))
    grs = float((n - assets - 1) / assets * quadratic / (1 + lambda_hat ** 2 / variance_rw))
    grs_p, grs_log_p = grs_tail(grs, assets, n - assets - 1)
    return dict(n=n, assets=assets, z=z, q=q, q_inverse=q_inverse, alpha=alpha, beta=beta,
                fitted=fitted, residual=residual, covariance=covariance, score_hac=score_hac,
                lags=lags, bandwidths=bandwidths, alpha_se=alpha_se, alpha_t=alpha_t, alpha_p=alpha_p,
                lambda_hat=lambda_hat, lambda_se=lambda_se, lambda_hac=lambda_hac,
                lambda_lag=lambda_lag, lambda_bw=lambda_bw, variance_rw=variance_rw,
                residual_covariance=residual_covariance, rank=rank, condition=condition,
                grs=grs, grs_p=grs_p, grs_log_p=grs_log_p, numerator_df=assets, denominator_df=n-assets-1)


def main():
    sample, _, _ = load_annual_returns()
    fit = estimate_time_series(sample)
    sdf_fit = estimate_linear_sdf(sample)
    lambda_sdf = sdf_fit["b"] * fit["variance_rw"]
    np.testing.assert_allclose(sample[LABELS].mean().to_numpy() - fit["beta"] * lambda_sdf,
                               sdf_fit["errors"], atol=1e-12, rtol=0)
    n = fit["n"]
    result_dir, figure_dir = ROOT / "results", ROOT / "figures"
    result_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)
    summary = pd.DataFrame({
        "Statistic": ["lambda_hat (mean gross R_w)", "SE(lambda_hat)", "Conventional GRS F", "GRS p-value",
                      "T", "Lambda HAC lag", "GRS numerator df", "GRS denominator df", "GRS log p-value",
                      "Residual covariance rank", "Residual covariance condition number", "Factor variance",
                      "lambda_SDF (Q4b)", "Q4b b_hat"],
        "Value": [fit["lambda_hat"], fit["lambda_se"], fit["grs"], fit["grs_p"], n, fit["lambda_lag"],
                  fit["numerator_df"], fit["denominator_df"], fit["grs_log_p"], fit["rank"], fit["condition"],
                  fit["variance_rw"], lambda_sdf, sdf_fit["b"]],
    })
    summary.to_csv(result_dir / "q4d_regression_summary.csv", index=False, float_format="%.4f")
    summary.to_csv(result_dir / "q4d_regression_summary_full_precision.csv", index=False)
    statistics = pd.DataFrame({
        "Portfolio": LABELS, "Alpha OLS": fit["alpha"], "SE(alpha)": fit["alpha_se"],
        "t(alpha)": fit["alpha_t"], "Normal p(alpha)": fit["alpha_p"], "Beta OLS": fit["beta"],
        "SE(beta)": np.sqrt(fit["covariance"][:, 1, 1]), "HAC lag": fit["lags"],
        "Q4b SDF pricing error": sdf_fit["errors"],
    })
    statistics.to_csv(result_dir / "q4d_portfolio_regressions_full_precision.csv", index=False)
    statistics.to_csv(result_dir / "q4d_portfolio_regressions.csv", index=False, float_format="%.4f")
    sample.to_csv(result_dir / "q4d_estimation_sample.csv")
    for filename, values in [("q4d_fitted_returns.csv", fit["fitted"]), ("q4d_ols_residuals.csv", fit["residual"])]:
        pd.DataFrame(values, index=sample.index, columns=LABELS).to_csv(result_dir / filename)
    pd.DataFrame(fit["residual_covariance"], index=LABELS, columns=LABELS).to_csv(
        result_dir / "q4d_grs_residual_covariance.csv",
    )
    pd.DataFrame(fit["q"], index=["Constant", "R_w"], columns=["Constant", "R_w"]).to_csv(result_dir / "q4d_ols_Q.csv")
    for filename, matrices in [("q4d_coefficient_covariances.csv", fit["covariance"]),
                               ("q4d_score_hac_matrices.csv", fit["score_hac"])]:
        rows = [dict(Portfolio=label, Row=row, Column=column, Value=matrices[j][i, ell])
                for j, label in enumerate(LABELS) for i, row in enumerate(["Alpha", "Beta"])
                for ell, column in enumerate(["Alpha", "Beta"])]
        pd.DataFrame(rows).to_csv(result_dir / filename, index=False)
    pd.DataFrame([dict(Series="Lambda", L=fit["lambda_lag"], **fit["lambda_bw"])] + fit["bandwidths"]).to_csv(
        result_dir / "q4d_bandwidths.csv", index=False,
    )

    table = [r"\begin{table}[!htbp]", r"  \centering", r"  \small",
             r"  \caption{Time-series regressions on gross $R_w$,",
             f"  {sample.index[0].strftime('%B %Y')} through {sample.index[-1].strftime('%B %Y')}.",
             r"  Values are rounded to four decimal places. Individual intercept tests use HAC;",
             r"  the GRS test uses its conventional $F$ reference.}",
             r"  \label{tab:q4d-regression-results}", r"  \begin{tabular}{lrrrrr}", r"    \toprule",
             r"    Statistic & \multicolumn{5}{r}{Value} \\", r"    \midrule"]
    upper = [(r"Requested $\hat\lambda=\bar R_w$", f"{fit['lambda_hat']:.4f}"),
             (r"Newey--West $SE(\hat\lambda)$", f"{fit['lambda_se']:.4f}"),
             (r"Conventional GRS $F$ statistic", f"{fit['grs']:.4f}"),
             (r"GRS degrees of freedom", f"({fit['numerator_df']}, {fit['denominator_df']})"),
             (r"GRS $p$-value", f"{fit['grs_p']:.4f}"),
             (r"Observations $T$", str(n)), (r"HAC lag for $\hat\lambda$", str(fit["lambda_lag"]))]
    for label, value in upper:
        table.append(f"    {label} & " + r"\multicolumn{5}{r}{" + value + r"} \\")
    table += [r"    \midrule", r"    Portfolio & $\hat\alpha$ & $SE(\hat\alpha)$ & $t_\alpha$ & $\hat\beta$ & $L_j$ \\",
              r"    \midrule"]
    for j, label in enumerate(LABELS):
        if j in (1, 11):
            table.append(r"    \addlinespace")
        table.append(f"    {label} & {fit['alpha'][j]:.4f} & {fit['alpha_se'][j]:.4f} & "
                     f"{fit['alpha_t'][j]:.4f} & {fit['beta'][j]:.4f} & {fit['lags'][j]}" + r" \\")
    table += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    (result_dir / "q4d_regression_results.tex").write_text("\n".join(table))

    significant = int(np.sum(fit["alpha_p"] < 0.05))
    bw = fit["lambda_bw"]
    details = [
        f"The sample contains {n:,} monthly observations of annual real returns from "
        f"{sample.index[0].strftime('%B %Y')} through {sample.index[-1].strftime('%B %Y')}, identical to Q4(b).",
        r"The requested constant-only regression uses gross $R_w$ and reports its sample mean as $\hat\lambda$.",
        rf"The lambda-series pilot is $m={bw['pilot']}$, $C={bw['constant']:.4f}$,",
        rf"and $L=\lceil CT^{{1/3}}\rceil=\lceil {bw['bandwidth']:.4f}\rceil={fit['lambda_lag']}$.",
        r"Each portfolio regression selects its own common lag from the sum of the two centered score components.",
        r"All HAC calculations retain Bartlett weights, $T-\ell$ divisors, and ceiling pilot/final rounding.",
        "",
        r"The regression intercepts and the normalized-SDF pricing errors have different definitions:",
        r"\[",
        r"\hat\alpha_j^{OLS}=\hat\mu_j-\hat\beta_j\bar R_w,\qquad",
        r"\hat e_j^{SDF}=\hat\mu_j-\hat\beta_j\hat\lambda_{SDF},\qquad",
        r"\hat\lambda_{SDF}=\hat b_{4b}\widehat{\operatorname{Var}}(R_w).",
        r"\]",
        rf"Here $\bar R_w={fit['lambda_hat']:.6f}$, whereas $\hat\lambda_{{SDF}}={lambda_sdf:.6f}$.",
        r"The reported alphas are the requested gross-factor OLS intercepts, not the Q4(b) SDF errors.",
        "",
        rf"At the 5\% level, {significant} of the 21 intercepts have two-sided asymptotic-normal HAC $p$-values below 0.05.",
        rf"The conventional GRS reference yields $p\approx {latex_log_probability(fit['grs_log_p'])}$",
        r"and rejects the joint zero-intercept null at that level.",
        rf"The residual covariance has full rank {fit['rank']} and condition number {fit['condition']:.2f};",
        r"all 21 assets are retained, and its ordinary inverse is used with divisor $T$.",
        r"The GRS $F$ calibration relies on the conventional GRS assumptions; it is separate from",
        r"the HAC corrections used for individual intercepts. Annual returns observed in overlapping monthly windows have serial dependence.", "",
    ]
    (result_dir / "q4d_details.tex").write_text("\n".join(details))

    positions = np.arange(fit["assets"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), layout="constrained")
    axes[0].errorbar(fit["alpha"], positions, xerr=fit["alpha_se"], fmt="o", markersize=3.5,
                    color="#245b85", ecolor="#8caabe", elinewidth=1.1, capsize=2)
    axes[0].set(title="OLS intercepts with one HAC SE", xlabel=r"Regression intercept $\hat\alpha_j$")
    axes[1].scatter(fit["alpha_t"], positions, c=["#333333"]+["#245b85"]*10+["#be6a2e"]*10, s=23)
    axes[1].axvline(0, color="0.45", linewidth=0.8)
    axes[1].set(title="HAC intercept t-statistics", xlabel=r"$t_{\alpha_j}$")
    for ax in axes:
        ax.set_yticks(positions, LABELS, fontsize=8)
        ax.invert_yaxis()
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(figure_dir / "q4d_regression_results.png", dpi=300)
    plt.close(fig)

    diagnostics = (
        f"Sample: {sample.index[0]} through {sample.index[-1]}; T={n}; assets=21; factor=gross R_w\n"
        f"Requested lambda=mean R_w={fit['lambda_hat']:.15g}; HAC SE={fit['lambda_se']:.15g}\n"
        f"Lambda HAC variance={fit['lambda_hac']:.15g}; lag={fit['lambda_lag']}; bandwidth={bw}\n"
        f"Maximum absolute sample OLS score mean={np.abs(np.einsum('ti,tj->ij',fit['z'],fit['residual'])/n).max():.15g}\n"
        f"Residual covariance: divisor T; rank={fit['rank']}; condition={fit['condition']:.15g}\n"
        f"Conventional GRS F={fit['grs']:.15g}; df=({fit['numerator_df']},{fit['denominator_df']})\n"
        f"GRS float p-value={fit['grs_p']:.15g}; natural-log p-value={fit['grs_log_p']:.15g}\n"
        "If the float p-value is zero, the finite log tail retains its magnitude.\n"
        f"Factor variance (ddof=0)={fit['variance_rw']:.15g}; Q4b b={sdf_fit['b']:.15g}; lambda_SDF={lambda_sdf:.15g}\n"
        f"Individual HAC normal p<0.05 count={significant} of 21.\n"
        "OLS alpha=mean excess-beta*mean gross R_w; SDF error=mean excess-beta*lambda_SDF.\n"
        "GRS uses the conventional contemporaneous residual covariance and F reference, not a HAC joint test.\n"
    )
    (result_dir / "q4d_diagnostics.txt").write_text(diagnostics)
    print(diagnostics)
    print(statistics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
