"""Implement Prompts/Q4e_prompt.txt. Run: python3 scripts/q4e_fama_macbeth.py.

Reuse Q4(d)'s full-sample betas and common annual real-return panel. Each
monthly second pass is the specified one-regressor cross-section, no intercept.
"""

from q4d_time_series_beta import ROOT, LABELS, load_annual_returns, estimate_time_series, newey_west_1994

import numpy as np
import pandas as pd


def estimate_fama_macbeth(sample, beta):
    excess = sample[LABELS].to_numpy()
    beta = np.asarray(beta)
    if beta.shape != (len(LABELS),) or not np.isfinite(beta).all():
        raise ValueError("Need one finite fixed beta for each of the 21 portfolios.")
    beta_norm = float(beta @ beta)
    if beta_norm < 1e-20:
        raise ValueError("The cross-sectional regressor has zero norm.")
    lambda_series = np.einsum("j,tj->t", beta, excess) / beta_norm
    fitted = lambda_series[:, None] * beta[None, :]
    alpha_series = excess - fitted
    foc = np.einsum("j,tj->t", beta, alpha_series)
    np.testing.assert_allclose(foc, 0, atol=1e-12, rtol=0)
    lambda_hat = float(lambda_series.mean())
    alpha = alpha_series.mean(axis=0)
    np.testing.assert_allclose(alpha, excess.mean(axis=0) - lambda_hat * beta, atol=1e-12, rtol=0)
    np.testing.assert_allclose(beta @ alpha, 0, atol=1e-12, rtol=0)
    n = len(sample)
    lambda_hac, lambda_lag, lambda_bw = newey_west_1994(lambda_series - lambda_hat)
    lambda_se = np.sqrt(lambda_hac / n)
    alpha_hac, alpha_se, alpha_lag, bandwidths = [], [], [], []
    for j, label in enumerate(LABELS):
        hac, lag, bw = newey_west_1994(alpha_series[:, j] - alpha[j])
        alpha_hac.append(hac)
        alpha_se.append(np.sqrt(hac / n))
        alpha_lag.append(lag)
        bandwidths.append(dict(Series=label, L=lag, **bw))
    alpha_se = np.asarray(alpha_se)
    return dict(n=n, beta=beta, beta_norm=beta_norm, lambda_series=lambda_series,
                lambda_hat=lambda_hat, lambda_se=lambda_se, lambda_hac=lambda_hac,
                lambda_lag=lambda_lag, lambda_bw=lambda_bw, fitted=fitted,
                alpha_series=alpha_series, alpha=alpha, alpha_se=alpha_se,
                alpha_t=alpha / alpha_se, alpha_lag=alpha_lag, alpha_hac=alpha_hac,
                bandwidths=bandwidths, foc=foc)


def main():
    sample, _, _ = load_annual_returns()
    prior_sample = pd.read_csv(ROOT / "results/q4d_estimation_sample.csv", index_col=0)
    prior_sample.index = pd.PeriodIndex(prior_sample.index, freq="M", name="Month")
    if not sample.index.equals(prior_sample.index):
        raise ValueError("The estimation dates must match Q4(d) exactly.")
    np.testing.assert_allclose(sample[["R_w"] + LABELS], prior_sample[["R_w"] + LABELS], atol=1e-14, rtol=0)
    prior = pd.read_csv(ROOT / "results/q4d_portfolio_regressions_full_precision.csv").set_index("Portfolio")
    if list(prior.index) != LABELS:
        raise ValueError("Saved Q4(d) portfolio order does not match the return panel.")
    beta = prior["Beta OLS"].to_numpy()
    # Reproduce the original first pass for validation; use its saved full-precision betas.
    reproduced = estimate_time_series(sample)
    np.testing.assert_allclose(beta, reproduced["beta"], atol=1e-12, rtol=0)
    fit = estimate_fama_macbeth(sample, beta)
    n = fit["n"]
    result_dir = ROOT / "results"
    result_dir.mkdir(exist_ok=True)
    summary = pd.DataFrame({
        "Statistic": ["lambda_hat", "SE(lambda_hat)", "T", "Lambda HAC lag", "Assets", "Maximum absolute monthly FOC"],
        "Value": [fit["lambda_hat"], fit["lambda_se"], n, fit["lambda_lag"], len(LABELS), np.abs(fit["foc"]).max()],
    })
    summary.to_csv(result_dir / "q4e_fama_macbeth_summary.csv", index=False, float_format="%.4f")
    summary.to_csv(result_dir / "q4e_fama_macbeth_summary_full_precision.csv", index=False)
    statistics = pd.DataFrame({
        "Portfolio": LABELS, "Beta (Q4d)": beta, "Mean alpha FMB": fit["alpha"],
        "SE(mean alpha)": fit["alpha_se"], "t(alpha)": fit["alpha_t"],
        "Alpha HAC lag": fit["alpha_lag"], "Alpha long-run variance": fit["alpha_hac"],
    })
    statistics.to_csv(result_dir / "q4e_portfolio_results_full_precision.csv", index=False)
    statistics.to_csv(result_dir / "q4e_portfolio_results.csv", index=False, float_format="%.4f")
    pd.DataFrame({"lambda_t": fit["lambda_series"], "Cross-sectional FOC": fit["foc"]}, index=sample.index).to_csv(
        result_dir / "q4e_monthly_risk_premia.csv",
    )
    pd.DataFrame(fit["alpha_series"], index=sample.index, columns=LABELS).to_csv(
        result_dir / "q4e_monthly_pricing_errors.csv",
    )
    pd.DataFrame(fit["fitted"], index=sample.index, columns=LABELS).to_csv(result_dir / "q4e_monthly_fitted_returns.csv")
    pd.DataFrame({"Portfolio": LABELS, "Reused Q4d beta": beta, "Reproduced beta": reproduced["beta"],
                  "Difference": beta - reproduced["beta"]}).to_csv(result_dir / "q4e_beta_verification.csv", index=False)
    sample.to_csv(result_dir / "q4e_estimation_sample.csv")
    pd.DataFrame([dict(Series="Lambda", L=fit["lambda_lag"], **fit["lambda_bw"])] + fit["bandwidths"]).to_csv(
        result_dir / "q4e_bandwidths.csv", index=False,
    )

    table = [r"\begin{table}[!htbp]", r"  \centering", r"  \small",
             r"  \caption{Fama--MacBeth regressions with fixed full-sample Q4(d) betas,",
             f"  {sample.index[0].strftime('%B %Y')} through {sample.index[-1].strftime('%B %Y')}.",
             r"  Cross-sectional regressions have no intercept. Values are rounded to four decimal places;",
             r"  standard errors use Newey--West HAC on the premium and pricing-error series.}",
             r"  \label{tab:q4e-fama-macbeth-results}", r"  \begin{tabular}{lrrrrr}", r"    \toprule",
             r"    Statistic & \multicolumn{5}{r}{Value} \\", r"    \midrule"]
    upper = [(r"Fama--MacBeth $\hat\lambda$", f"{fit['lambda_hat']:.4f}"),
             (r"Newey--West $SE(\hat\lambda)$", f"{fit['lambda_se']:.4f}"),
             (r"Observations $T$", str(n)), (r"HAC lag for $\hat\lambda$", str(fit["lambda_lag"]))]
    for label, value in upper:
        table.append(f"    {label} & " + r"\multicolumn{5}{r}{" + value + r"} \\")
    table += [r"    \midrule", r"    Portfolio & $\hat\alpha$ & $SE(\hat\alpha)$ & $t_\alpha$ & $\hat\beta$ & $L_j$ \\",
              r"    \midrule"]
    for j, label in enumerate(LABELS):
        if j in (1, 11):
            table.append(r"    \addlinespace")
        table.append(f"    {label} & {fit['alpha'][j]:.4f} & {fit['alpha_se'][j]:.4f} & "
                     f"{fit['alpha_t'][j]:.4f} & {beta[j]:.4f} & {fit['alpha_lag'][j]}" + r" \\")
    table += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    (result_dir / "q4e_fama_macbeth_results.tex").write_text("\n".join(table))
    foc_mantissa, foc_exponent = f"{np.abs(fit['foc']).max():.3e}".split("e")
    details = [
        f"The sample contains {n:,} monthly observations of annual real returns from "
        f"{sample.index[0].strftime('%B %Y')} through {sample.index[-1].strftime('%B %Y')}, identical to Q4(d).",
        r"All 21 fixed full-sample betas are reused from Q4(d) and verified against its original first-pass code.",
        r"Each monthly cross-section uses one beta regressor and no intercept.",
        r"The pricing errors are averages of the monthly cross-sectional residuals.",
        rf"The maximum absolute monthly OLS first-order-condition residual is ${foc_mantissa}\times10^{{{int(foc_exponent)}}}$.",
        r"Standard errors follow the specified constant-only time-series HAC calculation, with fixed estimated betas.",
        rf"The premium-series pilot is $m={fit['lambda_bw']['pilot']}$, $C={fit['lambda_bw']['constant']:.4f}$,",
        rf"and $L=\lceil CT^{{1/3}}\rceil=\lceil {fit['lambda_bw']['bandwidth']:.4f}\rceil={fit['lambda_lag']}$.",
        r"Each pricing-error series selects its own lag, reported in the table.",
        r"The monthly premium and all 21 monthly pricing-error series are saved in",
        r"\texttt{results/q4e\_monthly\_risk\_premia.csv} and",
        r"\texttt{results/q4e\_monthly\_pricing\_errors.csv}, respectively.", "",
    ]
    (result_dir / "q4e_details.tex").write_text("\n".join(details))
    diagnostics = (
        f"Sample: {sample.index[0]} through {sample.index[-1]}; T={n}; assets=21\n"
        "Full-sample fixed betas reused from Q4(d), verified within 1e-12. Factor=gross R_w.\n"
        "Monthly cross-sectional OLS: beta regressor only; no intercept; no rolling window.\n"
        f"Maximum absolute monthly beta'alpha_t={np.abs(fit['foc']).max():.15e}\n"
        f"lambda_hat={fit['lambda_hat']:.15g}; HAC SE={fit['lambda_se']:.15g}\n"
        f"Lambda HAC variance={fit['lambda_hac']:.15g}; L={fit['lambda_lag']}; bandwidth={fit['lambda_bw']}\n"
        f"Mean beta'alpha={beta @ fit['alpha']:.15e}\n"
        "HAC for each constant-only series: centered residuals, Bartlett, T-lag divisor, ceiling pilot/final lags.\n"
        "Inference follows the requested mean-series HAC calculation with fixed estimated betas.\n"
    )
    (result_dir / "q4e_diagnostics.txt").write_text(diagnostics)
    print(diagnostics)
    print(statistics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
