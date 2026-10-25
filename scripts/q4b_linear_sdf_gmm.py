"""Implement Prompts/Q4b_prompt.txt. Run: python3 scripts/q4b_linear_sdf_gmm.py.

Reuse Q3(e)'s annual real-return panel and HAC routine. The HAC input is the
specified moment series adjusted for estimation of the wealth-return mean.
"""

from q3e_21_moment_gmm import ROOT, LABELS, load_annual_returns, moment_hac

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2


def estimate_linear_sdf(sample):
    rw = sample["R_w"].to_numpy()
    excess = sample[LABELS].to_numpy()
    n, k = excess.shape
    mean_rw = rw.mean()
    centered_rw = rw - mean_rw
    mu = excess.mean(axis=0)
    c = (centered_rw[:, None] * excess).mean(axis=0)
    cc = float(c @ c)
    if not np.isfinite(cc) or cc < 1e-20:
        raise ValueError("The covariance vector is zero or nonfinite.")
    b = float(c @ mu / cc)
    a = 1 + b * mean_rw
    sdf = 1 - b * centered_rw
    np.testing.assert_allclose(sdf.mean(), 1, atol=1e-12, rtol=0)
    raw_moments = sdf[:, None] * excess
    errors = raw_moments.mean(axis=0)
    np.testing.assert_allclose(errors, mu - b * c, atol=1e-12, rtol=0)
    jacobian = -c
    foc = float(jacobian @ errors)
    if abs(foc) > 1e-12:
        raise ValueError("The analytical estimator does not satisfy its first-order condition.")
    # Account for the estimated sample mean R_w as explicitly specified.
    adjusted_moments = raw_moments + b * centered_rw[:, None] * mu[None, :]
    np.testing.assert_allclose(adjusted_moments.mean(axis=0), errors, atol=1e-12, rtol=0)
    hac, lag, bw = moment_hac(adjusted_moments)
    b_variance = float((jacobian @ hac @ jacobian) / (n * cc ** 2))
    if not np.isfinite(b_variance) or b_variance <= 0:
        raise ValueError("The coefficient sandwich variance must be positive and finite.")
    b_se = np.sqrt(b_variance)
    projection = np.eye(k) - np.outer(jacobian, jacobian) / cc
    # Explicit contractions avoid the local BLAS warning issue seen in Q3(e).
    error_covariance = np.einsum("ij,jk,lk->il", projection, hac, projection) / n
    error_covariance = (error_covariance + error_covariance.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(error_covariance)
    keep = np.abs(eigenvalues) > 1e-10 * np.max(np.abs(eigenvalues))
    if keep.sum() != k - 1 or np.any(np.diag(error_covariance) <= 0):
        raise ValueError("The pricing-error covariance must have rank 20 and positive diagonals.")
    error_se = np.sqrt(np.diag(error_covariance))
    t_statistics = errors / error_se
    error_pseudoinverse = np.einsum("ij,j,kj->ik", eigenvectors[:, keep],
                                  1 / eigenvalues[keep], eigenvectors[:, keep])
    wald = float(errors @ error_pseudoinverse @ errors)
    p_value = float(chi2.sf(wald, df=k - 1))
    return dict(b=b, a=a, b_se=b_se, mean_rw=mean_rw, mu=mu, c=c, sdf=sdf,
                raw_moments=raw_moments, adjusted_moments=adjusted_moments,
                errors=errors, jacobian=jacobian, hac=hac, lag=lag, bandwidth=bw,
                projection=projection, error_covariance=error_covariance,
                error_se=error_se, t_statistics=t_statistics, wald=wald,
                p_value=p_value, foc=foc, rank=int(keep.sum()), n=n, k=k)


def main():
    sample, audit, excluded = load_annual_returns()
    fit = estimate_linear_sdf(sample)
    b, b_se, n, lag = fit["b"], fit["b_se"], fit["n"], fit["lag"]
    bw = fit["bandwidth"]
    result_dir = ROOT / "results"
    figure_dir = ROOT / "figures"
    result_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)
    summary = pd.DataFrame({
        "Statistic": ["b_hat", "SE(b_hat)", "Joint Wald chi-square", "Joint p-value (df=20)",
                      "T", "L", "Moment conditions", "Mean R_w", "Mean SDF", "a_hat"],
        "Value": [b, b_se, fit["wald"], fit["p_value"], n, lag, fit["k"],
                  fit["mean_rw"], fit["sdf"].mean(), fit["a"]],
    })
    summary.to_csv(result_dir / "q4b_gmm_summary.csv", index=False, float_format="%.4f")
    summary.to_csv(result_dir / "q4b_gmm_summary_full_precision.csv", index=False)
    moment_stats = pd.DataFrame({
        "Portfolio": LABELS, "Mean excess return": fit["mu"], "Covariance with R_w": fit["c"],
        "Jacobian": fit["jacobian"], "Pricing error": fit["errors"],
        "Pricing-error SE": fit["error_se"], "t-statistic": fit["t_statistics"],
    })
    moment_stats.to_csv(result_dir / "q4b_moment_statistics.csv", index=False)
    moment_stats.to_csv(result_dir / "q4b_pricing_errors.csv", index=False, float_format="%.4f")
    for filename, matrix in [("q4b_hac_matrix.csv", fit["hac"]),
                             ("q4b_pricing_error_covariance.csv", fit["error_covariance"]),
                             ("q4b_pricing_error_projection.csv", fit["projection"])]:
        pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(result_dir / filename)
    sample.assign(M_hat=fit["sdf"]).to_csv(result_dir / "q4b_estimation_sample.csv")
    pd.DataFrame(fit["raw_moments"], index=sample.index, columns=LABELS).to_csv(
        result_dir / "q4b_raw_moments.csv",
    )
    pd.DataFrame(fit["adjusted_moments"], index=sample.index, columns=LABELS).to_csv(
        result_dir / "q4b_adjusted_moments.csv",
    )
    pd.DataFrame([dict(L=lag, **bw)]).to_csv(result_dir / "q4b_bandwidth.csv", index=False)
    audit.to_csv(result_dir / "q4b_annual_return_audit.csv")
    audit[["R_f_real_reconstructed", "R_f_EQ", "RF_difference"]].to_csv(
        result_dir / "q4b_risk_free_comparison.csv",
    )
    difference = audit["RF_difference"]
    rf_metrics = {
        "Mean signed difference": difference.mean(),
        "Mean absolute difference": difference.abs().mean(),
        "Root mean squared difference": np.sqrt(np.mean(difference ** 2)),
        "Maximum absolute difference": difference.abs().max(),
        "Correlation": audit["R_f_real_reconstructed"].corr(audit["R_f_EQ"]),
    }
    pd.Series(rf_metrics, name="Value").rename_axis("Statistic").to_csv(
        result_dir / "q4b_risk_free_diagnostics.csv",
    )

    table = [r"\begin{table}[!htbp]", r"  \centering", r"  \small",
             r"  \caption{Identity-weighting GMM for the normalized linear SDF,",
             f"  {sample.index[0].strftime('%B %Y')} through {sample.index[-1].strftime('%B %Y')}.",
             r"  Results are rounded to four decimal places; standard errors account for",
             r"  estimation of the mean wealth return. BM and MOM numbers denote deciles from low to high.}",
             r"  \label{tab:q4b-gmm-results}", r"  \begin{tabular}{lrr}", r"    \toprule",
             r"    Statistic & \multicolumn{2}{r}{Value} \\", r"    \midrule"]
    tex_labels = [r"SDF slope $\hat b$", r"Standard error $SE(\hat b)$",
                  r"Joint Wald $\chi^2(20)$ statistic", r"Joint-test $p$-value",
                  r"Observations $T$", r"Common HAC lag $L$"]
    for i, (label, value) in enumerate(zip(tex_labels, summary["Value"])):
        formatted = str(int(value)) if i >= 4 else f"{value:.4f}"
        table.append(f"    {label} & " + r"\multicolumn{2}{r}{" + formatted + r"} \\")
    table += [r"    \midrule", r"    Portfolio & Pricing error & $t$-statistic \\", r"    \midrule"]
    for j, label in enumerate(LABELS):
        if j in (1, 11):
            table.append(r"    \addlinespace")
        table.append(f"    {label} & {fit['errors'][j]:.4f} & {fit['t_statistics'][j]:.4f}" + r" \\")
    table += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    (result_dir / "q4b_gmm_results.tex").write_text("\n".join(table))

    p_mantissa, p_exponent = f"{fit['p_value']:.3e}".split("e")
    details = [
        f"The estimation uses {n:,} monthly observations of annual real returns, from "
        f"{sample.index[0].strftime('%B %Y')} through {sample.index[-1].strftime('%B %Y')}.",
        r"The return construction and ending-month alignment are identical to Question 3(e).",
        rf"The sample mean wealth return is $\bar R_w={fit['mean_rw']:.6f}$.",
        rf"The fitted SDF is $\hat M_t=1-{b:.6f}(R_{{w,t}}-\bar R_w)$,",
        rf"equivalently $\hat M_t={fit['a']:.6f}-{b:.6f}R_{{w,t}}$; its unrounded sample mean is 1.",
        "",
        r"The coefficient is calculated analytically with the identity weighting matrix.",
        r"HAC estimation uses the specified adjusted series",
        r"$\tilde g_t=g_t(\hat b)+\hat b\hat\mu(R_{w,t}-\bar R_w)$ to account for estimation of $\bar R_w$.",
        r"The common lag is selected from the sum of all 21 centered adjusted series.",
        rf"The pilot is $m={bw['pilot']}$, the NW94 constant is $C={bw['constant']:.4f}$,",
        rf"and $L=\lceil CT^{{1/3}}\rceil=\lceil {bw['bandwidth']:.4f}\rceil={lag}$.",
        r"Bartlett autocovariances use divisor $T-\ell$.",
        r"The full HAC matrix is saved in \texttt{results/q4b\_hac\_matrix.csv}.",
        r"Pricing-error tests account for estimation of both $b$ and $E[R_w]$; the joint test uses 20 degrees of freedom.",
        rf"The unrounded joint-test $p$-value is approximately ${p_mantissa}\times10^{{{int(p_exponent)}}}$.",
        "",
        r"For reconstructed annual real RF minus $e^{rf_t}$, the mean signed difference is",
        f"{rf_metrics['Mean signed difference']:.6f}, the mean absolute difference is "
        f"{rf_metrics['Mean absolute difference']:.6f}, the RMSE is {rf_metrics['Root mean squared difference']:.6f},",
        f"and the maximum absolute difference is {rf_metrics['Maximum absolute difference']:.6f}.",
        f"Their correlation is {rf_metrics['Correlation']:.9f}.",
        r"No identity of the risk-free benchmarks or inflation deflators is assumed.", "",
    ]
    (result_dir / "q4b_details.tex").write_text("\n".join(details))

    span = max(abs(b), 1.0)
    grid = np.linspace(b - span, b + span, 501)
    mean_grid = fit["mu"][None, :] - grid[:, None] * fit["c"][None, :]
    objectives = np.sum(mean_grid ** 2, axis=1)
    pd.DataFrame({"b": grid, "Objective": objectives}).to_csv(result_dir / "q4b_objective_grid.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1.15]})
    axes[0].plot(grid, objectives, color="#245b85", linewidth=1.8)
    axes[0].axvline(b, color="tab:red", linestyle="--", linewidth=1.1,
                    label=rf"$\hat b={b:.4f}$")
    axes[0].scatter([b], [fit["errors"] @ fit["errors"]], color="tab:red", s=20, zorder=3)
    axes[0].set(xlabel=r"SDF slope $b$", ylabel=r"GMM objective $\bar g'\bar g$",
                title="Identity-weighting objective", ylim=(0, None))
    axes[0].legend(frameon=False, fontsize=9)
    colors = ["#333333"] + ["#245b85"] * 10 + ["#be6a2e"] * 10
    axes[1].scatter(fit["t_statistics"], np.arange(fit["k"]), c=colors, s=23, zorder=3)
    axes[1].axvline(0, color="0.45", linewidth=0.8)
    axes[1].set_yticks(np.arange(fit["k"]), LABELS, fontsize=8)
    axes[1].set(xlabel="Pricing-error t-statistic", title="Adjusted pricing-error tests")
    axes[1].invert_yaxis()
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(figure_dir / "q4b_gmm_results.png", dpi=300)
    plt.close(fig)

    diagnostics = (
        f"Sample: {sample.index[0]} through {sample.index[-1]}; T={n}; moments={fit['k']}\n"
        f"EQ dates excluded for incomplete inputs: {excluded}\n"
        "Annual real-return construction reused unchanged from Q3(e).\n"
        "Analytical identity-weighted GMM; covariance divisor T; no parameter bounds.\n"
        f"Mean R_w={fit['mean_rw']:.15g}; b_hat={b:.15g}; a_hat={fit['a']:.15g}\n"
        f"Mean SDF={fit['sdf'].mean():.15g}; SE(b_hat)={b_se:.15g}\n"
        f"FOC G' gbar={fit['foc']:.15e}; objective={fit['errors'] @ fit['errors']:.15g}\n"
        "HAC uses g_t(b_hat) + b_hat * mu_hat * (R_wt - mean_R_w), centered.\n"
        f"Bandwidth details: {bw}; common lag L={lag}\n"
        f"Smallest HAC eigenvalue={np.linalg.eigvalsh(fit['hac']).min():.15g}\n"
        f"Pricing-error covariance rank={fit['rank']}; joint df=20\n"
        f"Joint Wald chi-square(20)={fit['wald']:.15g}; p-value={fit['p_value']:.15g}\n"
        + "\n".join(f"RF {name}: {value:.15g}" for name, value in rf_metrics.items()) + "\n"
    )
    (result_dir / "q4b_diagnostics.txt").write_text(diagnostics)
    print(diagnostics)
    print(moment_stats.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
