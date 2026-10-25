"""Implement Prompts/Q4c_prompt.txt. Run: python3 scripts/q4c_efficient_linear_sdf.py.

Use Q4(b)'s sample and first-step adjusted HAC matrix. Keep its inverse fixed
for the analytical second step; distinguish efficient J from adjusted Wald.
"""

from q4b_linear_sdf_gmm import ROOT, LABELS, load_annual_returns, moment_hac, estimate_linear_sdf

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2


def estimate_efficient_sdf(sample, first, hac1):
    n, k = first["n"], first["k"]
    condition = float(np.linalg.cond(hac1))
    if not np.isfinite(condition) or condition >= 1 / np.finfo(float).eps:
        raise ValueError("The first-step covariance is not numerically invertible.")
    if np.linalg.eigvalsh(hac1).min() <= 0:
        raise ValueError("The first-step covariance must be positive definite.")
    weight = np.linalg.inv(hac1)
    weight = (weight + weight.T) / 2
    np.testing.assert_allclose(np.einsum("ij,jk->ik", hac1, weight), np.eye(k), atol=1e-9, rtol=0)
    mu, c = first["mu"], first["c"]
    denominator = float(c @ weight @ c)
    if not np.isfinite(denominator) or denominator <= 1e-20:
        raise ValueError("The weighted covariance-vector quadratic form is invalid.")
    # Analytical second step. The weight is computed once above and held fixed.
    b = float(c @ weight @ mu / denominator)
    a = 1 + b * first["mean_rw"]
    centered_rw = sample["R_w"].to_numpy() - first["mean_rw"]
    sdf = 1 - b * centered_rw
    np.testing.assert_allclose(sdf.mean(), 1, atol=1e-12, rtol=0)
    raw_moments = sdf[:, None] * sample[LABELS].to_numpy()
    errors = mu - b * c
    np.testing.assert_allclose(raw_moments.mean(axis=0), errors, atol=1e-12, rtol=0)
    jacobian = -c
    foc = float(jacobian @ weight @ errors)
    if abs(foc) > 1e-10:
        raise ValueError("The analytical weighted first-order condition failed.")
    adjusted_moments = raw_moments + b * centered_rw[:, None] * mu[None, :]
    np.testing.assert_allclose(adjusted_moments.mean(axis=0), errors, atol=1e-12, rtol=0)
    hac2, lag2, bandwidth2 = moment_hac(adjusted_moments)
    weighted_jacobian = weight @ jacobian
    variance = float(weighted_jacobian @ hac2 @ weighted_jacobian / (n * denominator ** 2))
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("The weighted sandwich variance must be positive and finite.")
    se = np.sqrt(variance)
    projection = np.eye(k) - np.outer(jacobian, jacobian @ weight) / denominator
    error_covariance = np.einsum("ij,jk,lk->il", projection, hac2, projection) / n
    error_covariance = (error_covariance + error_covariance.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(error_covariance)
    keep = np.abs(eigenvalues) > 1e-10 * np.max(np.abs(eigenvalues))
    if keep.sum() != k - 1 or np.any(np.diag(error_covariance) <= 0):
        raise ValueError("The adjusted pricing-error covariance must have rank 20 and positive diagonals.")
    error_se = np.sqrt(np.diag(error_covariance))
    t_statistics = errors / error_se
    inverse_errors = np.einsum("ij,j,kj->ik", eigenvectors[:, keep],
                              1 / eigenvalues[keep], eigenvectors[:, keep])
    wald = float(errors @ inverse_errors @ errors)
    j_statistic = float(n * (errors @ weight @ errors))
    return dict(b=b, a=a, se=se, weight=weight, hac2=hac2, lag2=lag2, bandwidth2=bandwidth2,
                sdf=sdf, raw_moments=raw_moments, adjusted_moments=adjusted_moments,
                errors=errors, jacobian=jacobian, projection=projection,
                error_covariance=error_covariance, error_se=error_se, t_statistics=t_statistics,
                j=j_statistic, j_p=float(chi2.sf(j_statistic, df=k-1)),
                wald=wald, wald_p=float(chi2.sf(wald, df=k-1)), foc=foc, condition=condition)


def latex_pvalue(value):
    mantissa, exponent = f"{value:.3e}".split("e")
    return rf"{mantissa}\times10^{{{int(exponent)}}}"


def main():
    sample, _, _ = load_annual_returns()
    first = estimate_linear_sdf(sample)
    saved = pd.read_csv(ROOT / "results/q4b_gmm_summary_full_precision.csv").set_index("Statistic")["Value"]
    np.testing.assert_allclose([first["b"], first["b_se"], first["n"], first["lag"]],
                               [saved["b_hat"], saved["SE(b_hat)"], saved["T"], saved["L"]],
                               atol=1e-12, rtol=0)
    hac_path = ROOT / "results/q4b_hac_matrix.csv"
    hac1 = pd.read_csv(hac_path, index_col=0).loc[LABELS, LABELS].to_numpy() if hac_path.exists() else first["hac"]
    np.testing.assert_allclose(hac1, first["hac"], atol=1e-12, rtol=0)
    fit = estimate_efficient_sdf(sample, first, hac1)
    n, lag2 = first["n"], fit["lag2"]
    result_dir, figure_dir = ROOT / "results", ROOT / "figures"
    result_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)
    summary = pd.DataFrame({
        "Statistic": ["b_hat", "SE(b_hat)", "Efficient GMM J", "J p-value (df=20)",
                      "T", "L2", "Adjusted Wald chi-square", "Wald p-value (df=20)",
                      "First-step b_hat", "L1", "Mean R_w", "Mean SDF", "a_hat", "Condition number S1"],
        "Value": [fit["b"], fit["se"], fit["j"], fit["j_p"], n, lag2, fit["wald"], fit["wald_p"],
                  first["b"], first["lag"], first["mean_rw"], fit["sdf"].mean(), fit["a"], fit["condition"]],
    })
    summary.to_csv(result_dir / "q4c_gmm_summary.csv", index=False, float_format="%.4f")
    summary.to_csv(result_dir / "q4c_gmm_summary_full_precision.csv", index=False)
    for filename, matrix in [("q4c_hac_step1.csv", hac1), ("q4c_weight_matrix.csv", fit["weight"]),
                             ("q4c_hac_step2.csv", fit["hac2"]),
                             ("q4c_pricing_error_projection.csv", fit["projection"]),
                             ("q4c_pricing_error_covariance.csv", fit["error_covariance"])]:
        pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(result_dir / filename)
    pd.DataFrame({"Portfolio": LABELS, "Mean excess return": first["mu"], "Covariance with R_w": first["c"],
                  "Jacobian": fit["jacobian"], "Pricing error": fit["errors"],
                  "Pricing-error SE": fit["error_se"], "t-statistic": fit["t_statistics"]}).to_csv(
        result_dir / "q4c_moment_statistics.csv", index=False,
    )
    pd.DataFrame({"Portfolio": LABELS, "Pricing error": fit["errors"],
                  "t-statistic": fit["t_statistics"]}).to_csv(
        result_dir / "q4c_pricing_errors.csv", index=False, float_format="%.4f",
    )
    comparison = pd.DataFrame({"Portfolio": LABELS, "Q4b pricing error": first["errors"],
                               "Q4c pricing error": fit["errors"], "Q4b t-statistic": first["t_statistics"],
                               "Q4c t-statistic": fit["t_statistics"]})
    comparison.to_csv(result_dir / "q4c_portfolio_comparison.csv", index=False)
    sample.assign(M_hat=fit["sdf"]).to_csv(result_dir / "q4c_estimation_sample.csv")
    for filename, moments in [("q4c_raw_moments.csv", fit["raw_moments"]),
                              ("q4c_adjusted_moments.csv", fit["adjusted_moments"])]:
        pd.DataFrame(moments, index=sample.index, columns=LABELS).to_csv(result_dir / filename)
    pd.DataFrame([dict(Step=1, L=first["lag"], **first["bandwidth"]),
                  dict(Step=2, L=lag2, **fit["bandwidth2"])]).to_csv(
        result_dir / "q4c_bandwidths.csv", index=False,
    )

    # Match Q4(b)'s table layout, with the efficient J as the primary joint test.
    # The separately labelled adjusted Wald is included beneath the summary.
    table = [r"\begin{table}[!htbp]", r"  \centering", r"  \small",
             r"  \caption{Two-step efficient GMM for the normalized linear SDF,",
             f"  {sample.index[0].strftime('%B %Y')} through {sample.index[-1].strftime('%B %Y')}.",
             r"  Results are rounded to four decimal places; inference accounts for estimation",
             r"  of the mean wealth return. Both joint tests use 20 degrees of freedom.}",
             r"  \label{tab:q4c-gmm-results}", r"  \begin{tabular}{lrr}", r"    \toprule",
             r"    Statistic & \multicolumn{2}{r}{Value} \\", r"    \midrule"]
    tex_labels = [r"SDF slope $\hat b_{4c}$", r"Standard error $SE(\hat b_{4c})$",
                  r"Efficient-GMM $J$ statistic", r"$J$-test $p$-value",
                  r"Observations $T$", r"Second-step HAC lag $L_2$",
                  r"Covariance-adjusted Wald $\chi^2(20)$", r"Adjusted Wald $p$-value"]
    for i, (label, value) in enumerate(zip(tex_labels, summary["Value"])):
        formatted = str(int(value)) if i in (4, 5) else f"{value:.4f}"
        table.append(f"    {label} & " + r"\multicolumn{2}{r}{" + formatted + r"} \\")
    table += [r"    \midrule", r"    Portfolio & Pricing error & $t$-statistic \\", r"    \midrule"]
    for j, label in enumerate(LABELS):
        if j in (1, 11):
            table.append(r"    \addlinespace")
        table.append(f"    {label} & {fit['errors'][j]:.4f} & {fit['t_statistics'][j]:.4f}" + r" \\")
    table += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    (result_dir / "q4c_gmm_results.tex").write_text("\n".join(table))

    bw2 = fit["bandwidth2"]
    details = [
        f"The sample is unchanged from Question 4(b): {n:,} monthly observations of annual real returns",
        f"from {sample.index[0].strftime('%B %Y')} through {sample.index[-1].strftime('%B %Y')}.",
        rf"The first-step estimate is $\hat b_{{4b}}={first['b']:.4f}$ with $L_1={first['lag']}$.",
        rf"The first-step HAC matrix is positive definite, with condition number {fit['condition']:.2f}.",
        r"Its inverse $\hat W=\hat S_1^{-1}$ is held fixed in the analytical second-step calculation.",
        rf"The fitted SDF is $\hat M_t=1-{fit['b']:.6f}(R_{{w,t}}-\bar R_w)$; its unrounded sample mean is 1.",
        "",
        r"The second-step HAC uses the adjusted series specified in the prompt, including the",
        r"correction for estimation of $E[R_w]$. The lag uses the equal-weight sum of all 21 centered adjusted series.",
        rf"The pilot is $m={bw2['pilot']}$, $C={bw2['constant']:.4f}$, and",
        rf"$L_2=\lceil CT^{{1/3}}\rceil=\lceil {bw2['bandwidth']:.4f}\rceil={lag2}$.",
        r"Bartlett autocovariances retain divisor $T-\ell$.",
        r"The full second-step covariance is saved in \texttt{results/q4c\_hac\_step2.csv}.",
        rf"The efficient $J$-test $p$-value is approximately ${latex_pvalue(fit['j_p'])}$;",
        rf"the separate covariance-adjusted Wald $p$-value is approximately ${latex_pvalue(fit['wald_p'])}$.",
        "",
        rf"Relative to Q4(b), the coefficient changes from {first['b']:.4f} to {fit['b']:.4f},",
        rf"and its standard error changes from {first['b_se']:.4f} to {fit['se']:.4f}.",
        rf"The maximum absolute sample pricing error changes from {np.abs(first['errors']).max():.4f}",
        rf"to {np.abs(fit['errors']).max():.4f}.",
        rf"The covariance-adjusted Wald statistic is {first['wald']:.4f} in Q4(b) and {fit['wald']:.4f} in Q4(c);",
        rf"the separate Q4(c) efficient-GMM $J$ statistic is {fit['j']:.4f}.",
        rf"The Q4(b) covariance-adjusted Wald $p$-value is approximately ${latex_pvalue(first['p_value'])}$.",
        r"The figure compares all 21 pricing-error $t$-statistics; full numerical portfolio comparisons",
        r"are saved in \texttt{results/q4c\_portfolio\_comparison.csv}.", "",
    ]
    (result_dir / "q4c_details.tex").write_text("\n".join(details))
    span = max(abs(first["b"] - fit["b"]), abs(fit["b"]), 1.0)
    grid = np.linspace(fit["b"] - span, fit["b"] + span, 501)
    means = first["mu"][None, :] - grid[:, None] * first["c"][None, :]
    objectives = np.einsum("ti,ij,tj->t", means, fit["weight"], means)
    pd.DataFrame({"b": grid, "Fixed-weight objective": objectives}).to_csv(
        result_dir / "q4c_objective_grid.csv", index=False,
    )
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1.15]})
    axes[0].plot(grid, objectives, color="#245b85", linewidth=1.8)
    axes[0].axvline(fit["b"], color="tab:red", linestyle="--", linewidth=1.1,
                    label=rf"$\hat b_{{4c}}={fit['b']:.4f}$")
    axes[0].scatter([fit["b"]], [fit["j"] / n], color="tab:red", s=20, zorder=3)
    axes[0].set(xlabel=r"SDF slope $b$", ylabel=r"GMM objective $\bar g'\hat W\bar g$",
                title="Fixed efficient weighting", ylim=(0, None))
    axes[0].legend(frameon=False, fontsize=9)
    positions = np.arange(first["k"])
    axes[1].scatter(first["t_statistics"], positions - 0.12, label="Q4(b)", marker="o", s=20, color="#245b85")
    axes[1].scatter(fit["t_statistics"], positions + 0.12, label="Q4(c)", marker="x", s=25, color="#be6a2e")
    axes[1].axvline(0, color="0.45", linewidth=0.8)
    axes[1].set_yticks(positions, LABELS, fontsize=8)
    axes[1].set(xlabel="Pricing-error t-statistic", title="Adjusted pricing-error tests")
    axes[1].invert_yaxis()
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(figure_dir / "q4c_gmm_results.png", dpi=300)
    plt.close(fig)
    diagnostics = (
        f"Sample: {sample.index[0]} through {sample.index[-1]}; T={n}; moments=21\n"
        f"First-step b={first['b']:.15g}; SE={first['b_se']:.15g}; L1={first['lag']}\n"
        "First-step saved covariance verified against Q4(b); W=inverse(S1), fixed in step 2.\n"
        f"Condition number S1={fit['condition']:.15g}; smallest eigenvalue={np.linalg.eigvalsh(hac1).min():.15g}\n"
        "Second-step estimator: analytical (c'W mu)/(c'W c); no numerical optimization.\n"
        f"b_hat={fit['b']:.15g}; SE={fit['se']:.15g}; a_hat={fit['a']:.15g}; mean SDF={fit['sdf'].mean():.15g}\n"
        f"Weighted FOC G'W e={fit['foc']:.15e}\n"
        f"Second-step adjusted HAC bandwidth: {bw2}; L2={lag2}\n"
        f"Smallest S2 eigenvalue={np.linalg.eigvalsh(fit['hac2']).min():.15g}\n"
        "HAC uses the moment adjustment for estimated mean R_w at each step.\n"
        f"Efficient J=T*e'W*e={fit['j']:.15g}; p-value(df=20)={fit['j_p']:.15g}\n"
        f"Separate covariance-adjusted Wald={fit['wald']:.15g}; p-value(df=20)={fit['wald_p']:.15g}\n"
        "Weighted pricing-error covariance rank=20; both tests use 20 degrees of freedom.\n"
    )
    (result_dir / "q4c_diagnostics.txt").write_text(diagnostics)
    print(diagnostics)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
