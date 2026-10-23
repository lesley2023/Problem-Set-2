"""Implement Prompts/Q2c_prompt.txt. Run: python3 scripts/q2c_combined_sdf.py."""

from q2a_bm_sdf import ROOT, BM_COLUMNS, read_monthly_block
from q2b_mom_sdf import MOM_COLUMNS

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def estimate_sdf(excess):
    """Use Q2(a)/(b)'s estimator, population-divisor sample moments, and checks."""
    x = excess.to_numpy()
    mu = x.mean(axis=0)
    sigma = np.cov(x, rowvar=False, ddof=0)
    b = np.linalg.solve(sigma, mu)
    a = 1.0 + np.sum(b * mu)
    x_star = np.sum(x * b, axis=1)
    sdf = a - x_star
    pricing_moments = (sdf[:, None] * x).mean(axis=0)
    np.testing.assert_allclose(sdf.mean(), 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(pricing_moments, 0.0, rtol=0, atol=1e-12)
    return a, b, sdf, x_star, np.abs(pricing_moments).max()


def main():
    # Reuse the existing reader, selecting only monthly value-weighted deciles.
    bm = read_monthly_block(ROOT / "datasets/BM_Dataset.csv", ",<= 0,Lo 30")[BM_COLUMNS]
    mom = read_monthly_block(ROOT / "datasets/MOM_Dataset.csv", ",Lo PRIOR,PRIOR 2")[MOM_COLUMNS]
    rf = read_monthly_block(ROOT / "datasets/Fama_French_3_Factors.csv", ",Mkt-RF,SMB,HML,RF")[["RF"]]
    # Prefix columns to retain the requested BM-then-MOM order unambiguously.
    data = bm.add_prefix("BM: ").join(
        mom.add_prefix("MOM: "), how="inner", validate="one_to_one",
    ).join(rf, how="inner", validate="one_to_one")
    data = data.loc[pd.Period("1927-07", freq="M"):]
    if len(data) < 2:
        raise ValueError("At least two overlapping months from July 1927 are required.")
    if data.isna().any().any() or data.isin([-99.99, -999]).any().any():
        raise ValueError("Missing returns found; specify their treatment before estimation.")
    if not np.isfinite(data.to_numpy()).all():
        raise ValueError("Nonfinite returns found in the overlapping sample.")

    data = data / 100.0  # All three supplied files report percentages.
    excess = data.drop(columns="RF").sub(data["RF"], axis=0)
    # Re-estimate all cases on this common sample, avoiding rounded weights.
    fits = {
        "BM": estimate_sdf(excess.iloc[:, :10]),
        "MOM": estimate_sdf(excess.iloc[:, 10:]),
        "Combined": estimate_sdf(excess),
    }
    a, b, sdf, _, _ = fits["Combined"]
    portfolios = pd.DataFrame({case: fit[3] for case, fit in fits.items()}, index=excess.index)
    means = portfolios.mean()
    volatilities = portfolios.std(ddof=0)
    if (volatilities <= 0).any():
        raise ValueError("Sharpe ratios require positive portfolio volatility.")
    ratios = means / volatilities
    sharpe_table = pd.DataFrame({
        "Case": list(fits),
        "Mean monthly excess return (decimal)": means.to_numpy(),
        "Monthly volatility (decimal)": volatilities.to_numpy(),
        "Monthly Sharpe ratio": ratios.to_numpy(),
    })

    figure_dir = ROOT / "figures"
    result_dir = ROOT / "results"
    figure_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)
    pd.DataFrame({"M_star": sdf}, index=excess.index).to_csv(result_dir / "q2c_combined_sdf.csv")
    portfolios.to_csv(result_dir / "q2c_max_sharpe_portfolios.csv")
    sharpe_table.to_csv(result_dir / "q2c_sharpe_ratios.csv", index=False)

    parameters = pd.DataFrame({
        "Parameter": ["a"] + [f"b_{j}" for j in range(1, 21)],
        "Group": ["Intercept"] + ["BM"] * 10 + ["MOM"] * 10,
        "Portfolio": ["Intercept"] + BM_COLUMNS + MOM_COLUMNS,
        "Estimate": np.r_[a, b],
    })
    parameters.to_csv(result_dir / "q2c_combined_parameters.csv", index=False, float_format="%.4f")
    parameter_lines = [
        r"\begin{table}[!htbp]",
        r"  \centering",
        r"  \caption{Combined SDF intercept and coefficients, rounded to four decimal",
        r"  places. The first ten coefficients correspond to BM portfolios and the",
        r"  next ten to MOM portfolios. Estimation uses decimal excess returns.}",
        r"  \label{tab:q2c-combined-sdf-parameters}",
        r"  \small",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \begin{tabular}{l*{10}{r}}",
        r"    \toprule",
        r"    \multicolumn{10}{l}{SDF intercept $\hat a$} & " + f"{a:.4f}" + r" \\",
        r"    \midrule",
        r"    Decile $j$ & " + " & ".join(str(j) for j in range(1, 11)) + r" \\",
        r"    \midrule",
    ]
    parameter_lines += [
        "    BM & " + " & ".join(f"{value:.4f}" for value in b[:10]) + r" \\",
        "    MOM & " + " & ".join(f"{value:.4f}" for value in b[10:]) + r" \\",
    ]
    parameter_lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    (result_dir / "q2c_combined_parameters.tex").write_text("\n".join(parameter_lines))

    sharpe_lines = [
        r"\begin{table}[!htbp]",
        r"  \centering",
        r"  \caption{Monthly Sharpe ratios of $xR_t^*=\hat b'xR_t$, using the same",
        f"  {len(excess):,} months from {excess.index[0].strftime('%B %Y')} through "
        f"{excess.index[-1].strftime('%B %Y')}. "
        r"Variances use divisor $T$.}",
        r"  \label{tab:q2c-monthly-sharpe-ratios}",
        r"  \small",
        r"  \begin{tabular}{lr}",
        r"    \toprule",
        r"    Portfolio set & Monthly Sharpe ratio \\",
        r"    \midrule",
    ]
    for case, ratio in ratios.items():
        sharpe_lines.append(f"    {case} & {ratio:.4f}" + r" \\")
    sharpe_lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    (result_dir / "q2c_sharpe_ratios.tex").write_text("\n".join(sharpe_lines))

    fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
    deciles = np.arange(1, 11)
    ax.scatter(deciles, b[:10], s=65, color="tab:blue", label="BM", zorder=3)
    ax.scatter(deciles, b[10:], s=65, color="tab:red", label="MOM", zorder=3)
    ax.axhline(0, color="0.45", linewidth=0.9, zorder=2)
    ax.set_xticks(deciles)
    ax.set_xlim(0.5, 10.5)
    ax.set_xlabel("Portfolio decile (low to high BM or prior return)")
    ax.set_ylabel(r"Estimated SDF coefficient $\hat{b}_j$ (decimal returns)")
    ax.set_title("Unconditional SDF coefficients: combined BM and MOM", loc="left", pad=12)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(figure_dir / "q2c_combined_sdf_coefficients.png", dpi=300)
    plt.close(fig)

    diagnostics = (
        f"Common sample: {excess.index[0]} to {excess.index[-1]} (T = {len(excess)})\n"
        "Returns: decimals; sample covariance and variance divisor: T (ddof=0).\n"
        "Vector order: ten BM deciles followed by ten MOM deciles.\n"
        f"Combined a_hat = {a:.12f}\n"
    )
    for case, fit in fits.items():
        diagnostics += (
            f"{case}: mean(M_star) = {fit[2].mean():.12f}; "
            f"max absolute pricing moment = {fit[4]:.3e}; "
            f"monthly SR = {ratios[case]:.12f}\n"
        )
    (result_dir / "q2c_diagnostics.txt").write_text(diagnostics)
    print(diagnostics)
    print(parameters.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(sharpe_table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
