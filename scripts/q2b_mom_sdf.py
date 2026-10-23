"""Implement Prompts/Q2b_prompt.txt.

Run Q2(a) first: python3 scripts/q2a_bm_sdf.py
Then run: python3 scripts/q2b_mom_sdf.py
"""

# Reuse Q2(a)'s monthly-block reader, project path, and plotting setup.
from q2a_bm_sdf import ROOT, read_monthly_block

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MOM_COLUMNS = ["Lo PRIOR"] + [f"PRIOR {j}" for j in range(2, 10)] + ["Hi PRIOR"]


def main():
    # The first matching header is in the value-weighted monthly block.
    mom = read_monthly_block(ROOT / "datasets/MOM_Dataset.csv", ",Lo PRIOR,PRIOR 2")
    mom = mom.loc[pd.Period("1927-07", freq="M"):, MOM_COLUMNS]
    rf = read_monthly_block(ROOT / "datasets/Fama_French_3_Factors.csv", ",Mkt-RF,SMB,HML,RF")[["RF"]]
    data = mom.join(rf, how="inner", validate="one_to_one")
    if data.empty:
        raise ValueError("The MOM and RF monthly samples do not overlap.")
    # Require a specified treatment rather than silently dropping missing data.
    if data.isna().any().any() or data.isin([-99.99, -999]).any().any():
        raise ValueError("Missing returns found; specify their treatment before estimation.")
    if not np.isfinite(data.to_numpy()).all():
        raise ValueError("Nonfinite returns found in the overlapping sample.")

    # Both files report percentages. Match Q2(a)'s decimal returns and ddof=0.
    data = data / 100.0
    excess = data[MOM_COLUMNS].sub(data["RF"], axis=0)
    x = excess.to_numpy()
    mu = x.mean(axis=0)
    sigma = np.cov(x, rowvar=False, ddof=0)
    b = np.linalg.solve(sigma, mu)
    a = 1.0 + np.sum(b * mu)
    sdf = a - np.sum(x * b, axis=1)

    pricing_moments = (sdf[:, None] * x).mean(axis=0)
    np.testing.assert_allclose(sdf.mean(), 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(pricing_moments, 0.0, rtol=0, atol=1e-12)

    summary = pd.DataFrame({
        "MOM decile": np.arange(1, 11),
        "Portfolio": MOM_COLUMNS,
        "Mean excess return (monthly %)": 100.0 * mu,
        "Excess return volatility (monthly %)": 100.0 * np.sqrt(np.diag(sigma)),
        "b_j (decimal-return units)": b,
    })
    figure_dir = ROOT / "figures"
    result_dir = ROOT / "results"
    figure_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)
    summary.to_csv(result_dir / "q2b_mom_summary.csv", index=False)
    pd.DataFrame({"M_star": sdf}, index=excess.index).to_csv(result_dir / "q2b_mom_sdf.csv")

    # Match Q2(a)'s parameter-table format; keep full precision for estimation.
    parameters = pd.DataFrame({
        "Parameter": ["a"] + [f"b_{j}" for j in range(1, 11)],
        "Portfolio": ["Intercept"] + MOM_COLUMNS,
        "Estimate": np.r_[a, b],
    })
    parameters.to_csv(result_dir / "q2b_mom_parameters.csv", index=False, float_format="%.4f")
    headers = [r"$\hat a$"] + [rf"$\hat b_{{{j}}}$" for j in range(1, 11)]
    estimates = [f"{value:.4f}" for value in parameters["Estimate"]]
    table_tex = "\n".join([
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Estimated MOM SDF intercept and coefficients, rounded to four",
        r"  decimal places. Each $\hat b_j$ corresponds to MOM decile $j$;",
        r"  estimation uses decimal monthly excess returns.}",
        r"  \label{tab:q2b-mom-sdf-parameters}",
        r"  \small",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \begin{tabular}{*{11}{r}}",
        r"    \toprule",
        "    " + " & ".join(headers) + r" \\",
        r"    \midrule",
        "    " + " & ".join(estimates) + r" \\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ])
    (result_dir / "q2b_mom_parameters.tex").write_text(table_tex)

    # Correlate the saved Q2(a) SDF with MOM on their common monthly dates.
    bm_path = result_dir / "q2a_bm_sdf.csv"
    if not bm_path.exists():
        raise FileNotFoundError("Run python3 scripts/q2a_bm_sdf.py before Q2(b).")
    bm_sdf = pd.read_csv(bm_path, index_col="Month")[["M_star"]]
    bm_sdf.index = pd.PeriodIndex(bm_sdf.index, freq="M")
    aligned_sdf = bm_sdf.rename(columns={"M_star": "M_star_BM"}).join(
        pd.DataFrame({"M_star_MOM": sdf}, index=excess.index),
        how="inner", validate="one_to_one",
    ).sort_index()
    if len(aligned_sdf) < 2 or not np.isfinite(aligned_sdf.to_numpy()).all():
        raise ValueError("SDF correlation requires at least two finite overlapping observations.")
    if (aligned_sdf.std(ddof=0) == 0).any():
        raise ValueError("SDF correlation is undefined for a constant series.")
    correlation = aligned_sdf["M_star_BM"].corr(aligned_sdf["M_star_MOM"])
    aligned_sdf.to_csv(result_dir / "q2b_bm_mom_aligned_sdf.csv")
    pd.DataFrame({
        "Start month": [str(aligned_sdf.index[0])],
        "End month": [str(aligned_sdf.index[-1])],
        "Observations": [len(aligned_sdf)],
        "Correlation": [correlation],
    }).to_csv(result_dir / "q2b_bm_mom_correlation.csv", index=False)
    correlation_tex = (
        "The sample correlation between the BM and MOM SDFs is\n"
        r"\["
        "\n"
        r"  \widehat{\operatorname{Corr}}"
        r"\!\left(\hat M^{*,BM},\hat M^{*,MOM}\right) = "
        f"{correlation:.4f}.\n"
        r"\]"
        "\n"
        f"This uses {len(aligned_sdf):,} aligned monthly observations from "
        f"{aligned_sdf.index[0].strftime('%B %Y')} through "
        f"{aligned_sdf.index[-1].strftime('%B %Y')}.\n"
    )
    (result_dir / "q2b_bm_mom_correlation.tex").write_text(correlation_tex)

    fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
    ax.scatter(np.arange(1, 11), b, s=65, color="#245b85", zorder=3)
    ax.axhline(0, color="0.45", linewidth=0.9, zorder=2)
    ax.set_xticks(np.arange(1, 11))
    ax.set_xlim(0.5, 10.5)
    ax.set_xlabel("Momentum decile (1 = low prior return, 10 = high prior return)")
    ax.set_ylabel(r"Estimated SDF coefficient $\hat{b}_j$ (decimal returns)")
    ax.set_title("Unconditional SDF coefficients: MOM portfolios", loc="left", pad=12)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(figure_dir / "q2b_mom_sdf_coefficients.png", dpi=300)
    plt.close(fig)

    diagnostics = (
        f"Sample: {excess.index[0]} to {excess.index[-1]} (T = {len(excess)})\n"
        "Sample rule: July 1927 onward, keeping overlapping months, per Prompts/Q2b_prompt.txt.\n"
        "Returns in estimation: decimals; covariance divisor: T (ddof=0), as in Q2(a).\n"
        f"a_hat = {a:.12f}\n"
        f"Mean(M_star) = {sdf.mean():.12f}\n"
        f"Max absolute mean(M_star * xR_j) = {np.abs(pricing_moments).max():.3e}\n"
        f"Corr(M_star_BM, M_star_MOM) = {correlation:.12f}\n"
        f"Correlation sample: {aligned_sdf.index[0]} to {aligned_sdf.index[-1]} "
        f"(T = {len(aligned_sdf)})\n"
    )
    (result_dir / "q2b_mom_diagnostics.txt").write_text(diagnostics)
    print(diagnostics)
    print(parameters.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
