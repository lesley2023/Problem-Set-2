"""Implement Prompts/Q2b_prompt.txt. Run: python3 scripts/q2b_mom_sdf.py."""

# Reuse Q2(a)'s monthly-block reader, project path, and plotting setup.
from q2a_bm_sdf import ROOT, read_monthly_block

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MOM_COLUMNS = ["Lo PRIOR"] + [f"PRIOR {j}" for j in range(2, 10)] + ["Hi PRIOR"]


def main():
    # The first matching header is in the value-weighted monthly block.
    mom = read_monthly_block(ROOT / "datasets/MOM_Dataset.csv", ",Lo PRIOR,PRIOR 2")
    mom = mom[MOM_COLUMNS]
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
        "Sample rule: all overlapping months, per Prompts/Q2b_prompt.txt.\n"
        "Note: Problem Set 2.pdf specifies July 1927 as the dataset start.\n"
        "Returns in estimation: decimals; covariance divisor: T (ddof=0), as in Q2(a).\n"
        f"a_hat = {a:.12f}\n"
        f"Mean(M_star) = {sdf.mean():.12f}\n"
        f"Max absolute mean(M_star * xR_j) = {np.abs(pricing_moments).max():.3e}\n"
    )
    (result_dir / "q2b_mom_diagnostics.txt").write_text(diagnostics)
    print(diagnostics)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
