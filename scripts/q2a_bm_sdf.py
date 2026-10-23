"""Implement Prompts/Q2a_prompt.txt. Run: python3 scripts/q2a_bm_sdf.py."""

from io import StringIO
from pathlib import Path
import os
import re
import tempfile

# Keep Matplotlib's cache in a writable location when running in a sandbox.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "q2a-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BM_COLUMNS = ["Lo 10"] + [f"{j}-Dec" for j in range(2, 10)] + ["Hi 10"]


def read_monthly_block(path, header_marker):
    """Read one monthly CSV block, excluding later annual/other sections."""
    lines = path.read_text().splitlines()
    header = next(i for i, line in enumerate(lines) if header_marker in line)
    rows = []
    for line in lines[header + 1:]:
        if not re.match(r"^\s*\d{6},", line):
            break
        rows.append(line)
    frame = pd.read_csv(StringIO("\n".join([lines[header]] + rows)), index_col=0)
    frame.columns = frame.columns.str.strip()
    frame.index = pd.to_datetime(frame.index.astype(str), format="%Y%m").to_period("M")
    frame.index.name = "Month"
    if frame.empty or not frame.index.is_unique:
        raise ValueError(f"Expected a nonempty monthly block with unique dates: {path}")
    return frame.sort_index()


def main():
    # This header occurs first in the value-weighted monthly BM section.
    bm = read_monthly_block(ROOT / "datasets/BM_Dataset.csv", ",<= 0,Lo 30")
    bm = bm.loc[pd.Period("1927-07", freq="M"):, BM_COLUMNS]
    rf = read_monthly_block(ROOT / "datasets/Fama_French_3_Factors.csv", ",Mkt-RF,SMB,HML,RF")[["RF"]]
    data = bm.join(rf, how="inner", validate="one_to_one")
    if data.empty:
        raise ValueError("The BM and RF monthly samples do not overlap.")
    # Fail instead of silently choosing a missing-observation treatment.
    if data.isna().any().any() or data.isin([-99.99, -999]).any().any():
        raise ValueError("Missing returns found; specify their treatment before estimation.")
    if not np.isfinite(data.to_numpy()).all():
        raise ValueError("Nonfinite returns found in the overlapping sample.")

    # Both files report percentages. Keep overlapping months from July 1927.
    data = data / 100.0
    excess = data[BM_COLUMNS].sub(data["RF"], axis=0)
    x = excess.to_numpy()
    mu = x.mean(axis=0)
    sigma = np.cov(x, rowvar=False, ddof=0)
    b = np.linalg.solve(sigma, mu)  # Equivalent to inv(sigma) @ mu.
    a = 1.0 + b @ mu
    sdf = a - np.sum(x * b, axis=1)

    # Verify the normalization and all ten ex-post pricing moment conditions.
    pricing_moments = (sdf[:, None] * x).mean(axis=0)
    np.testing.assert_allclose(sdf.mean(), 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(pricing_moments, 0.0, rtol=0, atol=1e-12)

    summary = pd.DataFrame({
        "BM decile": np.arange(1, 11),
        "Portfolio": BM_COLUMNS,
        "Mean excess return (monthly %)": 100.0 * mu,
        "Excess return volatility (monthly %)": 100.0 * np.sqrt(np.diag(sigma)),
        "b_j (decimal-return units)": b,
    })
    figure_dir = ROOT / "figures"
    result_dir = ROOT / "results"
    figure_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)
    summary.to_csv(result_dir / "q2a_bm_summary.csv", index=False)
    pd.DataFrame({"M_star": sdf}, index=excess.index).to_csv(result_dir / "q2a_bm_sdf.csv")

    # Round only the displayed/exported parameter table, not the SDF estimates.
    parameters = pd.DataFrame({
        "Parameter": ["a"] + [f"b_{j}" for j in range(1, 11)],
        "Portfolio": ["Intercept"] + BM_COLUMNS,
        "Estimate": np.r_[a, b],
    })
    parameters.to_csv(result_dir / "q2a_bm_parameters.csv", index=False, float_format="%.4f")
    headers = [r"$\hat a$"] + [rf"$\hat b_{{{j}}}$" for j in range(1, 11)]
    estimates = [f"{value:.4f}" for value in parameters["Estimate"]]
    table_tex = "\n".join([
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Estimated BM SDF intercept and coefficients, rounded to four",
        r"  decimal places. Each $\hat b_j$ corresponds to BM decile $j$;",
        r"  estimation uses decimal monthly excess returns.}",
        r"  \label{tab:q2a-bm-sdf-parameters}",
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
    (result_dir / "q2a_bm_parameters.tex").write_text(table_tex)

    fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
    ax.scatter(np.arange(1, 11), b, s=65, color="#245b85", zorder=3)
    ax.axhline(0, color="0.45", linewidth=0.9, zorder=2)
    ax.set_xticks(np.arange(1, 11))
    ax.set_xlim(0.5, 10.5)
    ax.set_xlabel("Book-to-market decile (1 = low BM, 10 = high BM)")
    ax.set_ylabel(r"Estimated SDF coefficient $\hat{b}_j$ (decimal returns)")
    ax.set_title("Unconditional SDF coefficients: BM portfolios", loc="left", pad=12)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(figure_dir / "q2a_bm_sdf_coefficients.png", dpi=300)
    plt.close(fig)

    diagnostics = (
        f"Sample: {excess.index[0]} to {excess.index[-1]} (T = {len(excess)})\n"
        "Sample rule: July 1927 onward, keeping overlapping months, per Prompts/Q2a_prompt.txt.\n"
        "Returns in estimation: decimals; covariance divisor: T (ddof=0).\n"
        f"a_hat = {a:.12f}\n"
        f"Mean(M_star) = {sdf.mean():.12f}\n"
        f"Max absolute mean(M_star * xR_j) = {np.abs(pricing_moments).max():.3e}\n"
    )
    (result_dir / "q2a_bm_diagnostics.txt").write_text(diagnostics)
    print(diagnostics)
    print(parameters.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
