"""Implement Prompts/Q2d_prompt.txt. Run: python3 scripts/q2d_pricing_errors.py.

Requires the saved SDF series from scripts/q2a_bm_sdf.py,
scripts/q2b_mom_sdf.py, and scripts/q2c_combined_sdf.py.
"""

from q2c_combined_sdf import ROOT, load_excess_returns

import numpy as np
import pandas as pd


def main():
    excess = load_excess_returns()
    result_dir = ROOT / "results"
    sdf_files = {
        "BM SDF": "q2a_bm_sdf.csv",
        "MOM SDF": "q2b_mom_sdf.csv",
        "Combined SDF": "q2c_combined_sdf.csv",
    }
    # Saved full-precision series already use the existing a/b estimates.
    # Do not reconstruct SDFs using the rounded parameter-table coefficients.
    sdf_series = []
    for label, filename in sdf_files.items():
        path = result_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing {path.name}; run Q2(a), Q2(b), and Q2(c) first.")
        frame = pd.read_csv(path, index_col="Month")[["M_star"]]
        frame.index = pd.PeriodIndex(frame.index, freq="M")
        if not frame.index.is_unique:
            raise ValueError(f"Duplicate months in {filename}.")
        sdf_series.append(frame.rename(columns={"M_star": label}))
    data = excess
    for frame in sdf_series:
        data = data.join(frame, how="inner", validate="one_to_one")
    data = data.sort_index()
    if len(data) < 2 or not np.isfinite(data.to_numpy()).all():
        raise ValueError("Pricing errors require at least two finite common monthly observations.")
    excess = data[excess.columns]
    sdfs = data[list(sdf_files)]
    np.testing.assert_allclose(sdfs.mean(), 1.0, rtol=0, atol=1e-12)

    x = excess.to_numpy()
    negative_sdf = -sdfs.to_numpy()
    means = x.mean(axis=0)
    centered_x = x - means
    centered_negative_sdf = negative_sdf - negative_sdf.mean(axis=0)
    # Cov(-M, xR) = mean[(-M - mean(-M)) * (xR - mean(xR))], divisor T.
    covariances = (
        centered_negative_sdf[:, :, None] * centered_x[:, None, :]
    ).mean(axis=0)
    errors = (means[None, :] - covariances).T

    # Verify against the sample pricing moments, since each saved SDF has mean 1.
    direct_moments = (x[:, :, None] * sdfs.to_numpy()[:, None, :]).mean(axis=0)
    np.testing.assert_allclose(errors, direct_moments, rtol=0, atol=1e-12)
    np.testing.assert_allclose(errors[:10, 0], 0.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(errors[10:, 1], 0.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(errors[:, 2], 0.0, rtol=0, atol=1e-12)

    pricing_errors = pd.DataFrame(errors, index=excess.columns, columns=list(sdf_files))
    pricing_errors.index.name = "Portfolio"
    pricing_errors.to_csv(result_dir / "q2d_pricing_errors_full_precision.csv")

    # Replace only displayed negative zero; preserve the unrounded calculations.
    def display_error(value):
        formatted = f"{value:.6f}"
        return "0.000000" if formatted == "-0.000000" else formatted

    display_table = pricing_errors.map(display_error)
    display_table.to_csv(result_dir / "q2d_pricing_errors.csv")
    table_lines = [
        r"\begin{table}[!htbp]",
        r"  \centering",
        r"  \caption{Monthly pricing errors $\hat\alpha_p$ under the BM, MOM, and",
        r"  combined SDFs, in decimal return units and rounded to six decimal places.",
        f"  The common sample is {excess.index[0].strftime('%B %Y')} through "
        f"{excess.index[-1].strftime('%B %Y')} ({len(excess):,} months); "
        r"covariances use divisor $T$.}",
        r"  \label{tab:q2d-pricing-errors}",
        r"  \small",
        r"  \begin{tabular}{lrrr}",
        r"    \toprule",
        r"    Portfolio & BM SDF & MOM SDF & Combined SDF \\",
        r"    \midrule",
    ]
    for i, (portfolio, row) in enumerate(display_table.iterrows()):
        if i == 10:
            table_lines.append(r"    \midrule")
        table_lines.append("    " + portfolio + " & " + " & ".join(row) + r" \\")
    table_lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    (result_dir / "q2d_pricing_errors.tex").write_text("\n".join(table_lines))

    # Numerical comparisons only; economic interpretation is student-authored.
    own_bm = np.abs(errors[:10, 0]).max()
    own_mom = np.abs(errors[10:, 1]).max()
    combined = np.abs(errors[:, 2]).max()
    cross_bm = np.abs(errors[10:, 0]).max()
    cross_mom = np.abs(errors[:10, 1]).max()
    summary_tex = (
        "The maximum absolute pricing errors are "
        f"${own_bm:.2e}$ for BM portfolios under the BM SDF, "
        f"${own_mom:.2e}$ for MOM portfolios under the MOM SDF, and "
        f"${combined:.2e}$ across all 20 portfolios under the combined SDF. "
        "All are below $10^{-12}$ in decimal return units.\n\n"
        "For portfolios outside the group used to construct each SDF, "
        "the maximum absolute errors are "
        f"{cross_bm:.6f} for MOM portfolios under the BM SDF and "
        f"{cross_mom:.6f} for BM portfolios under the MOM SDF.\n"
    )
    # Use native LaTeX scientific notation rather than a literal e-notation string.
    def scientific_tex(value):
        mantissa, exponent = f"{value:.2e}".split("e")
        return rf"{mantissa}\times 10^{{{int(exponent)}}}"

    for value in [own_bm, own_mom, combined]:
        summary_tex = summary_tex.replace(f"${value:.2e}$", f"${scientific_tex(value)}$")
    (result_dir / "q2d_verification.tex").write_text(summary_tex)

    diagnostics = (
        f"Common sample: {excess.index[0]} to {excess.index[-1]} (T = {len(excess)})\n"
        "Returns and pricing errors: decimals; covariance divisor: T (ddof=0).\n"
        "Reused saved Q2(a), Q2(b), and Q2(c) SDFs; no re-estimation.\n"
        f"Max |alpha|, BM portfolios / BM SDF: {own_bm:.12e}\n"
        f"Max |alpha|, MOM portfolios / MOM SDF: {own_mom:.12e}\n"
        f"Max |alpha|, all portfolios / Combined SDF: {combined:.12e}\n"
        f"Max |alpha|, MOM portfolios / BM SDF: {cross_bm:.12e}\n"
        f"Max |alpha|, BM portfolios / MOM SDF: {cross_mom:.12e}\n"
        f"Max discrepancy from direct pricing moments: {np.abs(errors - direct_moments).max():.12e}\n"
    )
    (result_dir / "q2d_diagnostics.txt").write_text(diagnostics)
    print(diagnostics)
    print(display_table.to_string())


if __name__ == "__main__":
    main()
