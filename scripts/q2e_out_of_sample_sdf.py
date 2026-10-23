"""Implement Prompts/Q2e_prompt.txt.

Run: python3 scripts/q2e_out_of_sample_sdf.py
"""

from q2c_combined_sdf import ROOT, load_excess_returns, estimate_sdf
from q2d_pricing_errors import calculate_pricing_errors, format_pricing_error

import numpy as np
import pandas as pd


def main():
    excess = load_excess_returns()
    in_sample = excess.loc[:pd.Period("1989-12", freq="M")]
    out_of_sample = excess.loc[pd.Period("1990-01", freq="M"):]
    if len(in_sample) < 2 or len(out_of_sample) < 2:
        raise ValueError("At least two observations are required in each sample period.")
    if len(in_sample.index.intersection(out_of_sample.index)):
        raise ValueError("Estimation and evaluation periods must not overlap.")

    # Q2(c)'s estimator normalizes the SDF using only the estimation sample.
    a_is, b_is, sdf_is, _, max_is_moment = estimate_sdf(in_sample)
    x_os = out_of_sample.to_numpy()
    # Freeze both a and b: no re-estimation or out-of-sample renormalization.
    sdf_os = a_is - np.sum(x_os * b_is, axis=1)
    if not np.isfinite(sdf_os).all():
        raise ValueError("Nonfinite out-of-sample SDF values.")
    errors = calculate_pricing_errors(
        out_of_sample,
        pd.DataFrame({"Out-of-sample pricing error": sdf_os}, index=out_of_sample.index),
    )

    # Independently verify the covariance expression with Sigma_OS @ b_IS.
    mu_os = x_os.mean(axis=0)
    sigma_os = np.cov(x_os, rowvar=False, ddof=0)
    covariance_check = mu_os - np.sum(sigma_os * b_is[None, :], axis=1)
    np.testing.assert_allclose(errors.iloc[:, 0], covariance_check, rtol=0, atol=1e-12)

    result_dir = ROOT / "results"
    result_dir.mkdir(exist_ok=True)
    # Save unrounded fixed coefficients and SDF series for reproducibility.
    pd.DataFrame({
        "Parameter": ["a_IS"] + [f"b_IS_{j}" for j in range(1, 21)],
        "Portfolio": ["Intercept"] + excess.columns.tolist(),
        "Estimate": np.r_[a_is, b_is],
    }).to_csv(result_dir / "q2e_is_parameters.csv", index=False)
    pd.DataFrame({"M_star_IS": sdf_is}, index=in_sample.index).to_csv(result_dir / "q2e_is_sdf.csv")
    pd.DataFrame({"M_star_OS": sdf_os}, index=out_of_sample.index).to_csv(result_dir / "q2e_os_sdf.csv")
    errors.to_csv(result_dir / "q2e_os_pricing_errors_full_precision.csv")
    display_table = errors.map(format_pricing_error)
    display_table.to_csv(result_dir / "q2e_os_pricing_errors.csv")

    table_lines = [
        r"\begin{table}[!htbp]",
        r"  \centering",
        r"  \caption{Out-of-sample monthly pricing errors $\hat\alpha_{OS,p}$,",
        r"  in decimal return units and rounded to six decimal places. The SDF",
        f"  is estimated using {in_sample.index[0].strftime('%B %Y')} through "
        f"{in_sample.index[-1].strftime('%B %Y')} ({len(in_sample):,} months),",
        r"  with its intercept and coefficients held fixed for evaluation over",
        f"  {out_of_sample.index[0].strftime('%B %Y')} through "
        f"{out_of_sample.index[-1].strftime('%B %Y')} ({len(out_of_sample):,} months).",
        r"  Covariances use divisor $T_{OS}$.}",
        r"  \label{tab:q2e-out-of-sample-pricing-errors}",
        r"  \small",
        r"  \begin{tabular}{lr}",
        r"    \toprule",
        r"    Portfolio & Out-of-sample pricing error \\",
        r"    \midrule",
    ]
    for i, (portfolio, row) in enumerate(display_table.iterrows()):
        if i == 10:
            table_lines.append(r"    \midrule")
        table_lines.append("    " + portfolio + " & " + row.iloc[0] + r" \\")
    table_lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    (result_dir / "q2e_os_pricing_errors.tex").write_text("\n".join(table_lines))

    diagnostics = (
        f"In-sample: {in_sample.index[0]} to {in_sample.index[-1]} (T_IS = {len(in_sample)})\n"
        f"Out-of-sample: {out_of_sample.index[0]} to {out_of_sample.index[-1]} "
        f"(T_OS = {len(out_of_sample)})\n"
        "Returns/errors: decimals; covariance divisor: corresponding sample size (ddof=0).\n"
        "Vector order: ten BM portfolios followed by ten MOM portfolios.\n"
        f"Fixed a_IS = {a_is:.12f}\n"
        f"Mean(M_star_IS) = {sdf_is.mean():.12f}\n"
        f"Maximum absolute in-sample pricing moment = {max_is_moment:.12e}\n"
        f"Mean(M_star_OS) = {sdf_os.mean():.12f}\n"
        "No out-of-sample re-estimation or renormalization.\n"
        "Pricing errors use the specified mean-minus-covariance formula.\n"
        f"Maximum absolute out-of-sample pricing error = {errors.iloc[:, 0].abs().max():.12e}\n"
        f"Maximum covariance-check discrepancy = "
        f"{np.abs(errors.iloc[:, 0].to_numpy() - covariance_check).max():.12e}\n"
    )
    (result_dir / "q2e_diagnostics.txt").write_text(diagnostics)
    print(diagnostics)
    print(display_table.to_string())


if __name__ == "__main__":
    main()
