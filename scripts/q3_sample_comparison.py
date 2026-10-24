"""Additional sample check specified in Prompts/Q3_sample_comparison_prompt.txt.

Run: python3 scripts/q3_sample_comparison.py
Only writes q3_sample_comparison* outputs. Original scripts/results are retained.
"""

from hashlib import sha256

from q3b_gmm_risk_aversion import ROOT, newey_west_1994
from q3e_21_moment_gmm import LABELS, load_annual_returns, moment_hac

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


START, END = "1940-12", "2021-12"


def load_equity_returns():
    """Same input filtering and transformations as Q3(b)'s main function."""
    frame = pd.read_csv(ROOT / "datasets/EQ Dataset.csv")
    returns = frame[["re", "rf"]].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(returns.to_numpy()).all(axis=1)
    data = frame.loc[valid, ["YEAR", "MONTH"]].join(returns.loc[valid])
    data.index = pd.PeriodIndex(pd.to_datetime(dict(
        year=data["YEAR"], month=data["MONTH"], day=1,
    )), freq="M", name="Month")
    data = data.sort_index()
    if len(data) < 3 or not data.index.is_unique or not np.all(np.diff(data.index.asi8) == 1):
        raise ValueError("Q3(b) inputs must be valid, unique, consecutive monthly observations.")
    data["R_w"] = np.exp(data["re"])
    data["R_f"] = np.exp(data["rf"])
    data["xR_w"] = data["R_w"] - data["R_f"]
    return data


def fit_equity(data):
    """Retain Q3(b)'s scalar objective, unconstrained Brent, and scalar SE."""
    re, excess = data["re"].to_numpy(), data["xR_w"].to_numpy()

    def moments(gamma):
        return np.exp(-gamma * re) * excess

    def objective(gamma):
        with np.errstate(over="ignore", invalid="ignore"):
            value = np.mean(moments(gamma)) ** 2
        return float(value) if np.isfinite(value) else np.inf

    result = minimize_scalar(objective, method="brent", options={"xtol": 1e-13, "maxiter": 1000})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Q3(b) optimization failed: {result.message}")
    gamma = float(result.x)
    g = moments(gamma)
    jacobian = np.mean(-re * g)
    foc = jacobian * g.mean()
    if abs(foc) > 1e-10 or not np.isfinite(jacobian) or abs(jacobian) < 1e-12:
        raise ValueError("Q3(b) Jacobian or FOC check failed.")
    hac, lag, bw = newey_west_1994(g)
    se = np.sqrt(hac / (len(data) * jacobian ** 2))
    return dict(gamma=gamma, SE=se, L=lag, T=len(data), objective=result.fun, foc=foc,
                jacobian=np.array([jacobian]), hac=np.array([[hac]]), bandwidth=bw)


def fit_portfolios(sample):
    """Retain Q3(e)'s 21 moments, bounded optimizer, identity weight, and SE."""
    re, excess = sample["re"].to_numpy(), sample[LABELS].to_numpy()
    n = len(sample)

    def moments(gamma):
        return np.exp(-gamma * re)[:, None] * excess

    def objective(gamma):
        means = moments(gamma).mean(axis=0)
        return float(means @ means)

    optimized = minimize_scalar(objective, method="bounded", bounds=(0, 15),
                                options={"xatol": 1e-12, "maxiter": 1000})
    if not optimized.success or not np.isfinite(optimized.fun):
        raise RuntimeError(f"Q3(e) optimization failed: {optimized.message}")
    candidates = [(float(optimized.x), objective(optimized.x)),
                  (0.0, objective(0.0)), (15.0, objective(15.0))]
    gamma, minimum = min(candidates, key=lambda pair: pair[1])
    if gamma <= 1e-6 or gamma >= 15 - 1e-6:
        raise ValueError("Q3(e) boundary optimum requires review before interior inference.")
    g = moments(gamma)
    jacobian = (-re[:, None] * g).mean(axis=0)
    gg = float(jacobian @ jacobian)
    foc = float(jacobian @ g.mean(axis=0))
    if not np.isfinite(gg) or gg < 1e-20 or abs(foc) > 1e-8:
        raise ValueError("Q3(e) Jacobian or FOC check failed.")
    hac, lag, bw = moment_hac(g)
    se = np.sqrt((jacobian @ hac @ jacobian) / (n * gg ** 2))
    return dict(gamma=gamma, SE=se, L=lag, T=n, objective=minimum, foc=foc,
                jacobian=jacobian, hac=hac, bandwidth=bw)


def saved_summary(question):
    return pd.read_csv(ROOT / f"results/q3{question}_gmm_summary_full_precision.csv").set_index("Statistic")["Value"]


def main():
    protected = sorted(
        path for folder in ("scripts", "datasets", "results", "figures")
        for path in (ROOT / folder).iterdir()
        if path.is_file() and not path.name.startswith("q3_sample_comparison")
    )
    hashes = {path: sha256(path.read_bytes()).hexdigest() for path in protected}
    full_b = load_equity_returns()
    # Reuse the original full-history rolling-return construction before slicing.
    full_e, _, _ = load_annual_returns()
    restricted_b, restricted_e = full_b.loc[START:END], full_e.loc[START:END]
    expected = pd.period_range(START, END, freq="M")
    if not restricted_b.index.equals(expected) or not restricted_e.index.equals(expected):
        raise ValueError("Both restricted samples must contain exactly the same 973 months.")
    np.testing.assert_allclose(restricted_b["xR_w"], restricted_e["Equity"], atol=0, rtol=0)
    original = {q: saved_summary(q) for q in "bcde"}
    reproduced_b, reproduced_e = fit_equity(full_b), fit_portfolios(full_e)
    for q, fit in [("b", reproduced_b), ("e", reproduced_e)]:
        np.testing.assert_allclose([fit["gamma"], fit["SE"]],
                                   [original[q]["gamma_hat"], original[q]["SE(gamma_hat)"]],
                                   atol=1e-12, rtol=0)
        assert fit["L"] == original[q]["L"] and fit["T"] == original[q]["T"]
    fit_b, fit_e = fit_equity(restricted_b), fit_portfolios(restricted_e)
    rows = []

    def row(q, sample_type, start, end, moments, weighting, gamma, se, n, lag):
        rows.append(dict(Question=f"Q3({q})", Sample=sample_type, Start=str(start), End=str(end),
                         Moments=moments, Weight=weighting, T=int(n), L=int(lag), gamma=gamma, SE=se))

    for q, full, fit, moments in [("b", full_b, fit_b, 1), ("e", full_e, fit_e, 21)]:
        row(q, "Original", full.index[0], full.index[-1], moments, "Identity",
            original[q]["gamma_hat"], original[q]["SE(gamma_hat)"], original[q]["T"], original[q]["L"])
        row(q, "Restricted", START, END, moments, "Identity", fit["gamma"], fit["SE"], fit["T"], fit["L"])
    for q, gamma_key, se_key, weight, lag_key in [
        ("c", "gamma_hat", "SE(gamma_hat)", "Identity", "L"),
        ("d", "gamma_hat_2", "SE(gamma_hat_2)", "Inverse first-step HAC", "L2"),
    ]:
        dates = pd.read_csv(ROOT / f"results/q3{q}_estimation_sample.csv", usecols=[0]).iloc[:, 0]
        periods = pd.PeriodIndex(dates, freq="M")
        if not periods.equals(expected) or original[q]["T"] != len(expected):
            raise ValueError(f"Saved Q3({q}) dates do not match the comparison window.")
        row(q, "Existing", periods[0], periods[-1], 2, weight,
            original[q][gamma_key], original[q][se_key], original[q]["T"], original[q][lag_key])
    comparison = pd.DataFrame(rows).sort_values(["Question", "Sample"], kind="stable").reset_index(drop=True)
    result_dir = ROOT / "results"
    comparison.to_csv(result_dir / "q3_sample_comparison_full_precision.csv", index=False)
    comparison.to_csv(result_dir / "q3_sample_comparison.csv", index=False, float_format="%.4f")
    bandwidths, diagnostics = [], []
    for q, sample, fit, labels in [("b", restricted_b, fit_b, ["Equity"]),
                                    ("e", restricted_e, fit_e, LABELS)]:
        prefix = f"q3_sample_comparison_{q}_restricted"
        sample.to_csv(result_dir / f"{prefix}_sample.csv")
        pd.DataFrame(fit["hac"], index=labels, columns=labels).to_csv(result_dir / f"{prefix}_hac.csv")
        pd.Series(fit["jacobian"], index=labels, name="Jacobian").to_csv(result_dir / f"{prefix}_jacobian.csv")
        bandwidths.append(dict(Question=f"Q3({q})", L=fit["L"], **fit["bandwidth"]))
        diagnostics.append(f"Q3({q}) restricted: gamma={fit['gamma']:.15g}, SE={fit['SE']:.15g}, "
                           f"T={fit['T']}, L={fit['L']}, objective={fit['objective']:.15g}, FOC={fit['foc']:.15e}")
    pd.DataFrame(bandwidths).to_csv(result_dir / "q3_sample_comparison_bandwidths.csv", index=False)

    table = [r"\begin{table}[!htbp]", r"  \centering", r"  \small",
             r"  \caption{Additional sample comparison for Question 3.",
             r"  Original samples run from December 1927 through December 2021;",
             r"  restricted samples and existing Q3(c)/(d) results use December 1940",
             r"  through December 2021. Estimates and standard errors are rounded to four decimal places.}",
             r"  \label{tab:q3-sample-comparison}", r"  \setlength{\tabcolsep}{5pt}",
             r"  \begin{tabular}{llrlrrrr}", r"    \toprule",
             r"    Model & Sample & Moments & Weight & $T$ & $L$ & $\hat\gamma$ & $SE$ \\", r"    \midrule"]
    for _, record in comparison.iterrows():
        weight = r"$\hat S_1^{-1}$" if record["Question"] == "Q3(d)" else r"$I$"
        table.append(f"    {record['Question']} & {record['Sample']} & {record['Moments']} & "
                     f"{weight} & {record['T']} & {record['L']} & {record['gamma']:.4f} & {record['SE']:.4f}" + r" \\")
    table += [r"    \bottomrule", r"  \end{tabular}", r"  \par\medskip",
              r"  \begin{minipage}{0.97\textwidth}\footnotesize",
              r"  Only Q3(b) and Q3(e) are re-estimated on the restricted sample.",
              r"  Their original moment conditions, return construction, identity weights,",
              r"  optimization methods, and HAC rules are retained. Annual returns are",
              r"  constructed before restricting dates; HAC lags are selected anew on the",
              r"  restricted moments. Q3(c) and Q3(d) are copied from their saved results;",
              r"  the Q3(d) lag shown is the second-step lag. $I$ denotes the identity matrix",
              r"  of the appropriate dimension. All original results are retained.",
              r"  \end{minipage}", r"\end{table}", ""]
    (result_dir / "q3_sample_comparison.tex").write_text("\n".join(table))
    for path, digest in hashes.items():
        if sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Original file changed unexpectedly: {path}")
    diagnostics += ["Full-sample Q3(b)/(e) gamma, SE, T and L reproduce the saved outputs (tolerance 1e-12).",
                    "Q3(c)/(d) estimates and dates are read from saved results, not re-estimated.",
                    f"Verified SHA-256 preservation of {len(hashes)} original script/data/result/figure files."]
    (result_dir / "q3_sample_comparison_diagnostics.txt").write_text("\n".join(diagnostics) + "\n")
    print("\n".join(diagnostics))
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
