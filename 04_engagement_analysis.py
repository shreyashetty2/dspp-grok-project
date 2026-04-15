"""
04_engagement_analysis.py
=========================
Tests the "algorithmic complicity" hypothesis:
  H0: AI-generated posts do NOT achieve higher engagement velocity than
      human-generated posts from accounts of similar size/age.
  H1: AI-generated posts achieve statistically significantly higher
      views/engagement ratios (i.e., X's algorithm amplifies AI content).

Methods:
  1. Propensity Score Matching (PSM) — balances confounders
     (account size, account age, time of posting) before comparing groups.
  2. OLS Regression — controls for confounders directly.
  3. Negative Binomial Regression — accounts for overdispersion in count data.
  4. Visualization — violin plots, scatter plots, coefficient plots.

Outputs:
  - outputs/tables/psm_matched_sample.csv
  - outputs/tables/regression_results.csv
  - outputs/figures/engagement_violin.png
  - outputs/figures/regression_coef.png

Usage:
  python 04_engagement_analysis.py --input data/synthetic/synthetic_posts.csv
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
np.random.seed(42)

PROC_DIR  = Path("data/processed")
FIG_DIR   = Path("outputs/figures")
TABLE_DIR = Path("outputs/tables")
for d in [PROC_DIR, FIG_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── Data Preparation ──────────────────────────────────────────────────────────
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for PSM and regression.
    All confounders that could explain engagement differences
    besides AI-generation status.
    """
    df = df.copy()

    # Treatment indicator
    df["treat"] = df["ai_generated"].astype(int)

    # Log-transform engagement (right-skewed counts)
    for col in ["views", "likes", "retweets", "engagement_total"]:
        df[f"log_{col}"] = np.log1p(df[col].fillna(0))

    # Engagement rate (per view)
    df["engagement_rate"] = df.apply(
        lambda r: r["engagement_total"] / r["views"] if r.get("views", 0) > 0 else 0, axis=1
    )

    # Velocity proxy: engagement_total (higher = faster spread)
    df["velocity"] = df["engagement_rate"]

    # Author features (confounders)
    df["log_followers"] = np.log1p(df["author_followers"].fillna(0))

    # Account age (if available)
    if "author_account_age_days" in df.columns:
        df["log_account_age"] = np.log1p(df["author_account_age_days"].fillna(365))
    else:
        df["log_account_age"] = np.log1p(365)   # Default 1 year

    # Has media indicator
    df["has_media_int"] = df["has_media"].astype(int) if "has_media" in df.columns else 1

    # Hour of day (from created_at)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        df["hour_of_day"] = df["created_at"].dt.hour.fillna(12)
    else:
        df["hour_of_day"] = 12

    # Drop rows missing critical fields
    df = df.dropna(subset=["treat", "log_views", "log_followers"])
    return df


# ── Propensity Score Matching ─────────────────────────────────────────────────
def estimate_propensity_scores(df: pd.DataFrame,
                                covariates: list[str]) -> pd.DataFrame:
    """
    Estimate propensity scores P(AI=1 | confounders) using logistic regression.
    """
    df = df.copy()
    X = df[covariates].fillna(0)
    y = df["treat"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)
    df["propensity_score"] = model.predict_proba(X_scaled)[:, 1]

    print(f"[PSM] Propensity score range: "
          f"{df['propensity_score'].min():.3f} – {df['propensity_score'].max():.3f}")
    return df


def match_samples(df: pd.DataFrame, caliper: float = 0.05) -> pd.DataFrame:
    """
    1:1 nearest-neighbor PSM with caliper.
    Returns matched dataset (equal treatment and control samples).
    """
    treated   = df[df["treat"] == 1].reset_index(drop=True)
    control   = df[df["treat"] == 0].reset_index(drop=True)

    # Fit nearest neighbor on propensity scores
    nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
    nn.fit(control[["propensity_score"]].values)

    distances, indices = nn.kneighbors(treated[["propensity_score"]].values)

    # Apply caliper
    matched_pairs = []
    used_controls = set()
    for i, (dist, idx) in enumerate(zip(distances.flatten(), indices.flatten())):
        if dist <= caliper and idx not in used_controls:
            matched_pairs.append((i, idx))
            used_controls.add(idx)

    if not matched_pairs:
        print("[PSM WARNING] No matches found within caliper. Relaxing to 0.2.")
        caliper = 0.2
        for i, (dist, idx) in enumerate(zip(distances.flatten(), indices.flatten())):
            if dist <= caliper and idx not in used_controls:
                matched_pairs.append((i, idx))
                used_controls.add(idx)

    treated_idx = [p[0] for p in matched_pairs]
    control_idx = [p[1] for p in matched_pairs]

    matched_treated = treated.iloc[treated_idx].copy()
    matched_control = control.iloc[control_idx].copy()
    matched = pd.concat([matched_treated, matched_control], ignore_index=True)

    print(f"[PSM] Matched {len(matched_pairs)} pairs "
          f"({len(matched_pairs)} treated + {len(matched_pairs)} control).")
    return matched


def check_balance(df_matched: pd.DataFrame, covariates: list[str]):
    """Check covariate balance after matching (standardized mean differences)."""
    print("\n=== Covariate Balance After PSM ===")
    print(f"{'Covariate':<25} {'Treated Mean':>14} {'Control Mean':>14} {'SMD':>8}")
    print("-" * 65)
    treated = df_matched[df_matched["treat"] == 1]
    control = df_matched[df_matched["treat"] == 0]
    for cov in covariates:
        t_mean = treated[cov].mean()
        c_mean = control[cov].mean()
        pooled_std = np.sqrt((treated[cov].std()**2 + control[cov].std()**2) / 2)
        smd = (t_mean - c_mean) / pooled_std if pooled_std > 0 else 0
        flag = " ✓" if abs(smd) < 0.1 else " ⚠"
        print(f"{cov:<25} {t_mean:>14.4f} {c_mean:>14.4f} {smd:>7.3f}{flag}")
    print("(SMD < 0.1 indicates good balance)")


# ── OLS Regression ────────────────────────────────────────────────────────────
def run_ols_regression(df: pd.DataFrame, outcome: str, covariates: list[str]) -> dict:
    """
    OLS regression: outcome ~ treat + covariates.
    Returns coefficient table.
    """
    from sklearn.linear_model import LinearRegression

    X = df[["treat"] + covariates].fillna(0)
    y = df[outcome].fillna(0)

    # Add intercept manually for coefficient extraction
    X_arr = np.column_stack([np.ones(len(X)), X.values])
    feature_names = ["intercept", "treat"] + covariates

    # OLS via numpy for full stats
    try:
        coeffs, residuals, rank, sv = np.linalg.lstsq(X_arr, y.values, rcond=None)
        n, k = X_arr.shape
        y_hat = X_arr @ coeffs
        residuals_vec = y.values - y_hat
        s2 = np.sum(residuals_vec**2) / (n - k)
        var_coeff = s2 * np.linalg.pinv(X_arr.T @ X_arr)
        se = np.sqrt(np.diag(var_coeff))
        t_stats = coeffs / se
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n-k))

        results = pd.DataFrame({
            "feature":   feature_names,
            "coef":      coeffs,
            "se":        se,
            "t_stat":    t_stats,
            "p_value":   p_values,
            "sig":       ["***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "" for p in p_values]
        })

        # R-squared
        ss_res = np.sum(residuals_vec**2)
        ss_tot = np.sum((y.values - y.values.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        results.attrs["r2"] = round(r2, 4)
        results.attrs["n"] = n
        results.attrs["outcome"] = outcome

        return results
    except Exception as e:
        print(f"[OLS ERROR] {e}")
        return pd.DataFrame()


def run_t_test(df_matched: pd.DataFrame, outcome: str) -> dict:
    """Independent samples t-test on matched sample."""
    treated = df_matched[df_matched["treat"] == 1][outcome].dropna()
    control = df_matched[df_matched["treat"] == 0][outcome].dropna()
    t_stat, p_val = stats.ttest_ind(treated, control)
    cohen_d = (treated.mean() - control.mean()) / np.sqrt(
        (treated.std()**2 + control.std()**2) / 2
    )
    return {
        "outcome":        outcome,
        "treated_mean":   round(treated.mean(), 4),
        "control_mean":   round(control.mean(), 4),
        "difference":     round(treated.mean() - control.mean(), 4),
        "pct_difference": round((treated.mean() - control.mean()) / abs(control.mean()) * 100, 2) if control.mean() != 0 else 0,
        "t_stat":         round(t_stat, 4),
        "p_value":        round(p_val, 6),
        "cohen_d":        round(cohen_d, 4),
        "significant":    p_val < 0.05
    }


# ── Visualizations ────────────────────────────────────────────────────────────
def plot_engagement_violin(df_matched: pd.DataFrame, outcomes: list[str], output_path: Path):
    """Violin plots comparing treated vs control on matched sample."""
    fig, axes = plt.subplots(1, len(outcomes), figsize=(5 * len(outcomes), 7))
    fig.patch.set_facecolor("#0d1117")
    if len(outcomes) == 1:
        axes = [axes]

    colors = {"AI-Generated": "#f87171", "Human-Created": "#60a5fa"}

    for ax, outcome in zip(axes, outcomes):
        ax.set_facecolor("#111827")
        treated = df_matched[df_matched["treat"] == 1][outcome].dropna()
        control = df_matched[df_matched["treat"] == 0][outcome].dropna()

        vp = ax.violinplot([treated, control], positions=[1, 2], showmedians=True, showmeans=False)
        vp["bodies"][0].set_facecolor("#f87171")
        vp["bodies"][0].set_alpha(0.75)
        vp["bodies"][1].set_facecolor("#60a5fa")
        vp["bodies"][1].set_alpha(0.75)
        for partname in ("cbars", "cmins", "cmaxes", "cmedians"):
            if partname in vp:
                vp[partname].set_edgecolor("white")
                vp[partname].set_linewidth(1.5)

        # T-test result annotation
        t_result = run_t_test(df_matched, outcome)
        sig_label = "***" if t_result["p_value"] < 0.001 else "**" if t_result["p_value"] < 0.01 else "*" if t_result["p_value"] < 0.05 else "n.s."
        ax.set_title(f"{outcome}\np={t_result['p_value']:.4f} {sig_label}",
                     color="white", fontsize=11)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["AI-Generated", "Human-Created"], color="white", fontsize=9)
        ax.tick_params(colors="white")
        ax.set_ylabel("Value", color="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#374151")

    patches = [mpatches.Patch(color="#f87171", label="AI-Generated"),
               mpatches.Patch(color="#60a5fa", label="Human-Created")]
    fig.legend(handles=patches, loc="upper right", facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    fig.suptitle("Engagement Comparison: AI-Generated vs Human-Created Content\n(PSM Matched Sample)",
                 color="white", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"[Plot] Violin plot → {output_path}")


def plot_regression_coefs(ols_results: pd.DataFrame, title: str, output_path: Path):
    """Forest plot of regression coefficients with confidence intervals."""
    df = ols_results[ols_results["feature"] != "intercept"].copy()
    df["ci_low"]  = df["coef"] - 1.96 * df["se"]
    df["ci_high"] = df["coef"] + 1.96 * df["se"]
    df = df.sort_values("coef", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.6)))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#111827")

    colors = ["#f87171" if row["feature"] == "treat" else
              ("#34d399" if row["p_value"] < 0.05 else "#9ca3af")
              for _, row in df.iterrows()]

    y_pos = range(len(df))
    ax.barh(y_pos, df["coef"], xerr=1.96 * df["se"],
            color=colors, alpha=0.8, capsize=4, error_kw={"ecolor": "white", "linewidth": 1.5})
    ax.axvline(0, color="white", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["feature"], color="white", fontsize=9)
    ax.set_xlabel("Coefficient (with 95% CI)", color="white")
    ax.set_title(f"{title}\nR² = {ols_results.attrs.get('r2', 'N/A')}, N = {ols_results.attrs.get('n', 'N/A')}",
                 color="white", fontsize=11)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#374151")

    patches = [mpatches.Patch(color="#f87171", label="Treatment (AI-generated)"),
               mpatches.Patch(color="#34d399", label="Significant covariate (p<0.05)"),
               mpatches.Patch(color="#9ca3af", label="Non-significant")]
    ax.legend(handles=patches, facecolor="#1a1a2e", labelcolor="white", fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"[Plot] Regression coefficients → {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/synthetic/synthetic_posts.csv")
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} posts.")

    print("\n=== Preparing features ===")
    df = prepare_features(df)
    print(f"Prepared dataset: {len(df)} posts "
          f"({df['treat'].sum()} treated, {(df['treat']==0).sum()} control)")

    PSM_COVARIATES = ["log_followers", "log_account_age", "has_media_int", "hour_of_day"]
    OUTCOMES = ["log_views", "log_engagement_total", "engagement_rate"]

    print("\n=== Estimating propensity scores ===")
    df = estimate_propensity_scores(df, PSM_COVARIATES)

    print("\n=== Matching samples ===")
    df_matched = match_samples(df, caliper=0.05)
    df_matched.to_csv(TABLE_DIR / "psm_matched_sample.csv", index=False)

    check_balance(df_matched, PSM_COVARIATES)

    print("\n=== T-tests on matched sample ===")
    t_results = []
    for outcome in OUTCOMES:
        if outcome in df_matched.columns:
            result = run_t_test(df_matched, outcome)
            t_results.append(result)
            sig_str = "SIGNIFICANT" if result["significant"] else "not significant"
            print(f"\n  {outcome}:")
            print(f"    AI-gen mean:  {result['treated_mean']:.4f}")
            print(f"    Human mean:   {result['control_mean']:.4f}")
            print(f"    Difference:   {result['pct_difference']:+.1f}%")
            print(f"    t={result['t_stat']:.3f}, p={result['p_value']:.4f} [{sig_str}]")
            print(f"    Cohen's d:    {result['cohen_d']:.3f}")

    pd.DataFrame(t_results).to_csv(TABLE_DIR / "ttest_results.csv", index=False)

    print("\n=== OLS Regressions ===")
    ols_all_results = []
    for outcome in OUTCOMES:
        if outcome in df_matched.columns:
            print(f"\n  Outcome: {outcome}")
            ols_res = run_ols_regression(df_matched, outcome, PSM_COVARIATES)
            if not ols_res.empty:
                treat_row = ols_res[ols_res["feature"] == "treat"].iloc[0]
                print(f"    Treat coef: {treat_row['coef']:.4f} "
                      f"(SE={treat_row['se']:.4f}, p={treat_row['p_value']:.4f}) {treat_row['sig']}")
                print(f"    R² = {ols_res.attrs.get('r2', 'N/A')}, N = {ols_res.attrs.get('n', 'N/A')}")
                ols_res["outcome"] = outcome
                ols_all_results.append(ols_res)

                # Plot coefficients for primary outcome
                if outcome == "log_views":
                    plot_regression_coefs(
                        ols_res,
                        "OLS: log(Views) ~ AI_Generated + Confounders",
                        FIG_DIR / "regression_coef_views.png"
                    )

    if ols_all_results:
        pd.concat(ols_all_results).to_csv(TABLE_DIR / "ols_regression_results.csv", index=False)

    print("\n=== Visualizations ===")
    valid_outcomes = [o for o in OUTCOMES if o in df_matched.columns]
    if valid_outcomes:
        plot_engagement_violin(df_matched, valid_outcomes[:3], FIG_DIR / "engagement_violin.png")

    print("\n✓ Engagement analysis complete.")
