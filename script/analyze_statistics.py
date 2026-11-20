import os
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "processed" / "metrics"
TARGETS_DIR = ROOT / "processed" / "targets"
ONTOLOGY_DIR = ROOT / "processed" / "ontology"
HOMOLOGY_DIR = ROOT / "processed" / "homology"
PPI_DIR = ROOT / "processed" / "ppi"
PRED_DIR = ROOT / "processed" / "predictions"

OUT_DIR.mkdir(parents=True, exist_ok=True)

ASPECTS = ["BP", "MF", "CC"]
METHODS = [
    "naive",
    "blast",
    "esm2_8m",
    "esm2_150m",
    "esm2_650m",
    "esm2_3b",
    "protbert",
    "prott5",
]
THRESHOLDS = np.round(np.arange(0.00, 1.01, 0.01), 2)
N_BOOTSTRAP = 1000
CI_ALPHA = 0.05
RNG = np.random.default_rng(42)

trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


def load_ground_truth(aspect: str) -> dict:
    gt_file = TARGETS_DIR / f"ground_truth_{aspect}_propagated.tsv"
    df = pd.read_csv(
        gt_file, sep="\t", header=None, names=["Protein", "GO_ID", "Aspect"]
    )
    return df.groupby("Protein")["GO_ID"].apply(set).to_dict()


def load_ia_weights(aspect: str) -> dict:
    ia_file = ONTOLOGY_DIR / "go_term_metrics_2020.tsv"
    df = pd.read_csv(ia_file, sep="\t")
    ns_map = {
        "biological_process": "BP",
        "molecular_function": "MF",
        "cellular_component": "CC",
    }
    df["Aspect"] = df["Namespace"].map(ns_map)
    sub = df[df["Aspect"] == aspect]
    return dict(zip(sub["GO_ID"], sub["IA"]))


def load_protein_metadata() -> pd.DataFrame:
    homo_df = pd.read_csv(HOMOLOGY_DIR / "nk_homology_bins.tsv", sep="\t")
    ppi_df = pd.read_csv(PPI_DIR / "nk_targets_ppi_degrees.tsv", sep="\t")

    meta = pd.merge(
        homo_df,
        ppi_df[["Protein", "Degree", "Degree_Bin", "Mapped_ENSP"]],
        on="Protein",
        how="left",
    )

    meta.loc[meta["Degree_Bin"] == "Unmapped", "Degree"] = np.nan
    meta["log_degree"] = np.log1p(meta["Degree"])
    meta["pident_clean"] = meta["pident"].fillna(0.0)
    return meta


def precompute_matrices(
    pred_df: pd.DataFrame, gt_dict: dict, ia_map: dict, proteins: list
):
    n_p = len(proteins)
    n_t = len(THRESHOLDS)

    n_pred = np.zeros((n_p, n_t), dtype=np.float32)
    n_correct = np.zeros((n_p, n_t), dtype=np.float32)
    ia_pred = np.zeros((n_p, n_t), dtype=np.float32)
    ia_correct = np.zeros((n_p, n_t), dtype=np.float32)
    gt_size = np.zeros(n_p, dtype=np.float32)
    gt_ia_total = np.zeros(n_p, dtype=np.float32)

    grouped = (
        {k: v for k, v in pred_df.groupby("Protein")}
        if not pred_df.empty
        else {}
    )

    for i, p in enumerate(proteins):
        gt_terms = gt_dict.get(p, set())
        gt_size[i] = len(gt_terms)
        gt_ia_total[i] = sum(ia_map.get(t, 0.0) for t in gt_terms)

        sub = grouped.get(p)
        if sub is None or len(sub) == 0:
            continue

        terms = sub["GO_ID"].to_numpy()
        scores = sub["Score"].to_numpy(dtype=float)
        correct = np.array(
            [1.0 if t in gt_terms else 0.0 for t in terms], dtype=np.float32
        )
        ia_vals = np.array(
            [ia_map.get(t, 0.0) for t in terms], dtype=np.float32
        )

        order = np.argsort(scores, kind="mergesort")
        scores_sorted = scores[order]
        correct_sorted = correct[order]
        ia_sorted = ia_vals[order]
        ia_correct_sorted = ia_sorted * correct_sorted

        m = len(scores_sorted)
        suffix_n = np.arange(m, -1, -1, dtype=np.float32)
        suffix_c = np.concatenate([np.cumsum(correct_sorted[::-1])[::-1], [0.0]])
        suffix_ia_p = np.concatenate([np.cumsum(ia_sorted[::-1])[::-1], [0.0]])
        suffix_ia_c = np.concatenate(
            [np.cumsum(ia_correct_sorted[::-1])[::-1], [0.0]]
        )

        idx_arr = np.searchsorted(scores_sorted, THRESHOLDS, side="left")

        n_pred[i, :] = suffix_n[idx_arr]
        n_correct[i, :] = suffix_c[idx_arr]
        ia_pred[i, :] = suffix_ia_p[idx_arr]
        ia_correct[i, :] = suffix_ia_c[idx_arr]

    return n_pred, n_correct, ia_pred, ia_correct, gt_size, gt_ia_total


def compute_curve_vectorized(matrices, subset_idx=None):
    n_p, n_c, ia_p, ia_c, gt_s, gt_ia = matrices
    if subset_idx is not None:
        n_p = n_p[subset_idx]
        n_c = n_c[subset_idx]
        ia_p = ia_p[subset_idx]
        ia_c = ia_c[subset_idx]
        gt_s = gt_s[subset_idx]
        gt_ia = gt_ia[subset_idx]

    pred_mask = n_p > 0
    m_t = pred_mask.sum(axis=0)

    safe_np = np.where(n_p == 0, 1.0, n_p)
    prec_per_prot = np.where(pred_mask, n_c / safe_np, 0.0)
    safe_m = np.where(m_t == 0, 1, m_t)
    pr = prec_per_prot.sum(axis=0) / safe_m

    safe_gt = np.where(gt_s == 0, 1.0, gt_s)
    rec_per_prot = n_c / safe_gt[:, None]
    rc = rec_per_prot.mean(axis=0)

    denom = pr + rc
    f_scores = np.where(
        denom > 0, (2.0 * pr * rc) / np.where(denom == 0, 1.0, denom), 0.0
    )

    ru = (gt_ia[:, None] - ia_c).mean(axis=0)
    mi = (ia_p - ia_c).mean(axis=0)
    s_scores = np.sqrt(ru**2 + mi**2)

    return f_scores, s_scores, pr, rc


def compute_aupr(pr: np.ndarray, rc: np.ndarray) -> float:
    order = np.argsort(rc)
    x = rc[order]
    y = pr[order]
    _, unique_idx = np.unique(x, return_index=True)
    if len(unique_idx) < 2:
        return 0.0
    return float(trapezoid(y[unique_idx], x[unique_idx]))


def bootstrap_metric_ci(matrices, subset_idx, n_boot=N_BOOTSTRAP):
    n_samples = len(subset_idx)
    if n_samples < 5:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    fmax_arr = np.empty(n_boot, dtype=np.float32)
    smin_arr = np.empty(n_boot, dtype=np.float32)
    aupr_arr = np.empty(n_boot, dtype=np.float32)

    for b in range(n_boot):
        sample_idx = RNG.choice(subset_idx, size=n_samples, replace=True)
        f_vals, s_vals, pr, rc = compute_curve_vectorized(matrices, sample_idx)
        fmax_arr[b] = np.max(f_vals)
        smin_arr[b] = np.min(s_vals)
        aupr_arr[b] = compute_aupr(pr, rc)

    lo, hi = 100 * (CI_ALPHA / 2), 100 * (1 - CI_ALPHA / 2)
    return (
        np.percentile(fmax_arr, lo),
        np.percentile(fmax_arr, hi),
        np.percentile(smin_arr, lo),
        np.percentile(smin_arr, hi),
        np.percentile(aupr_arr, lo),
        np.percentile(aupr_arr, hi),
    )


def extract_protein_level_performance(matrices, proteins, opt_t_idx):
    n_p, n_c, _, _, gt_s, _ = matrices
    pred_t = n_p[:, opt_t_idx]
    corr_t = n_c[:, opt_t_idx]
    gt_t = gt_s

    prec = np.where(pred_t > 0, corr_t / np.maximum(pred_t, 1.0), 0.0)
    rec = np.where(gt_t > 0, corr_t / np.maximum(gt_t, 1.0), 0.0)
    denom = prec + rec
    f1 = np.where(denom > 0, (2.0 * prec * rec) / np.maximum(denom, 1e-9), 0.0)

    return pd.DataFrame(
        {
            "Protein": proteins,
            "n_pred": pred_t,
            "n_correct": corr_t,
            "gt_size": gt_t,
            "prot_precision": prec,
            "prot_recall": rec,
            "prot_f1": f1,
        }
    )


def run_stratified_evaluation(aspect_matrices, meta_df, overall_fmax_df):
    results = []

    for aspect in ASPECTS:
        sub_meta = meta_df[meta_df[f"In_{aspect}"] == 1].copy()
        prot_list = sub_meta["Protein"].tolist()
        prot_to_idx = {p: i for i, p in enumerate(prot_list)}

        stratifications = {
            "Homology": ("homology_bin", sub_meta["homology_bin"].dropna().unique()),
            "PPI_Degree": ("Degree_Bin", sub_meta["Degree_Bin"].dropna().unique()),
        }

        for method in METHODS:
            matrices = aspect_matrices[aspect][method]

            for strat_name, (col, categories) in stratifications.items():
                for cat in categories:
                    target_prots = sub_meta[sub_meta[col] == cat]["Protein"].tolist()
                    subset_idx = np.array(
                        [prot_to_idx[p] for p in target_prots if p in prot_to_idx]
                    )
                    n_subset = len(subset_idx)

                    if n_subset < 3:
                        continue

                    f_vals, s_vals, pr, rc = compute_curve_vectorized(
                        matrices, subset_idx
                    )
                    fmax_val = float(np.max(f_vals))
                    smin_val = float(np.min(s_vals))
                    aupr_val = compute_aupr(pr, rc)

                    f_lo, f_hi, s_lo, s_hi, a_lo, a_hi = bootstrap_metric_ci(
                        matrices, subset_idx
                    )

                    results.append(
                        {
                            "Aspect": aspect,
                            "Method": method,
                            "Stratification": strat_name,
                            "Group": cat,
                            "N_Proteins": n_subset,
                            "Fmax": fmax_val,
                            "Fmax_CI_Low": f_lo,
                            "Fmax_CI_High": f_hi,
                            "Smin": smin_val,
                            "Smin_CI_Low": s_lo,
                            "Smin_CI_High": s_hi,
                            "AUPR": aupr_val,
                            "AUPR_CI_Low": a_lo,
                            "AUPR_CI_High": a_hi,
                        }
                    )

    strat_df = pd.DataFrame(results)
    strat_df.to_csv(OUT_DIR / "stratified_performance.tsv", sep="\t", index=False)
    return strat_df


def run_regression_analysis(protein_perf_df, meta_df):
    df_reg = pd.merge(protein_perf_df, meta_df, on="Protein", how="inner")

    blast_ref = df_reg[df_reg["Method"] == "blast"][
        ["Protein", "Aspect", "prot_f1"]
    ].rename(columns={"prot_f1": "blast_prot_f1"})
    df_reg = pd.merge(df_reg, blast_ref, on=["Protein", "Aspect"], how="left")
    df_reg["f1_gain_over_blast"] = (
        df_reg["prot_f1"] - df_reg["blast_prot_f1"].fillna(0.0)
    )

    model_scales = {
        "esm2_8m": 8,
        "esm2_150m": 150,
        "esm2_650m": 650,
        "esm2_3b": 3000,
        "protbert": 420,
        "prott5": 3000,
    }
    df_reg["log_params"] = df_reg["Method"].map(
        lambda m: np.log10(model_scales[m]) if m in model_scales else np.nan
    )

    regression_rows = []

    plm_list = [m for m in METHODS if m not in ["naive", "blast"]]
    for asp in ASPECTS:
        for m in plm_list:
            sub = df_reg[
                (df_reg["Aspect"] == asp) & (df_reg["Method"] == m)
            ].dropna(subset=["prot_f1", "pident_clean", "log_degree"])
            if len(sub) < 30:
                continue

            ols = smf.ols("prot_f1 ~ pident_clean + log_degree", data=sub).fit()
            for var in ["pident_clean", "log_degree"]:
                regression_rows.append(
                    {
                        "Aspect": asp,
                        "Model_Scope": m,
                        "Formula": "prot_f1 ~ pident_clean + log_degree",
                        "Covariate": var,
                        "Coef": ols.params.get(var, np.nan),
                        "Std_Err": ols.bse.get(var, np.nan),
                        "t_val": ols.tvalues.get(var, np.nan),
                        "p_value": ols.pvalues.get(var, np.nan),
                        "R_squared": ols.rsquared,
                        "N_obs": int(ols.nobs),
                    }
                )

    orphan_sub = df_reg[
        (
            df_reg["homology_bin"].isin(
                ["Orphan", "Remote homolog", "Twilight zone"]
            )
        )
        & (df_reg["Method"].isin(plm_list))
    ].dropna(subset=["f1_gain_over_blast", "log_degree"])

    if not orphan_sub.empty:
        ols_orphan = smf.ols(
            "f1_gain_over_blast ~ C(Method) + log_degree + C(Aspect)",
            data=orphan_sub,
        ).fit()
        for var in ols_orphan.params.index:
            regression_rows.append(
                {
                    "Aspect": "ALL_DARK",
                    "Model_Scope": "PLMs_vs_BLAST",
                    "Formula": "f1_gain_over_blast ~ C(Method) + log_degree + C(Aspect)",
                    "Covariate": var,
                    "Coef": ols_orphan.params[var],
                    "Std_Err": ols_orphan.bse[var],
                    "t_val": ols_orphan.tvalues[var],
                    "p_value": ols_orphan.pvalues[var],
                    "R_squared": ols_orphan.rsquared,
                    "N_obs": int(ols_orphan.nobs),
                }
            )

    esm_sub = df_reg[df_reg["Method"].str.startswith("esm2_")].dropna(
        subset=["prot_f1", "log_params", "pident_clean", "log_degree"]
    )

    if not esm_sub.empty:
        ols_scale = smf.ols(
            "prot_f1 ~ log_params + pident_clean + log_degree + C(Aspect)",
            data=esm_sub,
        ).fit(cov_type="cluster", cov_kwds={"groups": esm_sub["Protein"]})
        ols_interact = smf.ols(
            "prot_f1 ~ log_params * C(Aspect) + pident_clean + log_degree",
            data=esm_sub,
        ).fit(cov_type="cluster", cov_kwds={"groups": esm_sub["Protein"]})
        esm_sub = esm_sub.copy()
        esm_sub["log_params_sq"] = esm_sub["log_params"] ** 2
        ols_quad = smf.ols(
            "prot_f1 ~ log_params + log_params_sq + C(Aspect) + pident_clean + log_degree",
            data=esm_sub,
        ).fit(cov_type="cluster", cov_kwds={"groups": esm_sub["Protein"]})
        macro_df = (
            esm_sub.groupby(["Method", "Aspect", "log_params"])["prot_f1"]
            .mean()
            .reset_index()
        )
        ols_macro = smf.ols(
            "prot_f1 ~ log_params + C(Aspect)", data=macro_df
        ).fit()

        models_to_record = [
            ("ESM2_Linear_Cluster", ols_scale, ols_scale.nobs),
            ("ESM2_ScaleXAspect_Cluster", ols_interact, ols_interact.nobs),
            ("ESM2_Quadratic_Cluster", ols_quad, ols_quad.nobs),
            ("ESM2_Macro_N12", ols_macro, ols_macro.nobs),
        ]

        for label, model, n_obs in models_to_record:
            for var in model.params.index:
                regression_rows.append(
                    {
                        "Aspect": "ALL",
                        "Model_Scope": label,
                        "Formula": getattr(model.model, "formula", str(model.model)),
                        "Covariate": var,
                        "Coef": model.params[var],
                        "Std_Err": model.bse[var],
                        "t_val": model.tvalues[var],
                        "p_value": model.pvalues[var],
                        "R_squared": getattr(model, "rsquared", np.nan),
                        "N_obs": int(n_obs),
                    }
                )

    reg_df = pd.DataFrame(regression_rows)
    reg_df.to_csv(OUT_DIR / "regression_analysis_results.tsv", sep="\t", index=False)
    return reg_df


def main():
    print("Loading benchmark ground truth and annotations...")
    meta_df = load_protein_metadata()
    overall_df = pd.read_csv(OUT_DIR / "fmax_smin_overall.tsv", sep="\t")

    aspect_matrices = {asp: {} for asp in ASPECTS}
    prot_perf_records = []

    for aspect in ASPECTS:
        print(f"Processing aspect: {aspect}")
        gt_dict = load_ground_truth(aspect)
        ia_map = load_ia_weights(aspect)
        sub_meta = meta_df[meta_df[f"In_{aspect}"] == 1]
        benchmark_proteins = sub_meta["Protein"].tolist()

        for method in METHODS:
            pred_file = PRED_DIR / f"predictions_{method}.tsv"
            if not pred_file.exists():
                continue

            pred_df = pd.read_csv(pred_file, sep="\t")
            pred_df = pred_df[
                (pred_df["Aspect"] == aspect)
                & (pred_df["Protein"].isin(benchmark_proteins))
            ]

            matrices = precompute_matrices(
                pred_df, gt_dict, ia_map, benchmark_proteins
            )
            aspect_matrices[aspect][method] = matrices

            opt_row = overall_df[
                (overall_df["aspect"] == aspect)
                & (overall_df["method"] == method)
            ]
            t_opt = (
                float(opt_row["t_fmax"].values[0])
                if not opt_row.empty
                else 0.50
            )
            t_idx = int(np.argmin(np.abs(THRESHOLDS - t_opt)))

            df_prot = extract_protein_level_performance(
                matrices, benchmark_proteins, t_idx
            )
            df_prot["Method"] = method
            df_prot["Aspect"] = aspect
            df_prot["t_opt"] = t_opt
            prot_perf_records.append(df_prot)

    all_prot_perf = pd.concat(prot_perf_records, ignore_index=True)
    all_prot_perf.to_csv(
        OUT_DIR / "protein_level_metrics.tsv", sep="\t", index=False
    )

    print("Running stratified evaluation with bootstrap CIs...")
    run_stratified_evaluation(aspect_matrices, meta_df, overall_df)

    print("Fitting statistical regression models...")
    run_regression_analysis(all_prot_perf, meta_df)

    print(f"Analysis completed successfully. Results saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()