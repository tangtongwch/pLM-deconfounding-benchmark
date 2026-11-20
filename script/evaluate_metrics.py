import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

ASPECTS = ["BP", "MF", "CC"]

DEFAULT_METHODS = [
    "naive",
    "blast",
    "esm2_8m",
    "esm2_150m",
    "esm2_650m",
    "esm2_3b",
    "protbert",
    "prott5",
]

BASELINE_METHODS = ["naive", "blast"]

METHOD_DISPLAY_NAMES = {
    "naive": "Naive",
    "blast": "BLAST",
    "esm2_8m": "ESM-2 (8M)",
    "esm2_150m": "ESM-2 (150M)",
    "esm2_650m": "ESM-2 (650M)",
    "esm2_3b": "ESM-2 (3B)",
    "protbert": "ProtBERT",
    "prott5": "ProtT5",
}

NAMESPACE_TO_ASPECT = {
    "biological_process": "BP",
    "molecular_function": "MF",
    "cellular_component": "CC",
}

THRESHOLDS = np.round(np.arange(0.00, 1.01, 0.01), 2)
CI_ALPHA = 0.05


@dataclass
class Paths:
    root: Path
    targets: Path = field(init=False)
    ontology: Path = field(init=False)
    predictions: Path = field(init=False)
    out_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        processed = self.root / "processed"
        self.targets = processed / "targets"
        self.ontology = processed / "ontology"
        self.predictions = processed / "predictions"
        self.out_dir = processed / "metrics"


def load_ia_table(paths: Paths) -> pd.DataFrame:
    path = paths.ontology / "go_term_metrics_2020.tsv"
    df = pd.read_csv(path, sep="\t")
    df["Aspect"] = df["Namespace"].map(NAMESPACE_TO_ASPECT)
    df = df.dropna(subset=["Aspect"])
    return df[["GO_ID", "Aspect", "IA"]]


def load_ground_truth(paths: Paths, aspect: str) -> Dict[str, set]:
    path = paths.targets / f"ground_truth_{aspect}_propagated.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")
    df = pd.read_csv(path, sep="\t", header=None, names=["Protein", "GO_ID", "Aspect"])
    return df.groupby("Protein")["GO_ID"].apply(set).to_dict()


def load_benchmark_proteins(paths: Paths, aspect: str) -> List[str]:
    path = paths.targets / f"targets_NK_{aspect}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Benchmark target list not found: {path}")
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_predictions(paths: Paths, method: str, aspect: str,
                      benchmark_proteins: List[str]) -> pd.DataFrame:
    path = paths.predictions / f"predictions_{method}.tsv"
    if not path.exists():
        logging.warning("Prediction file not found for method '%s': %s", method, path)
        return pd.DataFrame(columns=["Protein", "GO_ID", "Score"])

    df = pd.read_csv(path, sep="\t")
    colmap = {"aspect": "Aspect", "protein": "Protein",
              "term": "GO_ID", "go_id": "GO_ID", "score": "Score"}
    df = df.rename(columns={c: colmap[c.lower()] for c in df.columns if c.lower() in colmap})

    required = {"Protein", "GO_ID", "Score", "Aspect"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prediction file {path} is missing required columns: {missing}")

    df = df[df["Aspect"] == aspect]
    df = df[df["Protein"].isin(benchmark_proteins)]
    return df[["Protein", "GO_ID", "Score"]].copy()


Matrices = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def precompute_protein_matrices(pred_df: pd.DataFrame, gt_dict: Dict[str, set],
                                 ia_map: Dict[str, float], benchmark_proteins: List[str],
                                 thresholds: np.ndarray) -> Matrices:
    n_p = len(benchmark_proteins)
    n_t = len(thresholds)

    N_PRED = np.zeros((n_p, n_t))
    N_CORRECT = np.zeros((n_p, n_t))
    IA_PRED = np.zeros((n_p, n_t))
    IA_CORRECT = np.zeros((n_p, n_t))
    GT_SIZE = np.zeros(n_p)
    GT_IA_TOTAL = np.zeros(n_p)

    grouped = {k: v for k, v in pred_df.groupby("Protein")} if len(pred_df) > 0 else {}

    for i, p in enumerate(benchmark_proteins):
        gt_terms = gt_dict.get(p, set())
        GT_SIZE[i] = len(gt_terms)
        GT_IA_TOTAL[i] = sum(ia_map.get(t, 0.0) for t in gt_terms)

        sub = grouped.get(p)
        if sub is None or len(sub) == 0:
            continue

        terms = sub["GO_ID"].to_numpy()
        scores = sub["Score"].to_numpy(dtype=float)
        correct = np.array([1.0 if t in gt_terms else 0.0 for t in terms], dtype=float)
        ia_vals = np.array([ia_map.get(t, 0.0) for t in terms], dtype=float)

        order = np.argsort(scores, kind="mergesort")
        scores_sorted = scores[order]
        correct_sorted = correct[order]
        ia_sorted = ia_vals[order]
        ia_correct_sorted = ia_sorted * correct_sorted

        m = len(scores_sorted)
        suffix_n = np.arange(m, -1, -1, dtype=float)
        suffix_correct = np.concatenate([np.cumsum(correct_sorted[::-1])[::-1], [0.0]])
        suffix_ia_pred = np.concatenate([np.cumsum(ia_sorted[::-1])[::-1], [0.0]])
        suffix_ia_correct = np.concatenate([np.cumsum(ia_correct_sorted[::-1])[::-1], [0.0]])

        idx_arr = np.searchsorted(scores_sorted, thresholds, side="left")

        N_PRED[i, :] = suffix_n[idx_arr]
        N_CORRECT[i, :] = suffix_correct[idx_arr]
        IA_PRED[i, :] = suffix_ia_pred[idx_arr]
        IA_CORRECT[i, :] = suffix_ia_correct[idx_arr]

    return N_PRED, N_CORRECT, IA_PRED, IA_CORRECT, GT_SIZE, GT_IA_TOTAL


def curve_from_matrices(N_PRED: np.ndarray, N_CORRECT: np.ndarray, IA_PRED: np.ndarray,
                         IA_CORRECT: np.ndarray, GT_SIZE: np.ndarray, GT_IA_TOTAL: np.ndarray,
                         thresholds: np.ndarray,
                         protein_idx: Optional[np.ndarray] = None) -> pd.DataFrame:
    if protein_idx is not None:
        N_PRED = N_PRED[protein_idx]
        N_CORRECT = N_CORRECT[protein_idx]
        IA_PRED = IA_PRED[protein_idx]
        IA_CORRECT = IA_CORRECT[protein_idx]
        GT_SIZE = GT_SIZE[protein_idx]
        GT_IA_TOTAL = GT_IA_TOTAL[protein_idx]

    pred_mask = N_PRED > 0
    m_t = pred_mask.sum(axis=0)

    safe_n_pred = np.where(N_PRED == 0, 1.0, N_PRED)
    precision_i = np.where(pred_mask, N_CORRECT / safe_n_pred, 0.0)
    safe_m = np.where(m_t == 0, 1, m_t)
    pr = precision_i.sum(axis=0) / safe_m

    safe_gt = np.where(GT_SIZE == 0, 1.0, GT_SIZE)
    recall_i = N_CORRECT / safe_gt[:, None]
    rc = recall_i.mean(axis=0)

    denom = pr + rc
    f = np.where(denom > 0, 2 * pr * rc / np.where(denom == 0, 1.0, denom), 0.0)

    ru = (GT_IA_TOTAL[:, None] - IA_CORRECT).mean(axis=0)
    mi = (IA_PRED - IA_CORRECT).mean(axis=0)
    s = np.sqrt(ru ** 2 + mi ** 2)

    return pd.DataFrame({
        "threshold": thresholds,
        "precision": pr, "recall": rc, "f": f,
        "ru": ru, "mi": mi, "s": s,
        "n_pred_proteins": m_t,
    })


def summarize_curve(curve: pd.DataFrame) -> dict:
    if curve.empty:
        return dict(fmax=np.nan, t_fmax=np.nan, smin=np.nan, t_smin=np.nan)
    i_f = curve["f"].idxmax()
    i_s = curve["s"].idxmin()
    return dict(
        fmax=curve.loc[i_f, "f"], t_fmax=curve.loc[i_f, "threshold"],
        smin=curve.loc[i_s, "s"], t_smin=curve.loc[i_s, "threshold"],
    )


def compute_aupr(curve: pd.DataFrame) -> float:
    if curve.empty:
        return np.nan
    df = curve[["recall", "precision"]].drop_duplicates(subset="recall").sort_values("recall")
    if len(df) < 2:
        return 0.0
    x = df["recall"].to_numpy(dtype=float)
    y = df["precision"].to_numpy(dtype=float)
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz_fn(y, x))


def compute_coverage(curve: pd.DataFrame, n_benchmark: int, t_fmax: float) -> Tuple[float, float]:
    if curve.empty or n_benchmark == 0 or pd.isna(t_fmax):
        return np.nan, np.nan

    row0 = curve.loc[curve["threshold"] == 0.00]
    cov0 = row0["n_pred_proteins"].values[0] / n_benchmark if not row0.empty else np.nan

    t_fmax_r = round(float(t_fmax), 2)
    row_t = curve.loc[curve["threshold"] == t_fmax_r]
    if row_t.empty:
        idx = (curve["threshold"] - t_fmax_r).abs().idxmin()
        row_t = curve.loc[[idx]]
    cov_topt = row_t["n_pred_proteins"].values[0] / n_benchmark

    return cov0, cov_topt


def bootstrap_ci(matrices: Matrices, thresholds: np.ndarray, n_bootstrap: int,
                  rng: np.random.Generator, alpha: float = CI_ALPHA) -> dict:
    N_PRED, N_CORRECT, IA_PRED, IA_CORRECT, GT_SIZE, GT_IA_TOTAL = matrices
    n = N_PRED.shape[0]

    fmax_samples = np.empty(n_bootstrap)
    smin_samples = np.empty(n_bootstrap)
    aupr_samples = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        curve = curve_from_matrices(N_PRED, N_CORRECT, IA_PRED, IA_CORRECT,
                                     GT_SIZE, GT_IA_TOTAL, thresholds, idx)
        fmax_samples[b] = curve["f"].max()
        smin_samples[b] = curve["s"].min()
        aupr_samples[b] = compute_aupr(curve)

    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return dict(
        fmax_ci_low=np.percentile(fmax_samples, lo), fmax_ci_high=np.percentile(fmax_samples, hi),
        smin_ci_low=np.percentile(smin_samples, lo), smin_ci_high=np.percentile(smin_samples, hi),
        aupr_ci_low=np.percentile(aupr_samples, lo), aupr_ci_high=np.percentile(aupr_samples, hi),
    )


def paired_bootstrap_test(matrices_a: Matrices, matrices_b: Matrices, thresholds: np.ndarray,
                           n_bootstrap: int, rng: np.random.Generator) -> dict:
    n = matrices_a[0].shape[0]
    fmax_diffs = np.empty(n_bootstrap)
    smin_diffs = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        curve_a = curve_from_matrices(*matrices_a, thresholds, idx)
        curve_b = curve_from_matrices(*matrices_b, thresholds, idx)
        fmax_diffs[b] = curve_a["f"].max() - curve_b["f"].max()
        smin_diffs[b] = curve_a["s"].min() - curve_b["s"].min()

    def two_sided_p(diffs: np.ndarray) -> float:
        p_pos = np.mean(diffs <= 0)
        p_neg = np.mean(diffs >= 0)
        return float(min(1.0, 2 * min(p_pos, p_neg)))

    return dict(
        delta_fmax=float(fmax_diffs.mean()),
        fmax_ci_low=float(np.percentile(fmax_diffs, 2.5)),
        fmax_ci_high=float(np.percentile(fmax_diffs, 97.5)),
        p_fmax=two_sided_p(fmax_diffs),
        delta_smin=float(smin_diffs.mean()),
        smin_ci_low=float(np.percentile(smin_diffs, 2.5)),
        smin_ci_high=float(np.percentile(smin_diffs, 97.5)),
        p_smin=two_sided_p(smin_diffs),
    )


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    fdr = ranked * n / (np.arange(n) + 1)
    fdr = np.minimum.accumulate(fdr[::-1])[::-1]
    fdr = np.clip(fdr, 0, 1)
    out = np.empty(n)
    out[order] = fdr
    return out


def evaluate_method_aspect(method: str, aspect: str, paths: Paths, ia_map: Dict[str, float],
                            gt_dict: Dict[str, set], benchmark_proteins: List[str],
                            thresholds: np.ndarray, n_bootstrap: int,
                            rng: np.random.Generator) -> dict:
    pred_df = load_predictions(paths, method, aspect, benchmark_proteins)
    matrices = precompute_protein_matrices(pred_df, gt_dict, ia_map, benchmark_proteins, thresholds)

    curve = curve_from_matrices(*matrices, thresholds, None)
    curve["method"] = method
    curve["aspect"] = aspect

    summ = summarize_curve(curve)
    aupr = compute_aupr(curve)

    n_benchmark = len(benchmark_proteins)
    cov0, cov_topt = compute_coverage(curve, n_benchmark, summ["t_fmax"])

    boot = bootstrap_ci(matrices, thresholds, n_bootstrap, rng)

    overall_row = dict(
        method=method, method_display=METHOD_DISPLAY_NAMES.get(method, method),
        aspect=aspect, n_benchmark=n_benchmark,
        fmax=summ["fmax"], t_fmax=summ["t_fmax"],
        fmax_ci_low=boot["fmax_ci_low"], fmax_ci_high=boot["fmax_ci_high"],
        smin=summ["smin"], t_smin=summ["t_smin"],
        smin_ci_low=boot["smin_ci_low"], smin_ci_high=boot["smin_ci_high"],
        aupr=aupr, aupr_ci_low=boot["aupr_ci_low"], aupr_ci_high=boot["aupr_ci_high"],
        coverage_0=cov0, coverage_topt=cov_topt,
    )

    return dict(overall_row=overall_row, curve=curve, matrices=matrices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CAFA-style protein function prediction benchmarks."
    )
    parser.add_argument("--root", type=str,
                         default=str(Path(__file__).resolve().parent.parent),
                         help="Project root containing the 'processed/' directory.")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                         help="Prediction methods to evaluate.")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                         help="Number of bootstrap resamples for CIs and significance tests.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--out-dir", type=str, default=None,
                         help="Override the default output directory (processed/metrics).")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    args = parse_args()
    setup_logging()

    rng = np.random.default_rng(args.seed)
    paths = Paths(Path(args.root))
    if args.out_dir:
        paths.out_dir = Path(args.out_dir)
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading GO term information-accretion (IA) table...")
    ia_table = load_ia_table(paths)

    overall_rows, all_curves, significance_rows = [], [], []

    for aspect in ASPECTS:
        logging.info("=== Aspect: %s ===", aspect)

        ia_map = dict(zip(
            ia_table.loc[ia_table["Aspect"] == aspect, "GO_ID"],
            ia_table.loc[ia_table["Aspect"] == aspect, "IA"],
        ))

        gt_dict = load_ground_truth(paths, aspect)
        benchmark_proteins = load_benchmark_proteins(paths, aspect)
        benchmark_proteins = [p for p in benchmark_proteins if p in gt_dict]
        n_benchmark = len(benchmark_proteins)
        logging.info("%d benchmark proteins with ground truth for %s", n_benchmark, aspect)

        aspect_matrices: Dict[str, Matrices] = {}

        for method in tqdm(args.methods, desc=f"Evaluating methods ({aspect})"):
            result = evaluate_method_aspect(
                method, aspect, paths, ia_map, gt_dict,
                benchmark_proteins, THRESHOLDS, args.n_bootstrap, rng,
            )

            overall_rows.append(result["overall_row"])
            all_curves.append(result["curve"])
            aspect_matrices[method] = result["matrices"]

            r = result["overall_row"]
            logging.info(
                "[%-12s] Fmax=%.4f [%.4f, %.4f]  Smin=%.4f [%.4f, %.4f]  "
                "AUPR=%.4f  cov0=%.3f  cov_t*=%.3f",
                method, r["fmax"], r["fmax_ci_low"], r["fmax_ci_high"],
                r["smin"], r["smin_ci_low"], r["smin_ci_high"],
                r["aupr"], r["coverage_0"], r["coverage_topt"],
            )

        logging.info("Running paired bootstrap significance tests vs. baselines for %s...", aspect)
        for method in args.methods:
            if method in BASELINE_METHODS:
                continue
            for baseline in BASELINE_METHODS:
                if baseline not in aspect_matrices:
                    continue
                test = paired_bootstrap_test(
                    aspect_matrices[method], aspect_matrices[baseline],
                    THRESHOLDS, args.n_bootstrap, rng,
                )
                significance_rows.append(dict(aspect=aspect, method=method, baseline=baseline, **test))

    sig_df = pd.DataFrame(significance_rows)
    if not sig_df.empty:
        sig_df["p_fmax_fdr"] = bh_fdr(sig_df["p_fmax"].to_numpy())
        sig_df["p_smin_fdr"] = bh_fdr(sig_df["p_smin"].to_numpy())

    overall_df = pd.DataFrame(overall_rows)

    logging.info("Writing outputs to %s", paths.out_dir)
    overall_df.to_csv(paths.out_dir / "fmax_smin_overall.tsv", sep="\t", index=False)
    pd.concat(all_curves, ignore_index=True).to_csv(paths.out_dir / "pr_curves.tsv", sep="\t", index=False)
    sig_df.to_csv(paths.out_dir / "significance_tests.tsv", sep="\t", index=False)

    config = dict(
        seed=args.seed, n_bootstrap=args.n_bootstrap,
        methods=args.methods, thresholds=THRESHOLDS.tolist(),
    )
    with open(paths.out_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    logging.info("Done. All results written to %s", paths.out_dir)


if __name__ == "__main__":
    main()