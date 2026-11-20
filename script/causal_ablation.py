import os
import gzip
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "processed" / "causal_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ASPECTS = ["BP", "MF", "CC"]

df_metrics = pd.read_csv(ROOT / "processed" / "metrics" / "protein_level_metrics.tsv", sep="\t")
df_ppi = pd.read_csv(ROOT / "processed" / "ppi" / "nk_targets_ppi_degrees.tsv", sep="\t")
df_ppi["Degree"] = pd.to_numeric(df_ppi["Degree"], errors="coerce")
df_homo = pd.read_csv(ROOT / "processed" / "homology" / "nk_homology_bins.tsv", sep="\t")
df_homo["bitscore"] = pd.to_numeric(df_homo["bitscore"], errors="coerce").fillna(0.0)
df_homo["pident"] = pd.to_numeric(df_homo["pident"], errors="coerce").fillna(0.0)

results_semiparametric = []
for aspect in ASPECTS:
    df_metrics_asp = df_metrics[df_metrics["Aspect"] == aspect].copy()
    df_main = df_metrics_asp.merge(df_ppi[["Protein", "Mapped_ENSP", "Degree"]], on="Protein", how="inner")
    df_main = df_main.merge(df_homo[["Protein", "homology_bin", "bitscore", "pident"]], on="Protein", how="inner")
    df_valid = df_main.dropna(subset=["Degree", "prot_f1"]).copy()
    df_valid["log_degree"] = np.log1p(df_valid["Degree"])
    df_valid["log_bitscore"] = np.log1p(df_valid["bitscore"])
    print(f"[{aspect}] Valid protein-model pairs: {len(df_valid):,}, targets: {df_valid['Protein'].nunique():,}")
    for method, group in df_valid.groupby("Method"):
        if len(group) < 30:
            continue
        ols_raw = smf.ols("prot_f1 ~ log_degree", data=group).fit()
        beta_raw = ols_raw.params["log_degree"]
        p_raw = ols_raw.pvalues["log_degree"]
        r2_raw = ols_raw.rsquared
        ols_controlled = smf.ols(
            "prot_f1 ~ log_degree + C(homology_bin) + log_bitscore + gt_size", data=group
        ).fit()
        beta_controlled = ols_controlled.params["log_degree"]
        p_controlled = ols_controlled.pvalues["log_degree"]
        se_controlled = ols_controlled.bse["log_degree"]
        r2_controlled = ols_controlled.rsquared
        res_x = smf.ols("log_degree ~ C(homology_bin) + log_bitscore + gt_size", data=group).fit().resid
        res_y = smf.ols("prot_f1 ~ C(homology_bin) + log_bitscore + gt_size", data=group).fit().resid
        fwl_beta = sm.OLS(res_y, res_x).fit().params.iloc[0]
        ols_no_ppi = smf.ols("prot_f1 ~ C(homology_bin) + log_bitscore + gt_size", data=group).fit()
        partial_r2 = (
            (ols_no_ppi.ssr - ols_controlled.ssr) / ols_no_ppi.ssr if ols_no_ppi.ssr > 0 else 0.0
        )
        results_semiparametric.append({
            "Aspect": aspect,
            "Method": method,
            "N_targets": len(group),
            "Raw_Degree_Beta": beta_raw,
            "Raw_Degree_Pval": p_raw,
            "Raw_Degree_R2": r2_raw,
            "Raw_Degree_SE": ols_raw.bse["log_degree"],
            "Controlled_Degree_Beta": beta_controlled,
            "Controlled_Degree_SE": se_controlled,
            "Controlled_Degree_Pval": p_controlled,
            "Full_Model_R2": r2_controlled,
            "FWL_Residual_Beta": fwl_beta,
            "PPI_Partial_R2": partial_r2,
            "Attenuation_Pct": (1.0 - beta_controlled / beta_raw) * 100.0 if beta_raw != 0 else np.nan,
        })

df_res_decomp = pd.DataFrame(results_semiparametric).sort_values(
    ["Aspect", "Controlled_Degree_Beta"], ascending=[True, False]
)
out_fwl = OUT_DIR / "semiparametric_fwl_results.tsv"
df_res_decomp.to_csv(out_fwl, sep="\t", index=False)
print(f"Saved: {out_fwl}")

ppi_file = ROOT / "raw" / "9606.protein.physical.links.detailed.v12.0.txt.gz"
protein_to_gos_by_aspect = {}
for aspect in ASPECTS:
    gt_file = ROOT / "processed" / "targets" / f"ground_truth_{aspect}_propagated.tsv"
    df_gt = pd.read_csv(gt_file, sep="\t", header=None, names=["Protein", "GO_ID", "Aspect"])
    protein_to_gos_by_aspect[aspect] = df_gt.groupby("Protein")["GO_ID"].apply(set).to_dict()

edges = set()
all_nodes = set()
with gzip.open(ppi_file, "rt") as f:
    next(f)
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 3:
            p1, p2 = parts[0], parts[1]
            score = float(parts[-1])
            if score >= 400 and p1 != p2:
                edges.add((min(p1, p2), max(p1, p2)))
                all_nodes.add(p1)
                all_nodes.add(p2)

print(f"High-confidence physical PPIs: {len(edges):,} edges, {len(all_nodes):,} proteins")

adj = {node: set() for node in all_nodes}
for u, v in edges:
    adj[u].add(v)
    adj[v].add(u)

df_mapped = df_ppi.dropna(subset=["Mapped_ENSP"])
ensp_to_uniprot = dict(zip(df_mapped["Mapped_ENSP"], df_mapped["Protein"]))
uniprot_ensp_pairs = list(zip(df_mapped["Protein"], df_mapped["Mapped_ENSP"]))

def calc_mean_neighborhood_jaccard(current_adj, protein_to_gos):
    scores = {}
    for uniprot, ensp in uniprot_ensp_pairs:
        nbrs = current_adj.get(ensp)
        if not nbrs:
            continue
        tgt_gos = protein_to_gos.get(uniprot, set())
        if not tgt_gos:
            continue
        jaccards = []
        for nbr in nbrs:
            nbr_uniprot = ensp_to_uniprot.get(nbr)
            if nbr_uniprot and nbr_uniprot in protein_to_gos:
                nbr_gos = protein_to_gos[nbr_uniprot]
                union_len = len(tgt_gos | nbr_gos)
                if union_len > 0:
                    jaccards.append(len(tgt_gos & nbr_gos) / union_len)
        if jaccards:
            scores[uniprot] = np.mean(jaccards)
    return scores

obs_mean_homophily, obs_n_targets = {}, {}
for aspect in ASPECTS:
    scores = calc_mean_neighborhood_jaccard(adj, protein_to_gos_by_aspect[aspect])
    obs_mean_homophily[aspect] = np.mean(list(scores.values()))
    obs_n_targets[aspect] = len(scores)
    print(f"[{aspect}] Observed mean neighborhood GO Jaccard: {obs_mean_homophily[aspect]:.4f} ({len(scores)} targets)")

N_PERM = 1000
perm_means = {aspect: [] for aspect in ASPECTS}
edge_list = sorted([list(e) for e in edges])
E = len(edge_list)
np.random.seed(42)
for perm_i in range(N_PERM):
    n_swaps = int(E * 0.15)
    perm_adj = {k: set(v) for k, v in adj.items()}
    for _ in range(n_swaps):
        idx1, idx2 = np.random.randint(0, E, size=2)
        u1, v1 = edge_list[idx1]
        u2, v2 = edge_list[idx2]
        if len({u1, v1, u2, v2}) == 4:
            if v2 not in perm_adj[u1] and v1 not in perm_adj[u2]:
                perm_adj[u1].discard(v1)
                perm_adj[v1].discard(u1)
                perm_adj[u2].discard(v2)
                perm_adj[v2].discard(u2)
                perm_adj[u1].add(v2)
                perm_adj[v2].add(u1)
                perm_adj[u2].add(v1)
                perm_adj[v1].add(u2)
    for aspect in ASPECTS:
        perm_scores = calc_mean_neighborhood_jaccard(perm_adj, protein_to_gos_by_aspect[aspect])
        if perm_scores:
            perm_means[aspect].append(np.mean(list(perm_scores.values())))

perm_rows = []
for aspect in ASPECTS:
    pm = np.array(perm_means[aspect])
    obs = obs_mean_homophily[aspect]
    z = (obs - pm.mean()) / pm.std() if pm.std() > 0 else np.nan
    p = (np.sum(pm >= obs) + 1) / (N_PERM + 1)
    print(f"[{aspect}] Permuted mean homophily: {pm.mean():.4f} +/- {pm.std():.4f} | Z={z:.2f}, P={p:.4e}")
    perm_rows.append({
        "Aspect": aspect,
        "Observed_Mean_Jaccard": obs,
        "N_Targets": obs_n_targets[aspect],
        "Permuted_Mean_Jaccard": pm.mean(),
        "Permuted_Std_Jaccard": pm.std(),
        "Z_Score": z,
        "P_Value": p,
        "N_Permutations": N_PERM,
    })

df_perm = pd.DataFrame(perm_rows)
out_perm = OUT_DIR / "topology_permutation_results.tsv"
df_perm.to_csv(out_perm, sep="\t", index=False)
print(f"Saved: {out_perm}")