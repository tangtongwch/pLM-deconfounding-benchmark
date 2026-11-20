#!/usr/bin/env python3
import argparse
import gzip
import logging
import math
import subprocess
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "raw"
PROCESSED_DIR = ROOT / "processed"
TARGETS_DIR = PROCESSED_DIR / "targets"
ONTOLOGY_DIR = PROCESSED_DIR / "ontology"
PPI_DIR = PROCESSED_DIR / "ppi"
HOMOLOGY_DIR = PROCESSED_DIR / "homology"

SPROT_2020_DAT = RAW_DIR / "sprot_2020" / "uniprot_sprot.dat.gz"
SPROT_2020_FASTA = RAW_DIR / "sprot_2020" / "uniprot_sprot.fasta.gz"
SPROT_2026_FASTA = RAW_DIR / "sprot_2026" / "uniprot_sprot.fasta.gz"
BACKGROUND_FILE = RAW_DIR / "background_human_sprot_2020.txt"
OBO_FILE = RAW_DIR / "go-basic_2026.obo"
GAF_2020_FILE = RAW_DIR / "goa_human_2020.gaf.gz"
GAF_2026_FILE = RAW_DIR / "goa_human_2026.gaf.gz"
ALIAS_FILE = RAW_DIR / "9606.protein.aliases.v12.0.txt.gz"
NETWORK_FILE = RAW_DIR / "9606.protein.physical.links.detailed.v12.0.txt.gz"
SEC_AC_FILE = RAW_DIR / "sec_ac.txt"

EXP_CODES = {'EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'HTP', 'HDA', 'HMP', 'HGI', 'HEP'}
ASPECT_MAP = {'P': 'BP', 'F': 'MF', 'C': 'CC'}
NAMESPACE = {'BP': 'biological_process', 'MF': 'molecular_function', 'CC': 'cellular_component'}
ROOT_TERMS = {'BP': 'GO:0008150', 'MF': 'GO:0003674', 'CC': 'GO:0005575'}
ALL_ROOTS = set(ROOT_TERMS.values())
ROOT_BY_NS = {NAMESPACE[k]: v for k, v in ROOT_TERMS.items()}

ALIAS_PRIORITY = {
    'Ensembl_UniProt_AC': 0,
    'Ensembl_UniProt': 1,
    'Ensembl_UniProt_DR': 2,
    'BLAST_UniProt_AC': 3,
    'BLAST_UniProt_GN_Name': 4,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def build_background():
    if BACKGROUND_FILE.exists():
        return set(BACKGROUND_FILE.read_text().split())

    accessions = set()
    ac, is_human = None, False
    with gzip.open(SPROT_2020_DAT, "rt", errors="replace") as f:
        for line in f:
            if line.startswith("ID "):
                ac, is_human = None, False
            elif ac is None and line.startswith("AC "):
                ac = line[5:].split(";")[0].strip()
            elif line.startswith("OX   NCBI_TaxID=9606"):
                is_human = True
            elif line.startswith("//") and ac and is_human:
                accessions.add(ac)

    BACKGROUND_FILE.parent.mkdir(parents=True, exist_ok=True)
    BACKGROUND_FILE.write_text("\n".join(sorted(accessions)) + "\n")
    return accessions


def parse_gaf(path):
    proteins = {k: set() for k in ('BP', 'MF', 'CC')}
    annotations = {k: set() for k in ('BP', 'MF', 'CC')}
    with gzip.open(path, 'rt', errors='replace') as f:
        for line in f:
            if line.startswith('!'):
                continue
            cols = line.rstrip('\n').split('\t')
            if len(cols) < 15 or 'NOT' in cols[3].split('|') or cols[6] not in EXP_CODES:
                continue
            aspect = ASPECT_MAP.get(cols[8])
            if not aspect:
                continue
            acc = cols[1].split('-')[0]
            proteins[aspect].add(acc)
            annotations[aspect].add((acc, cols[4]))
    return proteins, annotations


def extract_nk_targets(background):
    TARGETS_DIR.mkdir(parents=True, exist_ok=True)
    prot_20, _ = parse_gaf(GAF_2020_FILE)
    prot_26, anno_26 = parse_gaf(GAF_2026_FILE)

    nk_by_aspect = {}
    growth_rows = []
    for asp in ('BP', 'MF', 'CC'):
        nk = (prot_26[asp] - prot_20[asp]) & background
        nk_by_aspect[asp] = nk
        (TARGETS_DIR / f"targets_NK_{asp}.txt").write_text("\n".join(sorted(nk)) + "\n")

        gt = sorted((a, go) for a, go in anno_26[asp] if a in nk)
        (TARGETS_DIR / f"ground_truth_{asp}.tsv").write_text(
            "\n".join(f"{a}\t{go}\t{asp}" for a, go in gt) + "\n"
        )

        bg20 = len(prot_20[asp] & background)
        bg26 = len(prot_26[asp] & background)
        growth_rows.append({
            "Aspect": asp,
            "Total_Background": len(background),
            "Background_Annotated_2020": bg20,
            "Background_Annotated_2026": bg26,
            "NK_Gain": len(nk),
        })
        logging.info("[%s] NK targets: %d, Ground truth annotations: %d", asp, len(nk), len(gt))

    pd.DataFrame(growth_rows).to_csv(
        TARGETS_DIR / "annotation_growth_summary.tsv", sep="\t", index=False
    )
    return nk_by_aspect


def parse_obo(path):
    parents = defaultdict(set)
    alt_ids, namespaces, obsolete = {}, {}, set()
    term = None

    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '[Term]':
                term = None
            elif line.startswith('['):
                term = 'IGNORE'
            elif term != 'IGNORE':
                if line.startswith('id: '):
                    term = line[4:].strip()
                elif term:
                    if line.startswith('namespace: '):
                        namespaces[term] = line[11:].strip()
                    elif line.startswith('alt_id: '):
                        alt_ids[line[8:].strip()] = term
                    elif line == 'is_obsolete: true':
                        obsolete.add(term)
                    elif line.startswith('is_a: '):
                        parents[term].add(line[6:].split('!')[0].strip())
                    elif line.startswith('relationship: part_of '):
                        parents[term].add(line[22:].split('!')[0].strip())

    for t in obsolete:
        namespaces.pop(t, None)
        parents.pop(t, None)
    parents = {k: {p for p in v if p not in obsolete} for k, v in parents.items()}
    alt_ids = {k: v for k, v in alt_ids.items() if v not in obsolete}
    return parents, alt_ids, namespaces


def build_ancestors(parents, namespaces):
    ancestors = {}
    for term, ns in namespaces.items():
        seen = {term}
        q = deque([term])
        while q:
            curr = q.popleft()
            for p in parents.get(curr, ()):
                if p not in seen and namespaces.get(p) == ns:
                    seen.add(p)
                    q.append(p)
        ancestors[term] = seen
    return ancestors


def build_depths(parents, namespaces):
    depths = {}
    for term, ns in namespaces.items():
        if term in ALL_ROOTS:
            depths[term] = 0
            continue
        seen, q = {term}, deque([(term, 0)])
        found = False
        while q:
            curr, dist = q.popleft()
            if curr in ALL_ROOTS:
                depths[term] = dist
                found = True
                break
            for p in parents.get(curr, ()):
                if p not in seen and namespaces.get(p) == ns:
                    seen.add(p)
                    q.append((p, dist + 1))
        if not found:
            depths[term] = 1
    return depths


def propagate_ground_truth(nk_by_aspect, ancestors, alt_ids, namespaces):
    for asp in ('BP', 'MF', 'CC'):
        in_tsv = TARGETS_DIR / f"ground_truth_{asp}.tsv"
        root_id, target_ns = ROOT_TERMS[asp], NAMESPACE[asp]
        pairs = set()

        for line in in_tsv.read_text().splitlines():
            cols = line.split('\t')
            if len(cols) != 3:
                continue
            acc, go_id = cols[0], alt_ids.get(cols[1], cols[1])
            for anc in ancestors.get(go_id, ()):
                if anc != root_id and namespaces.get(anc) == target_ns:
                    pairs.add((acc, anc))

        out_tsv = TARGETS_DIR / f"ground_truth_{asp}_propagated.tsv"
        out_tsv.write_text("\n".join(f"{acc}\t{go}\t{asp}" for acc, go in sorted(pairs)) + "\n")
        logging.info("[%s] Propagated ground truth pairs: %d", asp, len(pairs))


def compute_go_term_metrics(parents, alt_ids, namespaces, ancestors, depths):
    ONTOLOGY_DIR.mkdir(parents=True, exist_ok=True)

    protein_annotations = defaultdict(set)
    with gzip.open(GAF_2020_FILE, 'rt', errors='replace') as f:
        for line in f:
            if line.startswith('!'):
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 15 or parts[0] != 'UniProtKB' or 'NOT' in parts[3] or parts[6] not in EXP_CODES:
                continue
            acc = parts[1].split('-')[0]
            go_id = alt_ids.get(parts[4], parts[4])
            if go_id in ancestors:
                protein_annotations[acc].update(ancestors[go_id])

    term_to_proteins = defaultdict(set)
    for acc, terms in protein_annotations.items():
        for t in terms:
            term_to_proteins[t].add(acc)

    term_counts = {t: len(prots) for t, prots in term_to_proteins.items()}
    ns_totals = {ns: term_counts.get(root, 0) for ns, root in ROOT_BY_NS.items()}

    out_file = ONTOLOGY_DIR / "go_term_metrics_2020.tsv"
    with open(out_file, 'w') as out:
        out.write("GO_ID\tNamespace\tDepth\tCount_2020\tIC\tIA\n")
        for term, ns in sorted(namespaces.items()):
            cnt = term_counts.get(term, 0)
            n_ns = ns_totals.get(ns, 0)

            # IC
            p_t = max(1e-12, min(1.0, (cnt + 1) / (n_ns + 1))) if n_ns > 0 else 0.0
            ic = -math.log2(p_t) if n_ns > 0 else 0.0

            # IA
            if term in ALL_ROOTS:
                ia = 0.0
            else:
                direct_parents = [p for p in parents.get(term, ()) if namespaces.get(p) == ns]
                if not direct_parents:
                    ia = 0.0
                else:
                    joint = len(set.intersection(*(term_to_proteins.get(p, set()) for p in direct_parents)))
                    ratio = min(1.0, (cnt + 1) / (joint + 1))
                    ia = -math.log2(ratio) if ratio > 0 else 0.0

            d = depths.get(term, 1)
            out.write(f"{term}\t{ns}\t{d}\t{cnt}\t{ic:.4f}\t{ia:.4f}\n")


def compute_ppi_topology(nk_by_aspect, background):
    PPI_DIR.mkdir(parents=True, exist_ok=True)
    nk_proteins = set.union(*nk_by_aspect.values())
    all_prots = background | nk_proteins

    # Parse alias table
    best_alias = {}
    with gzip.open(ALIAS_FILE, 'rt', encoding='utf-8') as f:
        next(f)
        for idx, line in enumerate(f):
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 3 or parts[1] not in all_prots:
                continue
            ensp, alias, sources = parts[0], parts[1], parts[2].split(',')
            prio = min((ALIAS_PRIORITY[s] for s in sources if s in ALIAS_PRIORITY), default=len(ALIAS_PRIORITY))
            if alias not in best_alias or (prio, idx) < best_alias[alias][:2]:
                best_alias[alias] = (prio, idx, ensp)

    uniprot_to_ensp = {k: v[2] for k, v in best_alias.items()}
    target_ensps = set(uniprot_to_ensp.values())

    # Count degrees
    degrees = defaultdict(int)
    with gzip.open(NETWORK_FILE, 'rt', encoding='utf-8') as f:
        next(f)
        for line in f:
            p1, p2, *_, score = line.split()
            if p1 != p2 and int(score) >= 400 and p1 in target_ensps:
                degrees[p1] += 1

    rows = []
    for prot in sorted(all_prots):
        ensp = uniprot_to_ensp.get(prot)
        deg = degrees[ensp] if ensp else np.nan
        rows.append({
            'Protein': prot,
            'Mapped_ENSP': ensp if ensp else 'None',
            'Degree': deg,
            'In_BP': int(prot in nk_by_aspect['BP']),
            'In_MF': int(prot in nk_by_aspect['MF']),
            'In_CC': int(prot in nk_by_aspect['CC']),
            'In_NK_Target': int(prot in nk_proteins)
        })

    bg_df = pd.DataFrame(rows)
    bins = [-1, 0, 5, 20, 50, float('inf')]
    labels = ['0 (Isolated)', '1-5 (Low)', '6-20 (Medium)', '21-50 (High)', '>50 (Hub)']
    bg_df['Degree_Bin'] = pd.cut(bg_df['Degree'], bins=bins, labels=labels).astype(str)
    bg_df.loc[bg_df['Mapped_ENSP'] == 'None', 'Degree_Bin'] = 'Unmapped'

    bg_df.to_csv(PPI_DIR / "background_ppi_degrees.tsv", sep='\t', index=False, na_rep='NA')
    bg_df[bg_df['In_NK_Target'] == 1].to_csv(PPI_DIR / "nk_targets_ppi_degrees.tsv", sep='\t', index=False, na_rep='NA')

    # Statistics
    mapped = bg_df[bg_df['Mapped_ENSP'] != 'None']
    nk_deg = mapped.loc[mapped['In_NK_Target'] == 1, 'Degree'].to_numpy()
    rest_deg = mapped.loc[mapped['In_NK_Target'] == 0, 'Degree'].to_numpy()

    u_stat, u_p = stats.mannwhitneyu(nk_deg, rest_deg, alternative='two-sided')
    ks_stat, ks_p = stats.ks_2samp(nk_deg, rest_deg)
    r_rb = 1.0 - (2.0 * u_stat) / (len(nk_deg) * len(rest_deg))

    pd.DataFrame([
        {
            'Group': 'NK_Target', 'N': len(nk_deg),
            'Median_Degree': np.median(nk_deg),
            'IQR_25': np.percentile(nk_deg, 25), 'IQR_75': np.percentile(nk_deg, 75),
            'Pct_Isolated': round(100 * (nk_deg == 0).mean(), 2),
            'MannWhitney_p': u_p, 'RankBiserial_r': r_rb, 'KS_p': ks_p,
        },
        {
            'Group': 'Background_minus_NK', 'N': len(rest_deg),
            'Median_Degree': np.median(rest_deg),
            'IQR_25': np.percentile(rest_deg, 25), 'IQR_75': np.percentile(rest_deg, 75),
            'Pct_Isolated': round(100 * (rest_deg == 0).mean(), 2),
            'MannWhitney_p': u_p, 'RankBiserial_r': r_rb, 'KS_p': ks_p,
        },
    ]).to_csv(PPI_DIR / "nk_vs_background_summary.tsv", sep='\t', index=False)


def _clean_acc(raw_id):
    parts = str(raw_id).split("|")
    return parts[1] if len(parts) >= 2 else parts[0]


def classify_homolog(pident, qcov, evalue, length):
    if pd.isna(pident) or evalue > 1e-3 or qcov < 0.20 or length < 50:
        return "Orphan"
    if pident >= 50.0 and qcov >= 0.50:
        return "Close homolog"
    if pident >= 35.0:
        return "Remote homolog"
    return "Twilight zone" if pident >= 20.0 else "Midnight zone"


def compute_homology(nk_by_aspect):
    HOMOLOGY_DIR.mkdir(parents=True, exist_ok=True)
    nk_targets = set.union(*nk_by_aspect.values()) if nk_by_aspect else set()

    nk_fasta = HOMOLOGY_DIR / "nk_targets.fasta"
    db_prefix = HOMOLOGY_DIR / "swissprot_2020"
    diamond_out = HOMOLOGY_DIR / "nk_vs_swissprot2020_raw.tsv"

    # Extract query FASTA
    with gzip.open(SPROT_2026_FASTA, "rt") as fin, open(nk_fasta, "w") as fout:
        active = False
        for line in fin:
            if line.startswith(">"):
                acc = _clean_acc(line.split()[0].lstrip(">"))
                active = acc in nk_targets
                if active:
                    fout.write(f">{acc}\n")
            elif active:
                fout.write(line)

    # Build DB and run DIAMOND
    dmnd_bin = db_prefix.with_suffix(".dmnd")
    if not dmnd_bin.exists():
        subprocess.run(["diamond", "makedb", "--in", str(SPROT_2020_FASTA), "--db", str(db_prefix)], check=True)

    subprocess.run([
        "diamond", "blastp", "--query", str(nk_fasta), "--db", str(db_prefix),
        "--out", str(diamond_out), "--outfmt", "6", "qseqid", "sseqid", "pident",
        "length", "evalue", "bitscore", "qlen", "qcovhsp", "--max-target-seqs", "25",
        "--evalue", "1.0", "--threads", "16", "--very-sensitive"
    ], check=True)

    # Secondary accessions
    aliases = defaultdict(set)
    if SEC_AC_FILE.exists():
        for line in SEC_AC_FILE.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) == 2 and not line.startswith(('-', 'Secondary')):
                aliases[parts[0]].add(parts[1])
                aliases[parts[1]].add(parts[0])

    cols = ["qseqid", "sseqid", "pident", "length", "evalue", "bitscore", "qlen", "qcovhsp"]
    hits = pd.read_csv(diamond_out, sep="\t", names=cols)
    hits["q_acc"] = hits["qseqid"].apply(_clean_acc)
    hits["s_acc"] = hits["sseqid"].apply(_clean_acc)
    hits["qcov"] = hits["qcovhsp"] / 100.0

    def get_rel(row):
        q, s = row["q_acc"], row["s_acc"]
        if q == s:
            return "self"
        if s in aliases.get(q, set()):
            return "alias"
        if row["pident"] >= 100.0 and row["qcov"] >= 0.999:
            return "identical_distinct_acc"
        return "distinct"

    hits["relationship"] = hits.apply(get_rel, axis=1)
    valid_hits = hits[~hits["relationship"].isin(["self", "alias"]) & ~hits["s_acc"].isin(nk_targets)].copy()

    # Cleaned hits export
    qual_mask = (valid_hits["evalue"] <= 1e-3) & (valid_hits["qcov"] >= 0.20) & (valid_hits["length"] >= 50)
    distinct_hits = valid_hits[qual_mask & (valid_hits["relationship"] == "distinct")]
    distinct_hits[["q_acc", "s_acc", "pident", "bitscore", "evalue"]].rename(
        columns={"q_acc": "Protein", "s_acc": "Hit"}
    ).to_csv(HOMOLOGY_DIR / "nk_vs_swissprot2020_cleaned.tsv", sep="\t", index=False)

    # Assign strata by best match
    best_hits = valid_hits.sort_values(
        ["q_acc", "bitscore", "evalue", "length"], ascending=[True, False, True, False]
    ).drop_duplicates("q_acc")

    res = pd.DataFrame({"Protein": sorted(nk_targets)}).merge(best_hits, left_on="Protein", right_on="q_acc", how="left")
    res["homology_bin"] = [
        classify_homolog(p, c, e, l) for p, c, e, l in zip(res["pident"], res["qcov"], res["evalue"], res["length"])
    ]
    res = res.rename(columns={"s_acc": "best_hit", "relationship": "hit_relationship"})

    for asp in ("BP", "MF", "CC"):
        res[f"In_{asp}"] = res["Protein"].isin(nk_by_aspect[asp]).astype(int)

    out_cols = [
        "Protein", "best_hit", "pident", "evalue", "bitscore", "qcov",
        "length", "hit_relationship", "homology_bin", "In_BP", "In_MF", "In_CC"
    ]
    res[out_cols].to_csv(HOMOLOGY_DIR / "nk_homology_bins.tsv", sep="\t", index=False)
    res["homology_bin"].value_counts().to_csv(HOMOLOGY_DIR / "homology_bin_summary.tsv", sep="\t", header=["count"])

    aspect_counts = []
    for asp in ("BP", "MF", "CC"):
        counts = res[res[f"In_{asp}"] == 1]["homology_bin"].value_counts()
        for k, v in counts.items():
            aspect_counts.append({"Aspect": asp, "homology_bin": k, "count": v})
    pd.DataFrame(aspect_counts).to_csv(HOMOLOGY_DIR / "homology_bin_summary_by_aspect.tsv", sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser(description="Build CAFA Dark Proteome Benchmark dataset.")
    parser.add_argument("--skip-ppi", action="store_true")
    parser.add_argument("--skip-homology", action="store_true")
    args = parser.parse_args()

    logging.info("Building human background (2020 Swiss-Prot)...")
    bg = build_background()

    logging.info("Extracting NK targets from 2020/2026 GAF files...")
    nk_by_aspect = extract_nk_targets(bg)

    logging.info("Parsing GO structure and propagating annotations...")
    parents, alt_ids, namespaces = parse_obo(OBO_FILE)
    ancestors = build_ancestors(parents, namespaces)
    depths = build_depths(parents, namespaces)
    propagate_ground_truth(nk_by_aspect, ancestors, alt_ids, namespaces)

    logging.info("Computing GO term IC & IA weights...")
    compute_go_term_metrics(parents, alt_ids, namespaces, ancestors, depths)

    if not args.skip_ppi:
        logging.info("Computing PPI degree distributions...")
        compute_ppi_topology(nk_by_aspect, bg)

    if not args.skip_homology:
        logging.info("Running homology search against 2020 Swiss-Prot...")
        compute_homology(nk_by_aspect)

    logging.info("Pipeline completed. Artifacts written to: %s", PROCESSED_DIR)


if __name__ == "__main__":
    main()