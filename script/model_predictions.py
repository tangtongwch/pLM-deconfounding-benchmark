import argparse
import gzip
import hashlib
import json
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    BertModel,
    BertTokenizer,
    T5EncoderModel,
    T5Tokenizer,
    logging as hf_logging,
)

hf_logging.set_verbosity_error()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "raw"
PROCESSED_DIR = PROJECT_ROOT / "processed"

TARGETS_DIR = PROCESSED_DIR / "targets"
HOMOLOGY_DIR = PROCESSED_DIR / "homology"

EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings"
OUT_DIR = PROCESSED_DIR / "predictions"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OBO_FILE = RAW_DIR / "go-basic_2026.obo"
GAF_2020_FILE = RAW_DIR / "goa_human_2020.gaf.gz"
SPROT_2020_FASTA = RAW_DIR / "sprot_2020" / "uniprot_sprot.fasta.gz"
SPROT_2026_FASTA = RAW_DIR / "sprot_2026" / "uniprot_sprot.fasta.gz"

EXP_CODES = {'EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'HTP', 'HDA', 'HMP', 'HGI', 'HEP'}
ASPECT_MAP = {
    'biological_process': 'BP',
    'molecular_function': 'MF',
    'cellular_component': 'CC'
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PLM_CONFIG = {
    "esm2_8m":   {"path": "esm2/esm2_t6_8M_UR50D",       "type": "esm"},
    "esm2_150m": {"path": "esm2/esm2_t30_150M_UR50D",     "type": "esm"},
    "esm2_650m": {"path": "esm2/esm2_t33_650M_UR50D",     "type": "esm"},
    "esm2_3b":   {"path": "esm2/esm2_t36_3B_UR50D",       "type": "esm"},
    "protbert":  {"path": "protbert/prot_bert",           "type": "bert"},
    "prott5":    {"path": "prot_t5/prot_t5_xl_half_uniref50-enc", "type": "t5"},
}
AVAILABLE_MODELS = ["naive", "blast"] + list(PLM_CONFIG.keys())


def load_nk_targets():
    print("Loading NK targets from processed/targets/...")
    aspect_sets = {}
    for asp in ("BP", "MF", "CC"):
        target_file = TARGETS_DIR / f"targets_NK_{asp}.txt"
        if target_file.exists():
            aspect_sets[asp] = set(target_file.read_text().split())
            print(f"  - {asp}: {len(aspect_sets[asp])} targets")
        else:
            print(f"  - {asp}: Warning, file not found. Setting to empty.")
            aspect_sets[asp] = set()
    return aspect_sets

def parse_obo(path):
    print(f"Parsing ontology from {path.name}...")
    parents = defaultdict(set)
    alt_ids, namespaces = {}, {}
    obsolete = set()
    in_term, term = False, None
    with open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('['):
                in_term = (line == '[Term]')
                term = None
                continue
            if not in_term:
                continue
            if line.startswith('id: '):
                term = line[4:].strip()
            elif line.startswith('namespace: ') and term:
                namespaces[term] = line[11:].strip()
            elif line.startswith('alt_id: ') and term:
                alt_ids[line[8:].strip()] = term
            elif line.startswith('is_obsolete: true') and term:
                obsolete.add(term)
            elif (line.startswith('is_a: ') or line.startswith('relationship: part_of ')) and term:
                p = line.split()[1].strip()
                parents[term].add(p)
    for term in obsolete:
        namespaces.pop(term, None)
        parents.pop(term, None)
    return parents, alt_ids, namespaces

def build_ancestors(parents, namespaces):
    ancestors = {}
    for term, ns in namespaces.items():
        seen = {term}
        queue = deque([term])
        while queue:
            curr = queue.popleft()
            for p in parents.get(curr, ()):
                if p not in seen and namespaces.get(p) == ns:
                    seen.add(p)
                    queue.append(p)
        ancestors[term] = seen
    return ancestors

def load_2020_annotations(gaf_path, alt_ids, ancestors, namespaces):
    print(f"Loading and propagating training annotations from {gaf_path.name}...")
    annotations = {asp: defaultdict(set) for asp in ('BP', 'MF', 'CC')}
    with gzip.open(gaf_path, 'rt', errors='replace') as f:
        for line in f:
            if line.startswith('!') or not line.strip():
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 15 or parts[0] != 'UniProtKB':
                continue
            if 'NOT' in parts[3].strip().split('|') or parts[6] not in EXP_CODES:
                continue
            acc = parts[1].split('-')[0]
            go_id = alt_ids.get(parts[4], parts[4])
            ns = namespaces.get(go_id)
            if not ns or ns not in ASPECT_MAP:
                continue
            asp = ASPECT_MAP[ns]
            if go_id in ancestors:
                annotations[asp][acc].update(ancestors[go_id])
    return annotations

def load_fasta_sequences(fasta_gz: Path, wanted_ids: set) -> dict:
    seqs = {}
    print(f"Reading sequences from {fasta_gz.name}...")
    with gzip.open(fasta_gz, "rt") as f:
        curr_id = None
        curr_seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if curr_id and curr_id in wanted_ids:
                    seqs[curr_id] = "".join(curr_seq)
                header = line.split()[0].lstrip(">")
                curr_id = header.split("|")[1] if "|" in header else header
                curr_seq = []
            else:
                curr_seq.append(line)
        if curr_id and curr_id in wanted_ids:
            seqs[curr_id] = "".join(curr_seq)
    return seqs

def run_naive(annotations_2020, nk_targets_by_aspect):
    print("\n[Naive] Generating baseline predictions...")
    records = []
    for asp in ("BP", "MF", "CC"):
        nk_set = nk_targets_by_aspect[asp]
        train_proteins = set(annotations_2020[asp].keys()) - nk_set

        N_train = len(train_proteins)
        if N_train == 0: continue

        term_counts = defaultdict(int)
        for p in train_proteins:
            for t in annotations_2020[asp][p]:
                term_counts[t] += 1

        valid_terms = [(t, round(cnt / N_train, 4)) for t, cnt in term_counts.items() if (cnt / N_train) >= 0.01]
        targets = list(nk_set)

        if not valid_terms or not targets: continue

        for p in targets:
            for t, score in valid_terms:
                records.append((p, t, score, asp))

    out_file = OUT_DIR / "predictions_naive.tsv"
    pd.DataFrame(records, columns=["Protein", "GO_ID", "Score", "Aspect"]).to_csv(out_file, sep="\t", index=False)
    print(f" -> Saved Naive predictions to {out_file.name} ({len(records)} rows)")

def run_blast_knn(annotations_2020, nk_targets_by_aspect):
    print("\n[BLAST] Generating homology-based predictions...")
    cleaned_tsv = HOMOLOGY_DIR / "nk_vs_swissprot2020_cleaned.tsv"

    if not cleaned_tsv.exists() or cleaned_tsv.stat().st_size == 0:
        print(f"CRITICAL Error: Cleaned homology file not found or empty: {cleaned_tsv}")
        print("Please ensure Step 5 (prepare_data.py) completed successfully and exported the cleaned list.")
        sys.exit(1)

    hits = pd.read_csv(cleaned_tsv, sep="\t")
    if hits["pident"].max() > 1.0:
        hits["pident"] /= 100.0

    records = []
    for asp in ("BP", "MF", "CC"):
        targets = nk_targets_by_aspect[asp]
        aspect_hits = hits[hits["Protein"].isin(targets)]

        prot_term_scores = defaultdict(lambda: defaultdict(float))

        for _, row in aspect_hits.iterrows():
            query, subject, pident = row["Protein"], row["Hit"], row["pident"]
            if subject in annotations_2020[asp]:
                for term in annotations_2020[asp][subject]:
                    prot_term_scores[query][term] = max(prot_term_scores[query][term], pident)

        for q, term_dict in prot_term_scores.items():
            for t, score in term_dict.items():
                if score >= 0.01:
                    records.append((q, t, round(score, 4), asp))

    out_file = OUT_DIR / "predictions_blast.tsv"
    pd.DataFrame(records, columns=["Protein", "GO_ID", "Score", "Aspect"]).to_csv(out_file, sep="\t", index=False)
    print(f" -> Saved BLAST predictions to {out_file.name} ({len(records)} rows)")

def compute_sequence_hash(seq_dict):
    hasher = hashlib.md5()
    for acc in sorted(seq_dict.keys()):
        hasher.update(acc.encode("utf-8"))
        hasher.update(seq_dict[acc].encode("utf-8"))
    return hasher.hexdigest()

def clean_sequence(seq, model_type):
    seq = re.sub(r"[UZOB]", "X", seq.upper())
    if model_type in ("t5", "bert"):
        return " ".join(list(seq))
    return seq
    
def extract_embeddings(model_name, model_base_dir, seq_dict):
    config = PLM_CONFIG[model_name]
    model_path = model_base_dir / config["path"]
    model_type = config["type"]
    out_pt = EMBEDDINGS_DIR / f"embeddings_{model_name}.pt"
    meta_path = out_pt.with_suffix(".meta.json")
    ckpt_pt = out_pt.with_suffix(".ckpt.pt")

    current_hash = compute_sequence_hash(seq_dict)
    if out_pt.exists() and meta_path.exists():
        try:
            with open(meta_path, "r") as f: meta = json.load(f)
            if meta.get("hash") == current_hash and meta.get("count") == len(seq_dict):
                print(f"[{model_name}] Loading cached embeddings from {out_pt.name}...")
                return torch.load(out_pt, map_location="cpu", weights_only=True)
            print(f"[{model_name}] Sequence data changed or cache stale. Re-computing embeddings.")
        except Exception:
             print(f"[{model_name}] Cache metadata corrupted. Re-computing embeddings.")

    emb_dict = {}
    if ckpt_pt.exists():
        try:
            ckpt = torch.load(ckpt_pt, map_location="cpu")
            if ckpt.get("hash") == current_hash:
                emb_dict = ckpt["emb_dict"]
                print(f"[{model_name}] Resuming from checkpoint: {len(emb_dict)} sequences already done.")
        except Exception:
            print(f"[{model_name}] Checkpoint corrupted, starting fresh.")

    print(f"[{model_name}] Loading model from {model_path} on {DEVICE}...")
    if not model_path.exists():
         print(f"\nERROR: Model path not found: {model_path}")
         sys.exit(1)

    try:
        if model_type == "t5":
            tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
            dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
            model = T5EncoderModel.from_pretrained(model_path, torch_dtype=dtype).to(DEVICE)
            batch_size = 8
        elif model_type == "bert":
            tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
            model = BertModel.from_pretrained(model_path).to(DEVICE)
            batch_size = 32
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModel.from_pretrained(model_path).to(DEVICE)
            batch_size = 2 if "3b" in model_name else 16
    except Exception as e:
        print(f"\nError loading {model_name}: {e}")
        raise e

    model.eval()
    sorted_seqs = sorted(seq_dict.items(), key=lambda x: len(x[1]))
    remaining = [(acc, seq) for acc, seq in sorted_seqs if acc not in emb_dict]

    print(f"[{model_name}] {len(emb_dict)} cached, {len(remaining)} remaining to process...")

    save_every = 100
    try:
        for batch_idx, i in enumerate(tqdm(range(0, len(remaining), batch_size), desc="  Running inference")):
            batch = remaining[i : i + batch_size]
            accs, raw_seqs = zip(*batch)
            proc_seqs = [clean_sequence(s[:1022], model_type) for s in raw_seqs]

            inputs = tokenizer(proc_seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs)
                emb = outputs.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                mean_emb = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                mean_emb = mean_emb.cpu().to(torch.float32)

            for acc, vec in zip(accs, mean_emb):
                emb_dict[acc] = vec

            if (batch_idx + 1) % save_every == 0:
                torch.save({"hash": current_hash, "emb_dict": emb_dict}, ckpt_pt)

    except torch.cuda.OutOfMemoryError as e:
        print(f"\n[{model_name}] CUDA OOM caught. Saving checkpoint with {len(emb_dict)} sequences before exiting.")
        torch.save({"hash": current_hash, "emb_dict": emb_dict}, ckpt_pt)
        torch.cuda.empty_cache()
        raise e

    torch.save(emb_dict, out_pt)
    with open(meta_path, "w") as f:
        json.dump({"hash": current_hash, "count": len(seq_dict)}, f, indent=2)
    if ckpt_pt.exists():
        ckpt_pt.unlink()
    print(f"[{model_name}] Saved embeddings to {out_pt.name}")
    return emb_dict
    
    
class MultiLabelMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=1024, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

def train_and_predict_plm(model_name, emb_dict, annotations_2020, nk_targets_by_aspect):
    print(f"\n[{model_name}] Training MLP heads and generating predictions...")
    records = []

    for asp in ("BP", "MF", "CC"):
        train_proteins = [p for p in annotations_2020[asp] if p in emb_dict]
        test_proteins = [p for p in nk_targets_by_aspect[asp] if p in emb_dict]

        if not train_proteins or not test_proteins:
            print(f"  - [{asp}] Skipping: Insufficient data (Train: {len(train_proteins)}, Test: {len(test_proteins)})")
            continue

        term_counts = defaultdict(int)
        for p in train_proteins:
            for t in annotations_2020[asp][p]:
                term_counts[t] += 1

        vocab = sorted([t for t, cnt in term_counts.items() if cnt >= 5])
        if not vocab:
             vocab = sorted([t for t, cnt in term_counts.items() if cnt >= 1])

        n_terms = len(vocab)
        if n_terms == 0:
             print(f"  - [{asp}] Skipping: Empty vocabulary.")
             continue
        term_to_idx = {t: i for i, t in enumerate(vocab)}

        print(f"  - [{asp}] Training MLP: {len(train_proteins)} samples, {n_terms} GO terms.")

        X_train = torch.stack([emb_dict[p] for p in train_proteins])
        Y_train = torch.zeros((len(train_proteins), n_terms), dtype=torch.float32)
        for i, p in enumerate(train_proteins):
            indices = [term_to_idx[t] for t in annotations_2020[asp][p] if t in term_to_idx]
            if indices:
                Y_train[i, indices] = 1.0

        in_dim = X_train.shape[1]
        mlp = MultiLabelMLP(in_dim, n_terms).to(DEVICE)
        optimizer = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=128, shuffle=True)
        mlp.train()
        for _ in range(15):
            for bx, by in train_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(mlp(bx), by)
                loss.backward()
                optimizer.step()

        mlp.eval()
        X_test = torch.stack([emb_dict[p] for p in test_proteins])
        test_loader = DataLoader(TensorDataset(X_test), batch_size=512, shuffle=False)
        preds_list = []
        with torch.no_grad():
            for (bx,) in test_loader:
                bx = bx.to(DEVICE)
                scores = torch.sigmoid(mlp(bx)).cpu().numpy()
                preds_list.append(scores)
        all_scores = np.vstack(preds_list)

        for i, prot in enumerate(test_proteins):
            prot_scores = all_scores[i]
            idx_above_threshold = np.where(prot_scores >= 0.01)[0]
            for t_idx in idx_above_threshold:
                records.append((prot, vocab[t_idx], round(float(prot_scores[t_idx]), 4), asp))

    out_file = OUT_DIR / f"predictions_{model_name}.tsv"
    df = pd.DataFrame(records, columns=["Protein", "GO_ID", "Score", "Aspect"])
    df.to_csv(out_file, sep="\t", index=False)
    print(f" -> Saved {model_name} predictions to {out_file.name} ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Step 6: Model Predictions for NK Targets")
    parser.add_argument("--model-dir", type=Path, required=False,
                        help="Base directory containing downloaded model weights (e.g., /mnt/storage/models). Required for PLMs.")
    parser.add_argument("--models", nargs="+", default=["all"],
                        choices=["all"] + AVAILABLE_MODELS,
                        help=f"Which models to run. Choices: {AVAILABLE_MODELS}")
    args = parser.parse_args()

    selected_models = AVAILABLE_MODELS if "all" in args.models else args.models
    plm_models = [m for m in selected_models if m in PLM_CONFIG]

    if plm_models and (not args.model_dir or not args.model_dir.exists()):
        parser.error("--model-dir is required and must exist when running PLMs.")

    print("--- Step 6: Generating Predictions ---")

    parents, alt_ids, namespaces = parse_obo(OBO_FILE)
    ancestors = build_ancestors(parents, namespaces)
    annotations_2020 = load_2020_annotations(GAF_2020_FILE, alt_ids, ancestors, namespaces)
    nk_targets_by_aspect = load_nk_targets()
    all_nk_targets = set().union(*nk_targets_by_aspect.values())

    if not all_nk_targets:
        print("No NK targets found in processed/targets/. Is Step 1 complete? Exiting.")
        return

    if "naive" in selected_models:
        run_naive(annotations_2020, nk_targets_by_aspect)
    if "blast" in selected_models:
        run_blast_knn(annotations_2020, nk_targets_by_aspect)

    if plm_models:
        train_proteins = set().union(*[annotations_2020[asp].keys() for asp in ("BP", "MF", "CC")])
        train_proteins = train_proteins - all_nk_targets
        needed_proteins = train_proteins | all_nk_targets

        print(f"\nPreparing sequences for {len(needed_proteins)} required proteins...")
        seqs_2020 = load_fasta_sequences(SPROT_2020_FASTA, needed_proteins)
        seqs_2026 = load_fasta_sequences(SPROT_2026_FASTA, needed_proteins)
        seq_universe = {**seqs_2026, **seqs_2020}

        missing = needed_proteins - set(seq_universe.keys())
        if missing:
            print(f"Warning: {len(missing)} needed proteins could not be found in FASTA files. They will be skipped.")
            for p in missing:
                if p in train_proteins: train_proteins.remove(p)
                if p in all_nk_targets: all_nk_targets.remove(p)
                for asp in nk_targets_by_aspect:
                    if p in nk_targets_by_aspect[asp]: nk_targets_by_aspect[asp].remove(p)

        print(f"Final sequence set size: {len(seq_universe)}")

        for model_name in plm_models:
            emb_dict = extract_embeddings(model_name, args.model_dir, seq_universe)
            train_and_predict_plm(model_name, emb_dict, annotations_2020, nk_targets_by_aspect)

    print("\ncompleted successfully. Outputs in processed/predictions/")

if __name__ == "__main__":
    main()