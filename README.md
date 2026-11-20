# Disentangling Homology and Network Confounding in Protein Language Models: A Time-Delayed Benchmark of Human No-Knowledge Proteins

This repository contains the data curation pipeline, model training scripts, a CAFA-style stratified evaluation framework, and the confounding assessment and robustness analyses for our manuscript, which evaluates sequence homology and interactome topology confounding in pretrained protein language models (pLMs) for Gene Ontology (GO) function prediction.

## Overview

We establish a prospective, CAFA-style time-delayed benchmark comprising **5,157 human no-knowledge (NK) proteins** that lacked experimental annotations in February 2020 (GAF v2.1) and had acquired their first experimental GO annotations by January 2026 (GAF v2.2). Targets are jointly stratified across three orthogonal axes

1. **GO Aspect:** Biological Process (BP), Cellular Component (CC), and Molecular Function (MF).
2. **Sequence Homology Tier (via DIAMOND blastp against 2020 Swiss-Prot):** Close (≥ 50% identity), Remote (≥ 35%), Twilight Zone (20–35%), Midnight Zone (< 20%), and Homology Orphans.
3. **Network Connectivity Tier (via STRING v12.0 physical interactions, score ≥ 400):** Isolate (0), Low (1–5), Medium (6–20), High (21–50), Hub (50+), and Unmapped.

## Repository Structure

```text
.
├── script/
│   ├── prepare_data.py          # Step 1: 2020–2026 NK target curation, True Path Rule propagation, IC calculation, DIAMOND & STRING stratification
│   ├── model_predictions.py     # Step 2: Naive frequency & DIAMOND baselines, pLM embedding extraction (ESM-2, ProtBERT, ProtT5) & MLP training
│   ├── evaluate_metrics.py      # Step 3: CAFA protein-centric metrics (Fmax, Smin, AUPR), 1,000x bootstrapping & homology/degree-stratified evaluation
│   ├── analyze_statistics.py    # Step 4: Maslov-Sneppen degree-preserving network permutation tests & standardized multivariable OLS regressions
│   └── causal_ablation.py       # Step 5: Frisch-Waugh-Lovell (FWL) residualization, partial R2 analysis & Dark-Zone (<35% identity) gap closure
├── requirements.txt             # Python package dependencies
└── README.md
```

## Environment Setup

We recommend using Python 3.10+ with a CUDA-enabled GPU. Install all required dependencies via:

```bash
pip install -r requirements.txt
```

## Core Pipeline Usage

All scripts are designed to be executed sequentially from the project root directory:

### 1. Benchmark Curation, Ontology Propagation & Stratification

Curates the 5,157 human NK targets from the Feb 2020 and Jan 2026 UniProtKB/Swiss-Prot and human GAF snapshots (filtering for experimental evidence codes `EXP, IDA, IPI, IMP, IGI, IEP, HTP, HDA, HMP, HGI, HEP`). Propagates annotations upward via `is_a` and `part_of` relations (`go-basic`, June 2026 release) following the True Path Rule, computes term Shannon Information Content (IC), and assigns homology and STRING v12.0 degree strata.

```bash
python script/prepare_data.py
```

### 2. Baseline Scoring, pLM Embedding Extraction & Classifier Training

Generates predictions for the empirical naive baseline and DIAMOND homology-transfer baseline. Extracts attention-masked, mean-pooled embeddings (truncated to 1,022 N-terminal residues) from six pretrained pLMs:

* `ProtBERT`
* `ProtT5-XL-UniRef50` (half-precision encoder)
* `ESM-2` (`8M`, `150M`, `650M`, and `3B` parameters)

Trains aspect-specific multilayer perceptron (MLP) classifiers (1,024-unit hidden layer, LayerNorm, GELU, dropout 0.3, AdamW optimizer, 15 epochs) on 2020 reference proteins.

```bash
python script/model_predictions.py
```

### 3. CAFA-Style Evaluation & Multi-Axis Stratified Bootstrapping

Computes protein-centric precision, recall, F<sub>max</sub>, remaining uncertainty (ru), misinformation (mi), semantic distance (S<sub>min</sub>), relative S<sub>min</sub>, and AUPR across 101 thresholds (τ ∈ [0.00, 1.00]). Estimates 95% confidence intervals via non-parametric bootstrapping (B = 1,000 resamples) with Benjamini–Hochberg FDR correction (q < 0.05) globally and within each homology and PPI degree stratum.

```bash
python script/evaluate_metrics.py
```

### 4. Network Homophily Permutation Testing & Multivariable Regression

Computes mean neighborhood GO Jaccard similarity on the STRING v12.0 physical interactome and generates empirical null distributions across N<sub>perm</sub> = 1,000 networks rewired via Maslov–Sneppen degree-preserving double-edge swaps (15% edge swap attempts). Also fits standardized multivariable Ordinary Least Squares (OLS) regressions of protein-level F<sub>1</sub> on maximum sequence identity and ln(1 + PPI degree).

```bash
python script/analyze_statistics.py
```

### 5. Frisch–Waugh–Lovell (FWL) Causal Ablation & Dark-Zone Gap Closure

Performs semiparametric confounding adjustment by residualizing protein-level F<sub>1</sub> and log-degree on homology-stratum fixed effects, log-transformed DIAMOND bitscore, and ground-truth annotation size via the Frisch–Waugh–Lovell theorem to quantify partial R<sup>2</sup> and coefficient attenuation. Computes the fraction of the recoverable performance gap closed relative to homology transfer across the Homology Dark Zone (< 35% identity) versus the Non-Dark Zone (≥ 35% identity).

```bash
python script/causal_ablation.py
```

## License

This project is licensed under the MIT License.
