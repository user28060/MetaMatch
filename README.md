# MetaMatch — Supervised Schema Matching with a Meta-Space of Features

[![version](https://img.shields.io/github/v/release/your-org/metamatch?style=for-the-badge&logo=github)](#)
[![lint](https://img.shields.io/github/actions/workflow/status/your-org/metamatch/lint.yml?label=lint&style=for-the-badge&logo=github)](#)
[![build](https://img.shields.io/github/actions/workflow/status/your-org/metamatch/build.yml?label=build&style=for-the-badge&logo=github)](#)
[![test](https://img.shields.io/github/actions/workflow/status/your-org/metamatch/test.yml?label=test&style=for-the-badge&logo=github)](#)
[![codecov](https://img.shields.io/codecov/c/github/your-org/metamatch?style=for-the-badge&logo=codecov)](#)
[![conventional commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg?style=for-the-badge&logo=conventionalcommits)](https://www.conventionalcommits.org)
[![semantic-release](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--release-e10079.svg?style=for-the-badge)](https://github.com/semantic-release/semantic-release)
[![contributor covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg?style=for-the-badge)](https://www.contributor-covenant.org/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg?style=for-the-badge)](https://opensource.org/licenses/BSD-3-Clause)

## Purpose & Philosophy

This repository hosts the code and **reproducible pipeline** for the paper:

> **MetaMatch: Supervised Schema Matching for Heterogeneous Tabular Data using Meta-Space**  


MetaMatch treats schema matching as **supervised learning** in a **meta-space** that blends:
- **Classical** vector distances (cosine, euclidean, correlation, …)
- **Spectral** indicators on embedding matrices (stable rank, NESum, RankMe, Alpha-ReQ, SelfCluster)
- **Topological** cues (persistence entropy, bottleneck/Wasserstein distances)
- **Syntactic** name features (Levenshtein, Damerau–OSA, Jaro/JW, Jaccard & Dice on tokens/char-n-grams, prefix/suffix ratios, …)

The pipeline has two commands:
1. **`MetaMatch`** — builds the **meta-features** for all **S1×S2** attribute pairs and attaches labels from a **golden JSON**.
2. **`MetaLearn`** — trains/evaluates a classifier (e.g., **XGBoost**, **CatBoost**, **RandomForest**) to predict `true_match`.

---

## Abstract

Schema matching across heterogeneous tabular sources is difficult because names, values and contexts vary widely. We propose **MetaMatch**, a supervised framework that encodes each candidate pair of attributes into a **rich meta-feature vector** combining classical, spectral and topological signals derived from pre-trained embeddings, plus robust syntactic features on column names. Training a standard classifier over this meta-space yields strong accuracy and stability across datasets from the **Valentine** benchmark. We also identify a compact subset (~**58** features) that preserves effectiveness while significantly reducing runtime.

**Keywords:** Data matching · Meta-Learner · Meta-space · Supervised Learning

---

## Summary of the experimental evaluation

### Technical environment

- CPU-only experiments (macOS/Linux).  
- Python 3.11, scikit-learn; optional **XGBoost** and **CatBoost**.  
- Pre-trained text encoders such as **MiniLM**, **BERT**, **RoBERTa**, **DistilBERT**, **ALBERT**, **BART**.

### Datasets

We rely on the **Valentine** benchmark (TPC-DI, ChEMBL, OpenData, Wikidata, Magellan, …).  
Official release: **[Valentine on Zenodo](https://zenodo.org/records/5084605#.YOgWHBMzY-Q)**

We also publish our **generated meta-spaces** and **paper results** (per-RQ):  
**[Google Drive — article results](https://drive.google.com/drive/folders/1YgAXzpRy8__UqUHY833qkQcPXbDnEcOd?usp=sharing)**


Drive organization (as used in the paper):
```

Experimentation/
├── MetaSpace/    # Meta-Features & true_match (label) by dataset pairs across all categories
├── RQ1_Effeciency/
├── RQ2_Effectiveness/
├── RQ3_Feature_importance/
├── RQ4_Feature_Selection/
└── RQ5_Baseline/

````

### Key results (high level)

- **Effectiveness (RQ1):** On Valentine, **XGBoost** reaches **F1 ≈ 0.97** on average; **CatBoost ≈ 0.93**, **RandomForest ≈ 0.90**.
- **Efficiency (RQ2):** Pairwise meta-feature computation dominates runtime. A **~58-feature** subset preserves ≈0.97 F1 while reducing time by **≈33–40%**.
- **Feature importance (RQ3):** Classical features are strongest alone; spectral & topological features add complementary gains when combined.
- **Feature selection (RQ4):** Backward selection yields a compact, stable subset without hurting F1.
- **Baselines (RQ5):** MetaMatch is competitive vs. classical and LLM-aided baselines (Coma/Cupid/Magneto variants) reported in our study.

---

## System requirements

- **Python:** 3.11+
- **OS:** macOS or Linux (CPU-only is fine)
- **Poetry** for packaging & virtualenvs
- Optional: **xgboost** and **catboost** (for the extra classifiers)

---

## Installation

```bash
git clone <YOUR_REPO_URL> MetaMatch
cd MetaMatch

# (optional) pin Python 3.11 for the venv
poetry env use python3.11

# install project & dependencies
poetry install

# add optional learners if needed
poetry add xgboost catboost
````

If you later edit console scripts, re-run `poetry install`.

---

## Repository layout

```
MetaMatch/
├── pyproject.toml
├── src/
│   ├── meta_space/
│   │   ├── pipeline.py              # CLI: MetaMatch (build features from S1×S2)
│   │   ├── embed_utils.py
│   │   ├── golden_tools.py
│   │   └── meta_features/
│   │       ├── classical_distances.py
│   │       ├── spectral_features.py
│   │       ├── topology_features.py
│   │       └── syntax_string_features.py
│   └── Meta_learner/
│       └── train.py                 # CLI: MetaLearn (train/test classifiers)
├── tests/
│   ├── results_meta_space/          # default output dir for MetaMatch features
│   └── meta_learner/                # default output dir for models & metrics
└── (Experimentation/)               # paper results on Drive (not versioned here)
```

---

## Usage

### 1) Build the meta-space

**Golden mapping (JSON)**

Use column names **exactly** as in the CSV headers.

```json
{
  "dataset": "amazon_google_exp",
  "source": { "name": "S1", "csv": "path/to/source.csv" },
  "target": { "name": "S2", "csv": "path/to/target.csv" },
  "matches": [
    { "source_column": "title",        "target_column": "name" },
    { "source_column": "manufacturer", "target_column": "manufacturer" },
    { "source_column": "price",        "target_column": "price" }
  ]
}
```

**CLI help**

```bash
poetry run MetaMatch --help
```

```
Usage: MetaMatch [OPTIONS]

  Run MetaMatch: read CSVs, embed columns, build golden matrix, compute
  features, save results.

Options:
  --dataset TEXT       Dataset name for bookkeeping.  [required]
  --source-csv FILE    Path to source CSV (S1).  [required]
  --target-csv FILE    Path to target CSV (S2).  [required]
  --golden-json FILE   Path to golden JSON.
  --model TEXT         Embedding model alias or HF checkpoint.  [default: all-MiniLM-L6-v2]
  --out-dir DIRECTORY  Output directory.  [default: tests/results_meta_space]
  --help               Show this message and exit.
```

**Example**

```bash
poetry run MetaMatch \
  --dataset amazon_google_exp \
  --source-csv "/path/to/Experimentation/Datasets/Magellan/Unionable/amazon_google_exp/amazon_google_exp_source.csv" \
  --target-csv "/path/to/Experimentation/Datasets/Magellan/Unionable/amazon_google_exp/amazon_google_exp_target.csv" \
  --golden-json "/path/to/Experimentation/Datasets/Magellan/Unionable/amazon_google_exp/amazon_google_exp_mapping.json" \
  --model all-MiniLM-L6-v2
```

**Outputs**

* `tests/results_meta_space/Meta_Space__{dataset}__{model}.csv` (all meta-features + `true_match`)
* optional incremental `inter_*.csv` during long runs

---

### 2) Train & evaluate the meta-learner

**List available classifiers**

```bash
poetry run MetaLearn --features-csv tests/results_meta_space/Meta_Space__amazon_google_exp__all-MiniLM-L6-v2.csv --list-classifiers
```

Example list:

```
Available classifiers:
 - LogReg
 - RF
 - GBT
 - KNN
 - MLP
 - SVMlin
 - CatBoost
 - XGBoost
```

**CLI help**

```bash
poetry run MetaLearn --help
```

```
Options:
  --features-csv FILE          Path to MetaMatch features CSV.  [required]
  --classifier TEXT            One or more classifiers (default: RF). Choices
                               printed by --list-classifiers.
  --list-classifiers           List available classifier names and exit.
  --split [random|by-dataset]  Train/test split strategy.  [default: by-dataset]
  --test-size FLOAT            Hold-out split size if --split=random. [default: 0.2]
  --seed INTEGER               Random seed.  [default: 42]
  --out-dir DIRECTORY          Output directory.  [default: tests/meta_learner]
```

**Example**

```bash
poetry run MetaLearn \
  --features-csv "./tests/results_meta_space/Meta_Space__amazon_google_exp__all-MiniLM-L6-v2.csv" \
  --classifier XGBoost \
  --split by-dataset \
  --test-size 0.3 \
  --seed 42 \
  --out-dir "./tests/meta_learner"
```

**Outputs**

* `tests/meta_learner/metrics_global.csv` (accuracy, precision, recall, F1, ROC-AUC if available)
* `tests/meta_learner/metrics_per_category.csv` (per dataset category when present)
* `tests/meta_learner/predictions__{Classifier}.csv` (`y_true`, `y_pred`, `y_prob`)
* `tests/meta_learner/model__{Classifier}.joblib` (preprocessor + classifier pipeline)

---

## What’s included

* [Poetry](https://python-poetry.org) for dependency management
* [Click](https://palletsprojects.com/p/click/) for CLIs
* [pytest](https://docs.pytest.org) for tests
* [flake8](https://flake8.pycqa.org) & [mypy](http://mypy-lang.org/) for code quality (if enabled)
* CI workflows (lint/test/build/release) are encouraged via the badges above

---

## Everyday activity

### Build / install

```bash
poetry install
```

### Lint & type check

```bash
poetry run flake8 --count --show-source --statistics
poetry run mypy .
```

### Unit tests

```bash
poetry run pytest -v
```

### Docker (optional)

```bash
# build
docker build -t metamatch .

# run
docker run --rm -ti metamatch
```

---

## Troubleshooting

* **Console scripts not found (`MetaMatch`, `MetaLearn`)**
  Ensure they’re declared in `pyproject.toml` under `[project.scripts]`, then reinstall with `poetry install`.

* **Poetry/TOML issues (e.g., “<empty>” version)**
  Keep a single `[project]` section and valid versions:

  ```bash
  rm -f poetry.lock
  poetry lock
  poetry install
  ```

* **Golden JSON errors**
  Keys must be:

  ```json
  { "matches": [ { "source_column": "...", "target_column": "..." } ] }
  ```

  and the names must match the CSV headers exactly.

---

## License

BSD 3-Clause — see `LICENSE`.

---

