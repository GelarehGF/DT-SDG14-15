# Digital Transformation Research in SDG 14 and SDG 15

This repository contains the complete Python source code used to construct the canonical document corpus, generate sentence-transformer embeddings, and reproduce the quantitative analyses, tables, and figures for the SDG 14–SDG 15 study.

The workflow is Python-only. It does not call the OpenAI API, an LLM, or a generative text service.

## Repository contents

| Path | Purpose |
|---|---|
| `src/build_canonical_corpus_embeddings.py` | Validates the selected DOCX papers, extracts text, removes back matter from embedding input, creates tokenizer-based chunks, generates SentenceTransformer embeddings, pools chunk vectors, and writes audit metadata. |
| `src/regenerate_all_sdg14_15_python.py` | Recomputes theory mapping, clustering, thematic terms, SDG mapping, technology extraction, claim patterns, similarity networks, knowledge-graph tables, sensitivity analyses, figures, and validation reports. |
| `requirements.txt` | Exact Python package versions used by the analysis. |

## Software environment

Python 3.12 is recommended. The confirmed analysis environment used Python 3.12.14.

Create an isolated environment and install the pinned dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first embedding run downloads the pinned SentenceTransformer model revision from Hugging Face unless that revision is already cached.

## Input data

The repository contains code, not copyrighted article files or licensed full text. Supply the source documents locally.

The embedding builder expects paper files named:

```text
SDG-14-NN.docx
SDG-15-NN.docx
```

A reproducible local layout is:

```text
data/
├── papers/
│   ├── SDG-14-01.docx
│   ├── ...
│   └── SDG-15-NN.docx
├── input_snapshot/
│   └── retained_selection.csv
└── theory_corpus_manual_adjudications.csv
```

`retained_selection.csv` must contain the retained paper identifiers in a `paper_id` or `file` column. The manual-adjudication CSV must use the schema validated by `regenerate_all_sdg14_15_python.py`, including paper identifiers, eligibility decisions, and completed manual theory codes.

## Step 1: build the canonical corpus and embeddings

```bash
python src/build_canonical_corpus_embeddings.py \
  --papers-dir data/papers \
  --output-dir data/input_snapshot/embeddings \
  --include-ids-csv data/input_snapshot/retained_selection.csv \
  --expected-count 116
```

The default embedding configuration is fixed in the script:

- model: `sentence-transformers/all-MiniLM-L6-v2`
- model revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- deterministic seed: `314`
- tokenizer-based overlapping chunks
- length-weighted mean pooling
- L2-normalized document vectors

The builder writes:

```text
canonical_manifest.csv
canonical_corpus.csv
chunk_manifest.csv
doc_embeddings.npy
integrity_report.json
run_metadata.json
```

The source DOCX files remain unchanged. Existing derived CSV, NPY, pickle, and figure files are not used as authorities.

## Step 2: run the full analysis

```bash
python src/regenerate_all_sdg14_15_python.py \
  --base-analysis data/input_snapshot \
  --manual-adjudications data/theory_corpus_manual_adjudications.csv \
  --output analysis_output
```

The analysis uses the canonical corpus and embeddings from:

```text
data/input_snapshot/embeddings/
```

It generates machine-readable tables, publication figures, derived embeddings, method parameters, an input snapshot, an output manifest with SHA-256 hashes, and a final validation report.

Use `--offline` only when the exact pinned SentenceTransformer revision is already present in the local model cache.

Use `--force` only when intentionally replacing an existing output directory. The analysis script deletes that output directory before rebuilding it.

## Recomputed analyses

The full analysis script performs:

- KMeans and HDBSCAN clustering
- automatic TF-IDF thematic labeling
- theory-anchor mapping restricted to eligible manuscript sections
- explicit theory-name validation
- integration of completed manual adjudications
- SDG-anchor similarity mapping
- deterministic technology extraction and co-occurrence
- paper-similarity and cross-SDG network analysis
- threshold sensitivity analysis
- multilayer knowledge-graph construction
- rule-based claim extraction from the permitted sections
- descriptive and structural statistics
- all analysis tables and figures
- file-level integrity and reconciliation checks

Embedding similarity is treated as conceptual affinity. It is not treated as proof that a paper applied a theory, established causality, or demonstrated ecological effectiveness.

## Validation and auditability

Both scripts fail on inconsistent identifiers, unexpected corpus size, invalid embedding shapes, non-finite vectors, misaligned rows, or unreconciled outputs. Generated artifacts are recorded in an output manifest with SHA-256 hashes. The final `validation_report.json` must report `"valid": true`.

Manual adjudications remain separate from computational inference. Completed human codes can override an embedding-inferred theory label, while study type, conceptual focus, eligibility, and formal theory are stored as distinct decisions.
