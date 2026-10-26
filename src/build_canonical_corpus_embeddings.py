#!/usr/bin/env python3
"""Build a selected SDG 14/15 corpus and chunked SentenceTransformer embeddings.

The source DOCX files are the authority. Existing derived CSV, NPY, pickle, and
figure outputs are deliberately ignored except for an optional filename/index
cross-check. The program:

1. validates the selected source documents and their identifiers;
2. extracts text deterministically from DOCX XML;
3. records file and text hashes for auditability;
4. tokenizes each document with the selected SentenceTransformer tokenizer;
5. creates overlapping token windows without model truncation;
6. pools chunk embeddings into one length-weighted document vector; and
7. L2-normalizes and saves the document vectors in canonical row order.

Run ``python build_canonical_corpus_embeddings.py --help`` for options.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import re
import sys
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
import xml.etree.ElementTree as ET


SCRIPT_VERSION = "1.1.0"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DEFAULT_SEED = 314
DEFAULT_PROJECT_DIR = Path(__file__).resolve().parent
PAPER_NAME_RE = re.compile(r"^SDG-(14|15)-(\d{2})\.docx$", re.IGNORECASE)
BACK_MATTER_HEADINGS = {
    "references",
    "bibliography",
    "works cited",
    "literature cited",
}

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W_TAG = f"{{{W_NS}}}"


@dataclass(frozen=True)
class DocumentRecord:
    embedding_row: int
    paper_id: str
    source_sdg: str
    sequence: int
    filename: str
    source_path: str
    file_sha256: str
    full_text_sha256: str
    embedding_text_sha256: str
    paragraph_count: int
    full_char_count: int
    full_word_count: int
    embedding_char_count: int
    embedding_word_count: int
    back_matter_removed: bool
    back_matter_heading: str
    full_text: str
    embedding_text: str


@dataclass(frozen=True)
class ChunkRecord:
    embedding_row: int
    paper_id: str
    chunk_index: int
    start_token: int
    end_token: int
    token_count: int
    pooling_weight: int
    chunk_sha256: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_paragraph(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def paragraph_text(paragraph: ET.Element) -> str:
    parts: list[str] = []
    for element in paragraph.iter():
        if element.tag == W_TAG + "t":
            parts.append(element.text or "")
        elif element.tag == W_TAG + "tab":
            parts.append("\t")
        elif element.tag in {W_TAG + "br", W_TAG + "cr"}:
            parts.append("\n")
    return normalize_paragraph("".join(parts))


def extract_docx_paragraphs(path: Path) -> list[str]:
    """Extract body paragraphs, including paragraphs inside table cells."""
    try:
        with zipfile.ZipFile(path) as archive:
            xml_data = archive.read("word/document.xml")
    except (zipfile.BadZipFile, KeyError) as error:
        raise ValueError(f"Invalid or unreadable DOCX: {path}: {error}") from error

    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError as error:
        raise ValueError(f"Invalid document.xml in {path}: {error}") from error

    body = root.find(f".//{W_TAG}body")
    if body is None:
        raise ValueError(f"No Word document body found in {path}")

    paragraphs: list[str] = []
    for paragraph in body.iter(W_TAG + "p"):
        text = paragraph_text(paragraph)
        if text:
            paragraphs.append(text)
    if not paragraphs:
        raise ValueError(f"No extractable text found in {path}")
    return paragraphs


def normalized_heading(text: str) -> str:
    text = re.sub(
        r"^\s*(?:\d+(?:\.\d+)*|[IVXLC]+)[.)]?\s+",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", text).strip().lower().rstrip(":")


def embedding_paragraphs(
    paragraphs: Sequence[str], exclude_back_matter: bool
) -> tuple[list[str], bool, str]:
    if not exclude_back_matter:
        return list(paragraphs), False, ""

    total_chars = sum(len(paragraph) for paragraph in paragraphs)
    chars_seen = 0
    for index, paragraph in enumerate(paragraphs):
        heading = normalized_heading(paragraph)
        far_enough = chars_seen >= max(5_000, int(total_chars * 0.35))
        if far_enough and heading in BACK_MATTER_HEADINGS:
            selected = list(paragraphs[:index])
            if selected:
                return selected, True, paragraph
        chars_seen += len(paragraph)
    return list(paragraphs), False, ""


def parse_paper_name(path: Path) -> tuple[str, int, str]:
    match = PAPER_NAME_RE.fullmatch(path.name)
    if not match:
        raise ValueError(
            f"Unexpected filename {path.name!r}; required form is SDG-14-01.docx "
            "or SDG-15-01.docx"
        )
    sdg_number = match.group(1)
    sequence = int(match.group(2))
    return f"SDG{sdg_number}", sequence, path.stem.upper()


def paper_sort_key(path: Path) -> tuple[int, int]:
    source_sdg, sequence, _ = parse_paper_name(path)
    return int(source_sdg.removeprefix("SDG")), sequence


def build_document_record(
    row: int, path: Path, exclude_back_matter: bool
) -> DocumentRecord:
    source_sdg, sequence, paper_id = parse_paper_name(path)
    paragraphs = extract_docx_paragraphs(path)
    selected, removed, heading = embedding_paragraphs(
        paragraphs, exclude_back_matter
    )
    full_text = "\n".join(paragraphs).strip()
    embedding_text = "\n".join(selected).strip()
    if not embedding_text:
        raise ValueError(f"Embedding text is empty for {path.name}")
    return DocumentRecord(
        embedding_row=row,
        paper_id=paper_id,
        source_sdg=source_sdg,
        sequence=sequence,
        filename=path.name,
        source_path=str(path.resolve()),
        file_sha256=sha256_file(path),
        full_text_sha256=sha256_text(full_text),
        embedding_text_sha256=sha256_text(embedding_text),
        paragraph_count=len(paragraphs),
        full_char_count=len(full_text),
        full_word_count=len(full_text.split()),
        embedding_char_count=len(embedding_text),
        embedding_word_count=len(embedding_text.split()),
        back_matter_removed=removed,
        back_matter_heading=heading,
        full_text=full_text,
        embedding_text=embedding_text,
    )


def read_expected_index(path: Path) -> tuple[set[str], list[str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "file" not in rows[0]:
        raise ValueError(f"Index must contain a 'file' column: {path}")
    filenames = {row["file"].strip() for row in rows if row.get("file", "").strip()}
    failed_statuses = [
        f"{row.get('file', '<unknown>')}: {row.get('status', '')}"
        for row in rows
        if "status" in row and row.get("status", "").strip().lower() != "ok"
    ]
    return filenames, failed_statuses


def read_include_ids(path: Path) -> set[str]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Selection file is empty: {path}")
    fields = set(rows[0])
    if "paper_id" not in fields and "file" not in fields:
        raise ValueError(
            f"Selection file must contain a 'paper_id' or 'file' column: {path}"
        )
    values: list[str] = []
    for row in rows:
        paper_id = str(row.get("paper_id", "")).strip().upper()
        filename = str(row.get("file", "")).strip()
        if not paper_id and filename:
            paper_id = Path(filename).stem.upper()
        if paper_id:
            values.append(paper_id)
    duplicates = sorted(
        paper_id for paper_id, count in Counter(values).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate paper_id values in selection file: {duplicates}")
    return set(values)


def find_duplicates(records: Sequence[DocumentRecord], field: str) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        grouped[str(getattr(record, field))].append(record.paper_id)
    return {value: ids for value, ids in grouped.items() if len(ids) > 1}


def validate_records(
    records: Sequence[DocumentRecord],
    expected_count: int,
    expected_index: Path | None,
    strict_unique_text: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if len(records) != expected_count:
        errors.append(f"Expected {expected_count} papers but found {len(records)}")

    paper_ids = [record.paper_id for record in records]
    if len(set(paper_ids)) != len(paper_ids):
        errors.append("Duplicate paper_id values detected")

    duplicate_files = find_duplicates(records, "file_sha256")
    duplicate_texts = find_duplicates(records, "full_text_sha256")
    if duplicate_files:
        errors.append(f"Exact duplicate DOCX files detected: {duplicate_files}")
    if duplicate_texts and strict_unique_text:
        errors.append(f"Exact duplicate extracted texts detected: {duplicate_texts}")
    elif duplicate_texts:
        warnings.append(
            "Exact duplicate extracted texts were preserved and assigned duplicate "
            f"groups: {duplicate_texts}"
        )

    source_files = {record.filename for record in records}
    index_check: dict[str, Any] = {"performed": False}
    if expected_index is not None and expected_index.exists():
        index_files, failed_statuses = read_expected_index(expected_index)
        missing_from_index = sorted(source_files - index_files)
        missing_from_source = sorted(index_files - source_files)
        index_check = {
            "performed": True,
            "path": str(expected_index.resolve()),
            "rows": len(index_files),
            "missing_from_index": missing_from_index,
            "missing_from_source": missing_from_source,
            "non_ok_statuses": failed_statuses,
        }
        if missing_from_index or missing_from_source or failed_statuses:
            errors.append("Source filenames do not reconcile with the expected index")
    elif expected_index is not None:
        warnings.append(f"Expected index not found; cross-check skipped: {expected_index}")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "document_count": len(records),
        "source_sdg_counts": dict(Counter(record.source_sdg for record in records)),
        "duplicate_file_hashes": duplicate_files,
        "duplicate_text_hashes": duplicate_texts,
        "expected_index_check": index_check,
    }


def tokenizer_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        truncation=False,
        verbose=False,
    )
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def chunk_document(
    tokenizer: Any,
    record: DocumentRecord,
    chunk_tokens: int,
    chunk_overlap: int,
) -> tuple[list[str], list[ChunkRecord]]:
    token_ids = tokenizer_ids(tokenizer, record.embedding_text)
    if not token_ids:
        raise ValueError(f"Tokenizer produced no tokens for {record.paper_id}")

    step = chunk_tokens - chunk_overlap
    texts: list[str] = []
    metadata: list[ChunkRecord] = []
    chunk_index = 0
    start = 0
    while start < len(token_ids):
        end = min(start + chunk_tokens, len(token_ids))
        window = token_ids[start:end]
        text = tokenizer.decode(
            window,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        if not text:
            raise ValueError(
                f"Decoded chunk is empty for {record.paper_id}, chunk {chunk_index}"
            )
        weight = len(window) if chunk_index == 0 else len(window) - chunk_overlap
        if weight <= 0:
            raise ValueError(
                f"Non-positive pooling weight for {record.paper_id}, chunk {chunk_index}"
            )
        texts.append(text)
        metadata.append(
            ChunkRecord(
                embedding_row=record.embedding_row,
                paper_id=record.paper_id,
                chunk_index=chunk_index,
                start_token=start,
                end_token=end,
                token_count=len(window),
                pooling_weight=weight,
                chunk_sha256=sha256_text(text),
            )
        )
        if end == len(token_ids):
            break
        start += step
        chunk_index += 1

    covered_weight = sum(item.pooling_weight for item in metadata)
    if covered_weight != len(token_ids):
        raise AssertionError(
            f"Chunk weights cover {covered_weight} tokens, expected {len(token_ids)} "
            f"for {record.paper_id}"
        )
    return texts, metadata


def seed_everything(seed: int, torch_module: Any | None = None) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    if torch_module is not None:
        torch_module.manual_seed(seed)
        if torch_module.cuda.is_available():
            torch_module.cuda.manual_seed_all(seed)
        try:
            torch_module.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, TypeError):
            pass


def model_commit_hash(model: Any) -> str:
    candidates: list[Any] = []
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None:
        candidates.append(getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"))
    try:
        candidates.append(getattr(model[0].auto_model.config, "_commit_hash", None))
    except (AttributeError, IndexError, TypeError):
        pass
    return next((str(value) for value in candidates if value), "unavailable")


def compute_embeddings(
    records: Sequence[DocumentRecord], args: argparse.Namespace
) -> tuple[Any, list[ChunkRecord], dict[str, Any]]:
    try:
        import numpy as np
        import sentence_transformers
        import torch
        import transformers
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Embedding dependencies are missing. Install numpy, torch, transformers, "
            "and sentence-transformers. The required model is loaded through the "
            "SentenceTransformer class."
        ) from error

    seed_everything(args.seed, torch)
    model_kwargs: dict[str, Any] = {}
    if args.device != "auto":
        model_kwargs["device"] = args.device
    if args.model_revision:
        model_kwargs["revision"] = args.model_revision
    model = SentenceTransformer(args.model, **model_kwargs)
    tokenizer = model.tokenizer

    special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
    model_limit = int(getattr(model, "max_seq_length", 0) or 0)
    if model_limit and args.chunk_tokens + special_tokens > model_limit:
        raise ValueError(
            f"chunk_tokens={args.chunk_tokens} plus {special_tokens} special tokens "
            f"exceeds model.max_seq_length={model_limit}. Lower --chunk-tokens."
        )

    document_vectors: list[Any] = []
    all_chunk_records: list[ChunkRecord] = []
    total_chunks = 0
    started = time.time()

    for position, record in enumerate(records, start=1):
        chunk_texts, chunk_records = chunk_document(
            tokenizer,
            record,
            args.chunk_tokens,
            args.chunk_overlap,
        )
        chunk_vectors = model.encode(
            chunk_texts,
            batch_size=args.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        chunk_vectors = np.asarray(chunk_vectors, dtype=np.float32)
        if chunk_vectors.ndim != 2 or chunk_vectors.shape[0] != len(chunk_records):
            raise ValueError(
                f"Unexpected chunk embedding shape {chunk_vectors.shape} for "
                f"{record.paper_id}"
            )

        if args.pooling == "mean":
            pooled = chunk_vectors.mean(axis=0)
        else:
            weights = np.asarray(
                [item.pooling_weight for item in chunk_records], dtype=np.float64
            )
            pooled = np.average(chunk_vectors, axis=0, weights=weights)
        pooled = np.asarray(pooled, dtype=np.float32)
        norm = float(np.linalg.norm(pooled))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError(f"Invalid pooled vector norm for {record.paper_id}: {norm}")
        document_vectors.append(pooled / norm)
        all_chunk_records.extend(chunk_records)
        total_chunks += len(chunk_records)
        print(
            f"[{position:03d}/{len(records):03d}] {record.paper_id}: "
            f"{len(chunk_records)} chunks",
            flush=True,
        )

    matrix = np.vstack(document_vectors).astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1)
    if matrix.shape[0] != len(records):
        raise AssertionError("Embedding row count does not match canonical corpus")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Embedding matrix contains NaN or infinite values")
    if not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError(
            f"Document vectors are not L2-normalized; norm range is "
            f"{norms.min()} to {norms.max()}"
        )

    details = {
        "library": "sentence-transformers",
        "model_requested": args.model,
        "model_revision_requested": args.model_revision or "default",
        "model_commit_hash": model_commit_hash(model),
        "device": str(getattr(model, "device", args.device)),
        "sentence_transformers_version": sentence_transformers.__version__,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "model_max_seq_length": model_limit,
        "special_tokens_per_chunk": special_tokens,
        "chunk_tokens_excluding_special_tokens": args.chunk_tokens,
        "chunk_overlap_tokens": args.chunk_overlap,
        "chunk_step_tokens": args.chunk_tokens - args.chunk_overlap,
        "pooling": args.pooling,
        "chunk_vectors_normalized_before_pooling": False,
        "document_vectors_l2_normalized_after_pooling": True,
        "embedding_shape": list(matrix.shape),
        "embedding_dtype": str(matrix.dtype),
        "document_norm_min": float(norms.min()),
        "document_norm_max": float(norms.max()),
        "total_chunks": total_chunks,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    return matrix, all_chunk_records, details


def atomic_write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temp_path = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, path)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temp_path, path)


def atomic_save_npy(path: Path, matrix: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        temp_path = Path(handle.name)
        np.save(handle, matrix, allow_pickle=False)
    os.replace(temp_path, path)


def ensure_output_targets(output_dir: Path, corpus_only: bool, force: bool) -> list[Path]:
    targets = [
        output_dir / "canonical_manifest.csv",
        output_dir / "canonical_corpus.csv",
        output_dir / "integrity_report.json",
        output_dir / "run_metadata.json",
    ]
    if not corpus_only:
        targets.extend(
            [
                output_dir / "chunk_manifest.csv",
                output_dir / "doc_embeddings.npy",
            ]
        )
    conflicts = [path for path in targets if path.exists()]
    if conflicts and not force:
        formatted = "\n".join(f"  - {path}" for path in conflicts)
        raise FileExistsError(
            "Refusing to overwrite existing canonical outputs. Use --force only "
            f"after reviewing them:\n{formatted}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return targets


def duplicate_text_annotations(
    records: Sequence[DocumentRecord],
) -> dict[str, tuple[str, int]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        grouped[record.full_text_sha256].append(record.paper_id)
    annotations: dict[str, tuple[str, int]] = {
        record.paper_id: ("", 1) for record in records
    }
    duplicate_groups = sorted(
        (digest, sorted(paper_ids))
        for digest, paper_ids in grouped.items()
        if len(paper_ids) > 1
    )
    for number, (_, paper_ids) in enumerate(duplicate_groups, start=1):
        label = f"duplicate_text_{number:03d}"
        for paper_id in paper_ids:
            annotations[paper_id] = (label, len(paper_ids))
    return annotations


def output_hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {path.name: sha256_file(path) for path in paths if path.exists()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a selected canonical SDG 14/15 dataset and recompute "
            "chunked SentenceTransformer document embeddings."
        )
    )
    parser.add_argument(
        "--papers-dir",
        type=Path,
        default=DEFAULT_PROJECT_DIR / "papers",
        help="Directory containing only SDG-14-NN.docx and SDG-15-NN.docx files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PROJECT_DIR / "canonical_embeddings",
        help="New directory for canonical data, embeddings, and audit metadata.",
    )
    parser.add_argument(
        "--expected-index",
        type=Path,
        default=DEFAULT_PROJECT_DIR / "outputs" / "df_corpus_index.csv",
        help="Optional existing index used only to reconcile filenames and status.",
    )
    parser.add_argument(
        "--include-ids-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV containing a paper_id or file column. Only listed DOCX "
            "records are embedded; the source directory is not modified."
        ),
    )
    parser.add_argument("--expected-count", type=int, default=116)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--model-revision",
        default=DEFAULT_MODEL_REVISION,
        help="Optional Hugging Face revision or commit hash. The resolved hash is recorded.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="SentenceTransformer device: auto, cpu, cuda, mps, or a specific device.",
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=224,
        help="Maximum tokenizer tokens per chunk, excluding special tokens.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=32,
        help="Number of tokenizer tokens repeated between adjacent chunks.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--pooling",
        choices=["length_weighted_mean", "mean"],
        default="length_weighted_mean",
        help=(
            "Pool chunk vectors using unique-token weights or an unweighted mean. "
            "The default avoids double-counting overlap."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, max(1, os.cpu_count() or 1)),
        help="Parallel workers for deterministic DOCX extraction.",
    )
    parser.add_argument(
        "--exclude-back-matter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Exclude text after a late standalone References/Bibliography heading "
            "from embeddings. Full text remains in canonical_corpus.csv."
        ),
    )
    parser.add_argument(
        "--strict-unique-text",
        action="store_true",
        help=(
            "Fail if two source files extract to identical full text. By default, "
            "all source records are preserved and duplicate groups are recorded."
        ),
    )
    parser.add_argument(
        "--corpus-only",
        action="store_true",
        help="Build and validate canonical CSV files without loading the model.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite only this program's named output files.",
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if args.expected_count <= 0:
        raise ValueError("--expected-count must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.chunk_tokens <= 0:
        raise ValueError("--chunk-tokens must be positive")
    if args.chunk_overlap < 0:
        raise ValueError("--chunk-overlap cannot be negative")
    if args.chunk_overlap >= args.chunk_tokens:
        raise ValueError("--chunk-overlap must be smaller than --chunk-tokens")


def main() -> int:
    args = build_parser().parse_args()
    validate_arguments(args)
    started_at = utc_now()

    papers_dir = args.papers_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    expected_index = (
        args.expected_index.expanduser().resolve() if args.expected_index else None
    )
    if not papers_dir.is_dir():
        raise FileNotFoundError(f"Papers directory does not exist: {papers_dir}")

    all_paper_paths = sorted(
        (path for path in papers_dir.iterdir() if path.suffix.lower() == ".docx"),
        key=paper_sort_key,
    )
    unexpected = sorted(
        path.name
        for path in papers_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".docx" and not PAPER_NAME_RE.fullmatch(path.name)
    )
    if unexpected:
        raise ValueError(f"Unexpected DOCX filenames: {unexpected}")

    include_ids_csv = (
        args.include_ids_csv.expanduser().resolve() if args.include_ids_csv else None
    )
    if include_ids_csv:
        include_ids = read_include_ids(include_ids_csv)
        source_by_id = {path.stem.upper(): path for path in all_paper_paths}
        missing_ids = sorted(include_ids - set(source_by_id))
        if missing_ids:
            raise ValueError(
                f"Selection contains paper IDs absent from the source directory: {missing_ids}"
            )
        paper_paths = [
            path for path in all_paper_paths if path.stem.upper() in include_ids
        ]
    else:
        paper_paths = all_paper_paths

    print(f"Extracting {len(paper_paths)} DOCX files from {papers_dir}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(build_document_record, row, path, args.exclude_back_matter)
            for row, path in enumerate(paper_paths)
        ]
        records = [future.result() for future in futures]
    records.sort(key=lambda record: record.embedding_row)

    integrity = validate_records(
        records,
        expected_count=args.expected_count,
        expected_index=expected_index,
        strict_unique_text=args.strict_unique_text,
    )
    if not integrity["valid"]:
        details = "\n".join(f"  - {error}" for error in integrity["errors"])
        raise ValueError(f"Canonical corpus validation failed:\n{details}")

    targets = ensure_output_targets(output_dir, args.corpus_only, args.force)
    manifest_fields = [
        "embedding_row",
        "paper_id",
        "source_sdg",
        "sequence",
        "filename",
        "source_path",
        "file_sha256",
        "full_text_sha256",
        "embedding_text_sha256",
        "paragraph_count",
        "full_char_count",
        "full_word_count",
        "embedding_char_count",
        "embedding_word_count",
        "back_matter_removed",
        "back_matter_heading",
        "duplicate_text_group",
        "duplicate_text_count",
    ]
    corpus_fields = [
        "embedding_row",
        "paper_id",
        "source_sdg",
        "filename",
        "full_text",
        "embedding_text",
        "duplicate_text_group",
        "duplicate_text_count",
    ]
    record_dicts = [asdict(record) for record in records]
    duplicate_annotations = duplicate_text_annotations(records)
    for row in record_dicts:
        group, count = duplicate_annotations[row["paper_id"]]
        row["duplicate_text_group"] = group
        row["duplicate_text_count"] = count
    atomic_write_csv(
        output_dir / "canonical_manifest.csv", record_dicts, manifest_fields
    )
    atomic_write_csv(output_dir / "canonical_corpus.csv", record_dicts, corpus_fields)
    atomic_write_json(output_dir / "integrity_report.json", integrity)

    embedding_details: dict[str, Any] = {"status": "skipped_by_corpus_only"}
    if not args.corpus_only:
        matrix, chunk_records, embedding_details = compute_embeddings(records, args)
        chunk_fields = list(ChunkRecord.__dataclass_fields__)
        atomic_write_csv(
            output_dir / "chunk_manifest.csv",
            (asdict(record) for record in chunk_records),
            chunk_fields,
        )
        atomic_save_npy(output_dir / "doc_embeddings.npy", matrix)

    data_targets = [path for path in targets if path.name != "run_metadata.json"]
    metadata = {
        "script_version": SCRIPT_VERSION,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "papers_dir": str(papers_dir),
        "output_dir": str(output_dir),
        "expected_count": args.expected_count,
        "include_ids_csv": str(include_ids_csv) if include_ids_csv else "",
        "include_ids_csv_sha256": (
            sha256_file(include_ids_csv) if include_ids_csv else ""
        ),
        "canonical_order": "numeric SDG number, then numeric filename sequence",
        "source_of_truth": "source DOCX files",
        "existing_derived_outputs_used": False,
        "expected_index_used_only_for_filename_reconciliation": bool(
            expected_index and expected_index.exists()
        ),
        "exclude_back_matter": args.exclude_back_matter,
        "seed": args.seed,
        "corpus": {
            "document_count": len(records),
            "source_sdg_counts": integrity["source_sdg_counts"],
            "total_full_characters": sum(record.full_char_count for record in records),
            "total_full_words": sum(record.full_word_count for record in records),
            "documents_with_back_matter_removed": sum(
                record.back_matter_removed for record in records
            ),
        },
        "embedding": embedding_details,
        "output_sha256": output_hashes(data_targets),
    }
    atomic_write_json(output_dir / "run_metadata.json", metadata)

    print(f"Canonical outputs written to {output_dir}", flush=True)
    print(json.dumps(metadata["corpus"], indent=2), flush=True)
    if not args.corpus_only:
        print(json.dumps(embedding_details, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; no source documents were modified.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
