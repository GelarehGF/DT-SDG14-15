#!/usr/bin/env python3
"""Deterministically regenerate the complete SDG 14–15 quantitative analysis.

This script uses only explicit Python rules and fixed statistical algorithms.
It does not call an LLM, an external API, or any generative text service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import sys
import tempfile
import unicodedata
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "sdg1415-matplotlib"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

warnings.filterwarnings("ignore", message=r"urllib3 v2 only supports OpenSSL.*")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from scipy.stats import gaussian_kde, kurtosis, skew, skewtest, ttest_1samp
from sentence_transformers import SentenceTransformer
from sklearn import __version__ as sklearn_version
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score

warnings.filterwarnings(
    "ignore",
    message=r".*encountered in matmul",
    category=RuntimeWarning,
    module=r"sklearn\.utils\.extmath",
)


SCRIPT_VERSION = "1.5.0"
SEED = 314
EXPECTED_FINAL_COUNT = 116
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
PRIMARY_SIMILARITY_THRESHOLD = 0.65
THEORY_TOP_CHUNKS = 1
THEORY_EVIDENCE_CHUNKS = 3
THEORY_BOOTSTRAP_REPLICATES = 200
THEORY_LOW_MARGIN = 0.03
THEORY_MODERATE_MARGIN = 0.015
THEORY_STABLE_BOOTSTRAP = 0.80
THEORY_MODERATE_BOOTSTRAP = 0.60
PROJECT_ROOT = Path(__file__).resolve().parent
BASE_ANALYSIS = PROJECT_ROOT / "input_snapshot"
DEFAULT_OUTPUT = PROJECT_ROOT / "analysis_output"
MANUAL_ADJUDICATIONS = PROJECT_ROOT / "theory_corpus_manual_adjudications.csv"

THEORY_ANCHORS = {
    "TAM": "Technology Acceptance Model (TAM)",
    "UTAUT": "Unified Theory of Acceptance and Use of Technology (UTAUT and UTAUT2)",
    "TOE": "Technology-Organization-Environment framework (TOE)",
    "RBV": "Resource-Based View theory (RBV and NRBV)",
    "Dynamic_Capabilities": "Dynamic Capabilities theory",
    "IDT": "Innovation Diffusion Theory (IDT), also called Diffusion of Innovations theory",
    "Stakeholder_Theory": "Stakeholder Theory",
    "STS": "Socio-Technical Systems theory (STS)",
    "Just_Digital": "Digital Justice theory and Just Digital Transformation framework",
}
MANUAL_ONLY_THEORY_LABELS = ["UTAUT2", "STES", "Maturity_Models"]
FINAL_FORMAL_THEORY_LABELS = list(THEORY_ANCHORS) + MANUAL_ONLY_THEORY_LABELS
NON_FORMAL_THEORY_LABELS = {"Not_inferable", "No_formal_theory"}

THEORY_MENTION_PATTERNS = {
    "TAM": [
        r"\btechnology\s+acceptance\s+model\b",
        r"\bTAM\b",
    ],
    "UTAUT": [
        r"\bunified\s+theory\s+of\s+acceptance\s+and\s+use\s+of\s+technology\b",
        r"\bUTAUT2?\b",
    ],
    "TOE": [
        r"\btechnology[\s\-–—]+organi[sz]ation[\s\-–—]+environment(?:al)?\s+(?:framework|model)\b",
        r"\bTOE\s+(?:framework|model)\b",
    ],
    "RBV": [
        r"\b(?:natural[\s\-–—]+)?resource[\s\-–—]+based\s+(?:view|theory)\b",
        r"\bN?RBV\b",
    ],
    "Dynamic_Capabilities": [
        r"\bdynamic\s+capabilit(?:y|ies)\s+(?:theory|view|framework)\b",
        r"\bdynamic\s+capabilities\b",
    ],
    "IDT": [
        r"\binnovation\s+diffusion\s+theory\b",
        r"\bdiffusion\s+of\s+innovations?\s+(?:theory|model)\b",
        r"\bIDT\s+(?:theory|model|framework)\b",
    ],
    "Stakeholder_Theory": [
        r"\bstakeholder\s+theory\b",
    ],
    "STS": [
        r"\bsocio[\s\-–—]*technical\s+systems?\s+(?:theory|framework)\b",
        r"\bSTS\s+(?:theory|framework|analysis)\b",
    ],
    "Just_Digital": [
        r"\bdigital\s+(?:environmental\s+)?justice\b",
        r"\bjust\s+digital\s+(?:transition|transformation)\b",
    ],
}

SECTION_PREFIX = (
    r"(?:(?:section|chapter)\s+)?"
    r"(?:(?:\d+(?:\.\d+)*|[ivxlcdm]+|[a-h])\s*(?:[.)|:\-]\s*|\s+))?"
)
TARGET_SECTION_PATTERNS = [
    (
        "abstract",
        re.compile(rf"^{SECTION_PREFIX}(?:(?:structured\s+)?abstract|abstrak)\b", re.IGNORECASE),
    ),
    (
        "results_discussion",
        re.compile(
            rf"^{SECTION_PREFIX}(?:(?:research|empirical)\s+)?"
            r"(?:(?:results?|findings?)\s*(?:and|&)\s*(?:discussion|analysis)s?"
            r"|hasil\s+dan\s+pembahasan)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "results",
        re.compile(
            rf"^{SECTION_PREFIX}(?:(?:research|empirical|main|study)\s+)?"
            r"(?:results?|findings?|hasil)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "discussion",
        re.compile(
            rf"^{SECTION_PREFIX}(?:(?:general\s+)?discussion(?:s)?|pembahasan)\b",
            re.IGNORECASE,
        ),
    ),
]
SECTION_BOUNDARY_PATTERNS = {
    "keywords": r"(?:index\s+terms?|key\s*words?|kata\s+kunci|ccs\s+concepts?)",
    "introduction": r"(?:introduction|background|pendahuluan)",
    "literature_review": (
        r"(?:literature\s+review|review\s+of\s+literature|related\s+works?|"
        r"theoretical\s+(?:background|framework)|conceptual\s+framework|tinjauan\s+pustaka)"
    ),
    "methods": (
        r"(?:methods?|methodology|materials?\s+and\s+methods?|"
        r"research\s+(?:design|methodology|methods?)|data\s+and\s+methods?|"
        r"metode(?:\s+penelitian)?)"
    ),
    "conclusion": r"(?:conclusions?|summary\s+and\s+conclusions?|kesimpulan)",
    "implications": r"(?:(?:practical|theoretical|managerial|policy)\s+)?implications?",
    "limitations": r"(?:limitations?|study\s+limitations?|limitations?\s+and\s+future\s+research)",
    "recommendations": r"(?:recommendations?|future\s+(?:work|research|directions?))",
    "references": r"(?:references?|bibliography|works?\s+cited|daftar\s+pustaka)",
    "appendix": r"(?:appendix|appendices|supplementary\s+(?:material|information))",
    "administrative": (
        r"(?:acknowledg(?:e)?ments?|author\s+contributions?|funding|"
        r"conflicts?\s+of\s+interest|data\s+availability(?:\s+statement)?)"
    ),
}

SDG_ANCHORS = {
    "SDG14": (
        "SDG 14 Life Below Water: conserve and sustainably use oceans, seas and marine "
        "resources; marine biodiversity, fisheries, coastal ecosystems, marine pollution, "
        "ocean acidification, illegal fishing, and ocean governance."
    ),
    "SDG15": (
        "SDG 15 Life on Land: protect, restore and sustainably use terrestrial ecosystems; "
        "forests, biodiversity, land, soil, wildlife, deforestation, desertification, "
        "habitat loss, and land degradation."
    ),
}

TECHNOLOGY_PATTERNS = {
    "AI_ML": [
        r"\bartificial[ -]intelligence\b",
        r"\bmachine[ -]learning\b",
        r"\bdeep[ -]learning\b",
        r"\bneural[ -]networks?\b",
        r"\bcomputer[ -]vision\b",
        r"\blarge[ -]language[ -]models?\b",
        r"\bgenerative[ -]ai\b",
        r"\bai(?:[ -](?:based|enabled|driven|powered|model|models|system|systems))?\b",
    ],
    "IoT": [
        r"\binternet[ -]of[ -]things\b",
        r"\bindustrial[ -]internet[ -]of[ -]things\b",
        r"\biiot\b",
        r"\biot\b",
        r"\bwireless[ -]sensor[ -]networks?\b",
        r"\bsmart[ -]sensors?\b",
    ],
    "Blockchain": [
        r"\bblock[ -]?chain\b",
        r"\bdistributed[ -]ledger(?:[ -]technology)?\b",
        r"\bsmart[ -]contracts?\b",
    ],
    "Remote_Sensing": [
        r"\bremote[ -]sensing\b",
        r"\bsatellite[ -](?:imagery|imaging|images?)\b",
        r"\bearth[ -]observation\b",
        r"\bsynthetic[ -]aperture[ -]radar\b",
        r"\bunmanned[ -]aerial[ -]vehicles?\b",
        r"\buavs?\b",
        r"\bdrones?\b",
        r"\blidar\b",
    ],
    "Digital_Twin": [
        r"\bdigital[ -]twins?\b",
        r"\bvirtual[ -]twins?\b",
        r"\bvirtual[ -]replicas?\b",
    ],
    "Big_Data": [
        r"\bbig[ -]data\b",
        r"\bdata[ -]analytics\b",
        r"\bpredictive[ -]analytics\b",
        r"\bdata[ -]mining\b",
        r"\blarge[ -]scale[ -]data\b",
    ],
}

CLAIM_VERB_PATTERNS = {
    "improve": r"\b(?:improv(?:e|es|ed|ing)|enhanc(?:e|es|ed|ing)|increas(?:e|es|ed|ing)|strengthen(?:s|ed|ing)?|boost(?:s|ed|ing)?)\b",
    "reduce": r"\b(?:reduc(?:e|es|ed|ing)|decreas(?:e|es|ed|ing)|mitigat(?:e|es|ed|ing)|minimi[sz](?:e|es|ed|ing)|lower(?:s|ed|ing)?)\b",
    "enable_support": r"\b(?:enabl(?:e|es|ed|ing)|facilitat(?:e|es|ed|ing)|support(?:s|ed|ing)?|promot(?:e|es|ed|ing)|contribut(?:e|es|ed|ing))\b",
    "affect_influence": r"\b(?:affect(?:s|ed|ing)?|influenc(?:e|es|ed|ing)|impact(?:s|ed|ing)?|driv(?:e|es|en|ing)|determin(?:e|es|ed|ing))\b",
    "predict_detect": r"\b(?:predict(?:s|ed|ing)?|forecast(?:s|ed|ing)?|detect(?:s|ed|ing)?|identif(?:y|ies|ied|ying)|estimat(?:e|es|ed|ing))\b",
    "optimize": r"\b(?:optimi[sz](?:e|es|ed|ing)|streamlin(?:e|es|ed|ing))\b",
    "protect_restore": r"\b(?:protect(?:s|ed|ing)?|conserv(?:e|es|ed|ing)|preserv(?:e|es|ed|ing)|restor(?:e|es|ed|ing))\b",
    "monitor_track": r"\b(?:monitor(?:s|ed|ing)?|track(?:s|ed|ing)?|observ(?:e|es|ed|ing)|surveil(?:s|led|ling)?)\b",
}

CLAIM_EVIDENCE_PATTERN = re.compile(
    r"\b(?:results?|findings?|evidence)\b"
    r"|\b(?:show|shows|showed|shown|indicate|indicates|indicated|demonstrate|demonstrates|demonstrated|"
    r"reveal|reveals|revealed|suggest|suggests|suggested|confirm|confirms|confirmed)\b"
    r"|\bwe\s+(?:find|found|show|demonstrate|identify|observe)\b"
    r"|\b(?:significant|significantly|positive|negative|positively|negatively|associated|relationship|effect)\b",
    flags=re.IGNORECASE,
)

SUBJECT_CATEGORIES = {
    "Technology": ["digital technology", "digital transformation", "artificial intelligence", "machine learning", "iot", "blockchain", "remote sensing", "digital twin", "technology", "technologies"],
    "Behavioral_Construct": ["perceived usefulness", "perceived ease", "performance expectancy", "effort expectancy", "behavioral intention", "social influence", "facilitating conditions", "attitude", "trust", "habit", "motivation", "perceived value", "variables", "factors"],
    "Innovation_Strategy": ["product innovation", "innovation", "strategy", "strategies", "process digitalization", "digitalization", "implementation", "intervention", "initiative", "solution", "solutions", "practice", "practices"],
    "Finance_Market": ["fintech", "finance", "financial", "investment", "investments", "market", "markets", "revenue", "supply chain", "bank", "banks", "banking"],
    "Organization": ["firm", "firms", "company", "companies", "enterprise", "enterprises", "business", "businesses", "msme", "organization", "organisation"],
    "Governance": ["government", "policy", "regulation", "institution", "governance", "stakeholder", "authority", "authorities"],
    "Community_User": ["user", "users", "farmer", "farmers", "fisher", "fishers", "community", "communities", "consumer", "consumers", "manager", "managers"],
    "Ecosystem": ["marine", "ocean", "forest", "biodiversity", "land", "ecosystem", "species", "wildlife", "fishery", "fisheries"],
    "Data_System": ["data", "system", "systems", "platform", "model", "models", "network", "networks", "sensor", "sensors"],
}

OUTCOME_CATEGORIES = {
    "Efficiency_Performance": ["efficiency", "performance", "productivity", "cost", "profit", "competitive advantage", "optimization", "optimisation"],
    "Sustainability_Environment": ["sustainability", "sustainable", "environment", "environmental", "biodiversity", "conservation", "emission", "pollution", "ecosystem"],
    "Adoption_Intention": ["adoption", "acceptance", "intention", "usefulness", "ease of use", "usage", "uptake"],
    "Monitoring_Detection": ["monitoring", "detection", "tracking", "surveillance", "observation", "prediction", "forecasting"],
    "Governance_Coordination": ["governance", "coordination", "compliance", "regulation", "policy", "transparency", "accountability", "collaboration"],
    "Resilience_Risk": ["resilience", "risk", "vulnerability", "adaptation", "uncertainty", "safety"],
    "Knowledge_Decision": ["decision", "knowledge", "information", "awareness", "learning", "planning", "management"],
    "Equity_Inclusion": ["equity", "equality", "inclusion", "inclusive", "justice", "access", "livelihood", "well-being", "wellbeing"],
}

STOP_WORDS = sorted(
    set(ENGLISH_STOP_WORDS)
    | {
        "al", "et", "study", "studies", "paper", "papers", "article", "articles",
        "research", "result", "results", "finding", "findings", "method", "methods",
        "analysis", "using", "used", "use", "based", "data", "section", "figure",
        "table", "digital", "transformation", "technology", "technologies", "sdg",
        "sdgs", "sustainable", "sustainability", "doi", "journal", "author", "authors",
        "university", "international", "copyright", "new", "provide", "provides",
    }
)

COLORS = ["#174A6E", "#2A9D8F", "#E9C46A", "#E76F51", "#6A4C93", "#457B9D", "#8AB17D", "#D1495B", "#7F5539"]
SDG_COLORS = {"SDG14": "#1678B4", "SDG15": "#4A9B52"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_csv(frame: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index, encoding="utf-8", float_format="%.12g")


def save_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_figure(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#17324D",
            "xtick.color": "#425466",
            "ytick.color": "#425466",
            "axes.edgecolor": "#CBD5DF",
            "figure.facecolor": "white",
        }
    )


def style_axes(ax: Any, title: str, subtitle: str = "") -> None:
    ax.set_title(title, loc="left", color="#17324D", pad=18)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color="#5B6770", va="bottom")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E8EDF2", linewidth=0.8, zorder=0)


def annotated_heatmap(
    values: np.ndarray,
    rows: list[str],
    columns: list[str],
    title: str,
    path: Path,
    fmt: str = ".0f",
    colorbar_label: str = "Count",
    figsize: tuple[float, float] = (8.0, 6.0),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    cmap = LinearSegmentedColormap.from_list("research", ["#F7FBFF", "#9ED6D1", "#174A6E"])
    image = ax.imshow(values, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(columns)), labels=columns)
    ax.set_yticks(range(len(rows)), labels=rows)
    maximum = float(np.nanmax(values)) if values.size else 0.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = float(values[i, j])
            color = "white" if maximum and value > maximum * 0.56 else "#17324D"
            ax.text(j, i, format(value, fmt), ha="center", va="center", color=color, fontsize=9, fontweight="bold")
    ax.set_title(title, loc="left", color="#17324D", pad=16)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03, label=colorbar_label)
    save_figure(fig, path)


def normalize_for_matching(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def gini(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array) or np.sum(array) == 0:
        return math.nan
    return float(np.abs(array[:, None] - array[None, :]).sum() / (2 * len(array) * array.sum()))


def normalized_entropy(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    total = array.sum()
    if len(array) <= 1 or total <= 0:
        return math.nan
    probabilities = array[array > 0] / total
    return float(-(probabilities * np.log(probabilities)).sum() / np.log(len(array)))


def sarle_bimodality(values: np.ndarray) -> float:
    n = len(values)
    if n < 4:
        return math.nan
    sample_skew = float(skew(values, bias=False))
    excess_kurtosis = float(kurtosis(values, fisher=True, bias=False))
    correction = 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return float((sample_skew**2 + 1) / (excess_kurtosis + correction))


def pair_mean(similarity: np.ndarray, mask: np.ndarray) -> float:
    values = similarity[np.triu(mask, k=1)]
    return float(values.mean()) if len(values) else math.nan


def automatic_cluster_label(terms: list[str]) -> str:
    selected: list[str] = []
    used_tokens: set[str] = set()
    for term in terms:
        tokens = set(term.split())
        if tokens and tokens.issubset(used_tokens):
            continue
        selected.append(term)
        used_tokens.update(tokens)
        if len(selected) == 3:
            break
    return " / ".join(selected)


def load_manual_adjudications(path: Path) -> pd.DataFrame:
    required = {
        "paper_id", "corpus_eligible", "eligibility_status", "exclusion_reason",
        "study_type", "manual_theory", "conceptual_focus", "adjudication_status",
        "adjudicated_by", "notes",
    }
    adjudications = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = required - set(adjudications.columns)
    if missing:
        raise ValueError(f"Manual adjudication file is missing columns: {sorted(missing)}")
    if adjudications["paper_id"].duplicated().any():
        duplicates = adjudications.loc[adjudications["paper_id"].duplicated(), "paper_id"].tolist()
        raise ValueError(f"Manual adjudication paper IDs are duplicated: {duplicates}")
    normalized_eligibility = adjudications["corpus_eligible"].str.strip().str.lower()
    if not normalized_eligibility.isin({"true", "false"}).all():
        raise ValueError("corpus_eligible must contain only true or false")
    adjudications["corpus_eligible"] = normalized_eligibility.eq("true")
    allowed_theories = set(FINAL_FORMAL_THEORY_LABELS) | {"No_formal_theory"}
    supplied_theories = {
        label
        for value in adjudications["manual_theory"]
        for label in parse_theory_labels(value)
    }
    undeclared = sorted(supplied_theories - allowed_theories)
    if undeclared:
        raise ValueError(f"Manual adjudications contain undeclared theory labels: {undeclared}")
    if not adjudications["adjudication_status"].eq("complete").all():
        raise ValueError("Only completed manual adjudications may alter the analysis")
    return adjudications


def parse_theory_labels(value: object) -> list[str]:
    """Return stable, de-duplicated semicolon-separated theory labels."""
    labels: list[str] = []
    for label in str(value or "").split(";"):
        normalized = label.strip()
        if normalized and normalized not in labels:
            labels.append(normalized)
    return labels


def explode_final_theories(theory: pd.DataFrame) -> pd.DataFrame:
    """Create one row per paper-theory membership for multi-theory coding."""
    rows: list[dict[str, str]] = []
    for item in theory[["paper_id", "final_theories"]].itertuples(index=False):
        labels = parse_theory_labels(item.final_theories)
        if not labels:
            labels = ["Not_inferable"]
        rows.extend({"paper_id": item.paper_id, "theory": label} for label in labels)
    return pd.DataFrame(rows, columns=["paper_id", "theory"])


def compute_manual_theory_agreement(
    theory: pd.DataFrame,
    tables: Path,
) -> pd.DataFrame:
    """Write descriptive top-1 concordance for the purposively adjudicated papers."""
    manual = theory.loc[
        theory["theory_assignment_source"].eq("human_adjudication"),
        ["paper_id", "manual_theory", "embedding_inferred_theory"],
    ].copy()
    manual["manual_labels"] = manual["manual_theory"].map(parse_theory_labels)
    anchor_space = set(THEORY_ANCHORS)
    manual["within_nine_anchor_label_space"] = manual["manual_labels"].map(
        lambda labels: bool(labels) and set(labels).issubset(anchor_space)
    )
    manual["top1_matches_any_manual_code"] = manual.apply(
        lambda row: row["embedding_inferred_theory"] in row["manual_labels"], axis=1
    )
    manual["manual_labels"] = manual["manual_labels"].map(lambda labels: ";".join(labels))
    save_csv(manual, tables / "manual_theory_agreement_records.csv")

    total = len(manual)
    comparable = manual["within_nine_anchor_label_space"]
    comparable_total = int(comparable.sum())
    outside = total - comparable_total
    comparable_matches = int(
        manual.loc[comparable, "top1_matches_any_manual_code"].sum()
    )
    overall_matches = int(manual["top1_matches_any_manual_code"].sum())
    summary = pd.DataFrame(
        [
            ("manually_adjudicated_papers", total),
            ("papers_with_manual_codes_inside_nine_anchor_space", comparable_total),
            ("papers_with_manual_codes_outside_nine_anchor_space", outside),
            ("outside_anchor_space_percent", 100 * outside / total if total else math.nan),
            ("top1_matches_within_comparable_subset", comparable_matches),
            (
                "comparable_subset_top1_concordance_percent",
                100 * comparable_matches / comparable_total if comparable_total else math.nan,
            ),
            ("top1_matches_across_all_manual_papers", overall_matches),
            (
                "overall_raw_top1_concordance_percent",
                100 * overall_matches / total if total else math.nan,
            ),
        ],
        columns=["metric", "value"],
    )
    save_csv(summary, tables / "manual_theory_agreement_summary.csv")
    return summary


def load_inputs(
    base: Path,
    adjudications: pd.DataFrame,
    adjudications_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict[str, Any]]:
    selection = pd.read_csv(base / "retained_selection.csv")
    corpus = pd.read_csv(base / "embeddings" / "canonical_corpus.csv")
    embeddings = np.load(base / "embeddings" / "doc_embeddings.npy", allow_pickle=False).astype(np.float64)
    embedding_metadata = json.loads((base / "embeddings" / "run_metadata.json").read_text(encoding="utf-8"))
    if selection.empty or selection["paper_id"].nunique() != len(selection):
        raise ValueError("The retained selection must contain unique paper IDs")
    if corpus["paper_id"].tolist() != selection["paper_id"].tolist():
        raise ValueError("Retained selection and canonical corpus are not row-aligned")
    if embeddings.shape != (len(selection), 384):
        raise ValueError(
            f"Expected embeddings with shape ({len(selection)}, 384), found {embeddings.shape}"
        )
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=2e-6):
        raise ValueError("Document embeddings are not L2-normalized")
    if set(selection["source_sdg"]) != {"SDG14", "SDG15"}:
        raise ValueError("Unexpected source SDG values")
    if "source_embedding_row" not in selection:
        selection["source_embedding_row"] = np.arange(len(selection), dtype=int)
    else:
        selection["source_embedding_row"] = selection["source_embedding_row"].astype(int)
    if "source_embedding_row" not in corpus:
        corpus["source_embedding_row"] = corpus["embedding_row"].astype(int)
    else:
        corpus["source_embedding_row"] = corpus["source_embedding_row"].astype(int)
    eligible_manual_ids = set(
        adjudications.loc[adjudications["corpus_eligible"], "paper_id"]
    )
    missing_eligible_manual_ids = sorted(eligible_manual_ids - set(selection["paper_id"]))
    if missing_eligible_manual_ids:
        raise ValueError(
            "Eligible manual adjudications are absent from the input corpus: "
            f"{missing_eligible_manual_ids}"
        )
    excluded_ids = set(
        adjudications.loc[~adjudications["corpus_eligible"], "paper_id"]
    )
    keep = ~selection["paper_id"].isin(excluded_ids)
    selection = selection.loc[keep].reset_index(drop=True)
    corpus = corpus.loc[keep].reset_index(drop=True)
    embeddings = embeddings[keep.to_numpy()]
    if len(selection) != EXPECTED_FINAL_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FINAL_COUNT} papers after adjudication, found {len(selection)}"
        )
    selection["embedding_row"] = np.arange(len(selection), dtype=int)
    corpus["embedding_row"] = np.arange(len(corpus), dtype=int)
    embedding_metadata = json.loads(json.dumps(embedding_metadata))
    embedding_metadata["expected_count"] = len(selection)
    embedding_metadata["corpus"]["document_count"] = len(selection)
    embedding_metadata["corpus"]["source_sdg_counts"] = selection["source_sdg"].value_counts().to_dict()
    embedding_metadata["embedding"]["embedding_shape"] = list(embeddings.shape)
    embedding_metadata["embedding"]["document_norm_min"] = float(np.linalg.norm(embeddings, axis=1).min())
    embedding_metadata["embedding"]["document_norm_max"] = float(np.linalg.norm(embeddings, axis=1).max())
    chunk_manifest_path = base / "embeddings" / "chunk_manifest.csv"
    if chunk_manifest_path.is_file():
        chunk_manifest = pd.read_csv(chunk_manifest_path, usecols=["paper_id"])
        embedding_metadata["embedding"]["total_chunks"] = int(
            chunk_manifest["paper_id"].isin(selection["paper_id"]).sum()
        )
    embedding_metadata["manual_adjudication"] = {
        "file": adjudications_path.name,
        "excluded_paper_ids": sorted(excluded_ids),
        "included_adjudicated_paper_ids": sorted(
            set(adjudications.loc[adjudications["corpus_eligible"], "paper_id"])
        ),
    }
    return selection, corpus, embeddings, embedding_metadata


def clean_section_line(line: str) -> str:
    """Normalize OCR formatting without changing substantive wording."""
    line = "".join(character for character in str(line) if unicodedata.category(character) != "Cf")
    line = line.replace("\u00ad", "")
    line = re.sub(r"\s+", " ", line).strip()
    line = re.sub(r"(?i)\ba\s+b\s+s\s+t\s+r\s+a\s+c\s+t\b", "ABSTRACT", line)
    compact_replacements = {
        "RESEARCHFINDINGSANDDISCUSSIONS": "RESEARCH FINDINGS AND DISCUSSIONS",
        "RESEARCHFINDINGSANDDISCUSSION": "RESEARCH FINDINGS AND DISCUSSION",
        "RESULTSANDDISCUSSION": "RESULTS AND DISCUSSION",
        "RESULTANDDISCUSSION": "RESULT AND DISCUSSION",
    }
    for compact, expanded in compact_replacements.items():
        line = re.sub(compact, expanded, line, flags=re.IGNORECASE)
    return line


def split_embedded_section_markers(line: str) -> list[str]:
    """Split headings merged into body lines by multi-column PDF extraction."""
    normalized = clean_section_line(line)
    if not normalized:
        return []
    marker = re.compile(
        r"(?i)(?<!^)(?=(?:\s+)(?:"
        r"(?:\d+(?:\.\d+)*|[ivxlcdm]+)\s*[.)|:]\s*"
        r"(?:introduction|background|literature\s+review|methods?|methodology|"
        r"results?|findings?|discussion|conclusions?|references?)\b"
        r"|(?:key\s*words?|index\s+terms?|kata\s+kunci|ccs\s+concepts?)\s*[:;\-–—]"
        r"))"
    )
    return [part.strip() for part in marker.split(normalized) if part.strip()]


def detect_target_section(line: str) -> tuple[str, str] | None:
    """Return the eligible section label and any text sharing its heading line."""
    normalized = clean_section_line(line)
    if not normalized:
        return None
    for label, pattern in TARGET_SECTION_PATTERNS:
        match = pattern.match(normalized)
        if not match:
            continue
        tail = normalized[match.end() :].strip()
        if label == "abstract":
            inline = re.sub(r"^[\s:;,.\-–—]+", "", tail).strip()
            if re.fullmatch(r"ccs\s+concepts?", inline, flags=re.IGNORECASE):
                inline = ""
            return label, inline
        if not tail:
            return label, ""
        if tail[0] in ":;\u2013\u2014-":
            return label, re.sub(r"^[\s:;,.\-–—]+", "", tail).strip()
        heading_suffix = re.fullmatch(
            r"(?:of\s+(?:the\s+)?(?:results?|findings?|hypotheses|hypothesis(?:\s+tests?)?)"
            r"|and\s+(?:implications?|conclusions?|recommendations?|analysis)"
            r"|/\s*findings?|from\s+the\s+(?:survey|interviews?|case\s+study))\.?",
            tail,
            flags=re.IGNORECASE,
        )
        if heading_suffix:
            return label, ""
        matched_letters = "".join(character for character in normalized[: match.end()] if character.isalpha())
        if matched_letters and matched_letters.isupper():
            return label, tail
        return None
    return None


def classify_structural_heading(line: str) -> str | None:
    target = detect_target_section(line)
    if target:
        return target[0]
    normalized = clean_section_line(line)
    if not normalized or len(normalized) > 180 or len(normalized.split()) > 22:
        return None
    if re.match(
        rf"^{SECTION_PREFIX}(?:index\s+terms?|key\s*words?|kata\s+kunci|ccs\s+concepts?)\b\s*[:;\-–—]",
        normalized,
        flags=re.IGNORECASE,
    ):
        return "keywords"
    for label, body in SECTION_BOUNDARY_PATTERNS.items():
        if re.fullmatch(rf"{SECTION_PREFIX}{body}(?:\s*[:.]\s*)?", normalized, flags=re.IGNORECASE):
            return label
    return None


def extract_theory_sections(paper_id: str, text: str) -> list[dict[str, Any]]:
    """Extract only Abstract, Results/Findings, and Discussion text."""
    lines = [
        part
        for raw_line in str(text).splitlines()
        for part in split_embedded_section_markers(raw_line)
    ]
    sections: list[dict[str, Any]] = []
    for line_index, line in enumerate(lines):
        detected = detect_target_section(line)
        if not detected:
            continue
        section_type, inline_text = detected
        content = [inline_text] if inline_text else []
        end_index = len(lines)
        for candidate_index in range(line_index + 1, len(lines)):
            if classify_structural_heading(lines[candidate_index]) is not None:
                end_index = candidate_index
                break
            content.append(lines[candidate_index])
        section_text = "\n".join(value for value in content if value).strip()
        if len(section_text) < 40:
            continue
        sections.append(
            {
                "paper_id": paper_id,
                "section_index": len(sections),
                "section_type": section_type,
                "heading": line,
                "start_line": line_index,
                "end_line_exclusive": end_index,
                "section_char_count": len(section_text),
                "section_word_count": len(section_text.split()),
                "section_sha256": hashlib.sha256(section_text.encode("utf-8")).hexdigest(),
                "section_text": section_text,
            }
        )
    return sections


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


def chunk_theory_sections(
    model: SentenceTransformer,
    section_frame: pd.DataFrame,
    chunk_tokens: int,
    chunk_overlap: int,
) -> tuple[list[str], pd.DataFrame]:
    step = chunk_tokens - chunk_overlap
    if step <= 0:
        raise ValueError("Theory-section chunk step must be positive")
    chunk_texts: list[str] = []
    chunk_rows: list[dict[str, Any]] = []
    for section in section_frame.itertuples(index=False):
        token_ids = tokenizer_ids(model.tokenizer, section.section_text)
        if not token_ids:
            continue
        start = 0
        section_chunk_index = 0
        covered_weight = 0
        while start < len(token_ids):
            end = min(start + chunk_tokens, len(token_ids))
            window = token_ids[start:end]
            chunk_text = model.tokenizer.decode(
                window,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            if not chunk_text:
                raise ValueError(f"Empty theory chunk for {section.paper_id}, section {section.section_index}")
            weight = len(window) if section_chunk_index == 0 else len(window) - chunk_overlap
            covered_weight += weight
            global_index = len(chunk_texts)
            chunk_texts.append(chunk_text)
            chunk_rows.append(
                {
                    "theory_chunk_row": global_index,
                    "paper_id": section.paper_id,
                    "section_index": int(section.section_index),
                    "section_type": section.section_type,
                    "section_chunk_index": section_chunk_index,
                    "start_token": start,
                    "end_token": end,
                    "token_count": len(window),
                    "pooling_weight": weight,
                    "chunk_sha256": hashlib.sha256(chunk_text.encode("utf-8")).hexdigest(),
                    "chunk_text": chunk_text,
                }
            )
            if end == len(token_ids):
                break
            start += step
            section_chunk_index += 1
        if covered_weight != len(token_ids):
            raise AssertionError(
                f"Theory chunks cover {covered_weight} tokens, expected {len(token_ids)} for "
                f"{section.paper_id}, section {section.section_index}"
            )
    return chunk_texts, pd.DataFrame(chunk_rows)


def map_conceptual_anchors(
    selection: pd.DataFrame,
    corpus: pd.DataFrame,
    embeddings: np.ndarray,
    embedding_metadata: dict[str, Any],
    adjudications: pd.DataFrame,
    tables: Path,
    offline: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = SentenceTransformer(
        MODEL,
        revision=MODEL_REVISION,
        device="cpu",
        local_files_only=offline,
    )
    anchor_names = list(THEORY_ANCHORS) + list(SDG_ANCHORS)
    anchor_texts = list(THEORY_ANCHORS.values()) + list(SDG_ANCHORS.values())
    anchor_vectors = model.encode(
        anchor_texts,
        batch_size=16,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float64)
    theory_names = list(THEORY_ANCHORS)

    section_rows: list[dict[str, Any]] = []
    for row in corpus.itertuples(index=False):
        section_rows.extend(extract_theory_sections(row.paper_id, row.embedding_text))
    section_frame = pd.DataFrame(
        section_rows,
        columns=[
            "paper_id", "section_index", "section_type", "heading", "start_line",
            "end_line_exclusive", "section_char_count", "section_word_count",
            "section_sha256", "section_text",
        ],
    )
    save_csv(section_frame, tables / "theory_section_manifest.csv")

    chunk_tokens = int(embedding_metadata["embedding"]["chunk_tokens_excluding_special_tokens"])
    chunk_overlap = int(embedding_metadata["embedding"]["chunk_overlap_tokens"])
    chunk_texts, chunk_frame = chunk_theory_sections(
        model, section_frame, chunk_tokens, chunk_overlap
    )
    if not chunk_texts:
        raise ValueError("No eligible Abstract, Results/Findings, or Discussion chunks were extracted")
    chunk_vectors = model.encode(
        chunk_texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float64)
    if not np.all(np.isfinite(chunk_vectors)):
        raise ValueError("Theory-section chunk embeddings contain non-finite values")
    derived_embeddings = tables.parent / "derived_embeddings"
    derived_embeddings.mkdir(parents=True, exist_ok=True)
    np.save(derived_embeddings / "theory_section_chunk_embeddings.npy", chunk_vectors.astype(np.float32))
    save_csv(chunk_frame, tables / "theory_section_chunk_manifest.csv")

    chunk_similarity = np.einsum(
        "ij,kj->ik", chunk_vectors, anchor_vectors[: len(theory_names)], optimize=False
    )
    evidence_rows: list[dict[str, Any]] = []
    theory_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    sections_by_paper = {paper_id: group for paper_id, group in section_frame.groupby("paper_id", sort=False)}
    chunks_by_paper = {paper_id: group for paper_id, group in chunk_frame.groupby("paper_id", sort=False)}
    source_row_by_id = dict(zip(selection["paper_id"], selection["source_embedding_row"]))
    for paper_index, paper_id in enumerate(selection["paper_id"]):
        paper_sections = sections_by_paper.get(paper_id)
        paper_chunks = chunks_by_paper.get(paper_id)
        row: dict[str, Any] = {"paper_id": paper_id}
        if paper_sections is None or paper_chunks is None or paper_chunks.empty:
            row.update({name: math.nan for name in theory_names})
            row.update(
                {
                    "dominant_theory": "Not_inferable",
                    "runner_up_theory": "Not_inferable",
                    "theory_top1_similarity": math.nan,
                    "theory_runner_up_similarity": math.nan,
                    "theory_margin": math.nan,
                    "theory_low_margin_flag": pd.NA,
                    "theory_bootstrap_stability": math.nan,
                    "theory_assignment_confidence": "no_eligible_section",
                    "eligible_section_count": 0,
                    "eligible_section_types": "",
                    "eligible_section_char_count": 0,
                    "eligible_chunk_count": 0,
                }
            )
            for name in theory_names:
                row[f"bootstrap_probability_{name}"] = math.nan
            selected_text = ""
        else:
            global_indices = paper_chunks["theory_chunk_row"].to_numpy(dtype=int)
            paper_scores = chunk_similarity[global_indices]
            pooled = np.empty(len(theory_names), dtype=float)
            for theory_index, theory_name in enumerate(theory_names):
                theory_scores = paper_scores[:, theory_index]
                top_count = min(THEORY_TOP_CHUNKS, len(theory_scores))
                ranked_local = np.argsort(theory_scores)[::-1]
                pooled[theory_index] = float(np.mean(theory_scores[ranked_local[:top_count]]))
                row[f"top3_mean_{theory_name}"] = float(
                    np.mean(theory_scores[ranked_local[: min(3, len(theory_scores))]])
                )
                for evidence_rank, local_index in enumerate(
                    ranked_local[: min(THEORY_EVIDENCE_CHUNKS, len(theory_scores))], start=1
                ):
                    chunk_record = paper_chunks.iloc[int(local_index)]
                    evidence_rows.append(
                        {
                            "paper_id": paper_id,
                            "theory": theory_name,
                            "evidence_rank": evidence_rank,
                            "cosine_similarity": float(theory_scores[local_index]),
                            "section_type": chunk_record["section_type"],
                            "section_index": int(chunk_record["section_index"]),
                            "section_chunk_index": int(chunk_record["section_chunk_index"]),
                            "theory_chunk_row": int(chunk_record["theory_chunk_row"]),
                            "chunk_text": chunk_record["chunk_text"],
                        }
                    )
            order = np.argsort(pooled)[::-1]
            winner_index = int(order[0])
            runner_index = int(order[1])
            margin = float(pooled[winner_index] - pooled[runner_index])
            rng = np.random.default_rng(SEED + int(source_row_by_id[paper_id]))
            bootstrap_winners = np.empty(THEORY_BOOTSTRAP_REPLICATES, dtype=int)
            for replicate in range(THEORY_BOOTSTRAP_REPLICATES):
                sampled = paper_scores[rng.integers(0, len(paper_scores), size=len(paper_scores))]
                top_count = min(THEORY_TOP_CHUNKS, len(sampled))
                sampled_top = np.partition(sampled, len(sampled) - top_count, axis=0)[-top_count:]
                bootstrap_winners[replicate] = int(np.argmax(sampled_top.mean(axis=0)))
            bootstrap_probabilities = np.bincount(
                bootstrap_winners, minlength=len(theory_names)
            ) / THEORY_BOOTSTRAP_REPLICATES
            stability = float(bootstrap_probabilities[winner_index])
            if stability >= THEORY_STABLE_BOOTSTRAP and margin >= THEORY_LOW_MARGIN:
                confidence = "stable"
            elif stability >= THEORY_MODERATE_BOOTSTRAP and margin >= THEORY_MODERATE_MARGIN:
                confidence = "moderate"
            else:
                confidence = "ambiguous"
            row.update({name: float(pooled[index]) for index, name in enumerate(theory_names)})
            row.update(
                {
                    "dominant_theory": theory_names[winner_index],
                    "runner_up_theory": theory_names[runner_index],
                    "theory_top1_similarity": float(pooled[winner_index]),
                    "theory_runner_up_similarity": float(pooled[runner_index]),
                    "theory_margin": margin,
                    "theory_low_margin_flag": margin < THEORY_LOW_MARGIN,
                    "theory_bootstrap_stability": stability,
                    "theory_assignment_confidence": confidence,
                    "eligible_section_count": len(paper_sections),
                    "eligible_section_types": ";".join(sorted(set(paper_sections["section_type"]))),
                    "eligible_section_char_count": int(paper_sections["section_char_count"].sum()),
                    "eligible_chunk_count": len(paper_chunks),
                }
            )
            for index, name in enumerate(theory_names):
                row[f"bootstrap_probability_{name}"] = float(bootstrap_probabilities[index])
            selected_text = "\n".join(paper_sections["section_text"].astype(str))
        explicit = [
            name
            for name, patterns in THEORY_MENTION_PATTERNS.items()
            if any(re.search(pattern, selected_text, flags=re.IGNORECASE) for pattern in patterns)
        ]
        singleton = explicit[0] if len(explicit) == 1 else ""
        validation_rows.append(
            {
                "paper_id": paper_id,
                "eligible_section_count": int(row["eligible_section_count"]),
                "explicit_theory_mentions": ";".join(explicit),
                "explicit_theory_count": len(explicit),
                "embedding_inferred_theory": row["dominant_theory"],
                "top_theory_is_explicit": row["dominant_theory"] in explicit if explicit else pd.NA,
                "singleton_explicit_theory": singleton,
                "singleton_top1_match": row["dominant_theory"] == singleton if singleton else pd.NA,
            }
        )
        theory_rows.append(row)

    theory = pd.DataFrame(theory_rows)
    theory.rename(columns={"dominant_theory": "embedding_inferred_theory"}, inplace=True)
    included_adjudications = adjudications.loc[
        adjudications["corpus_eligible"],
        ["paper_id", "study_type", "manual_theory", "conceptual_focus", "adjudication_status", "adjudicated_by", "notes"],
    ].rename(
        columns={
            "study_type": "manual_study_type",
            "conceptual_focus": "manual_conceptual_focus",
            "adjudication_status": "manual_adjudication_status",
            "adjudicated_by": "manual_adjudicated_by",
            "notes": "manual_adjudication_notes",
        }
    )
    theory = theory.merge(included_adjudications, on="paper_id", how="left", validate="one_to_one")
    theory["final_theories"] = theory["embedding_inferred_theory"]
    manual_override = theory["manual_theory"].fillna("").ne("")
    theory.loc[manual_override, "final_theories"] = theory.loc[manual_override, "manual_theory"]
    theory["dominant_theory"] = theory["final_theories"].map(
        lambda value: parse_theory_labels(value)[0] if parse_theory_labels(value) else "Not_inferable"
    )
    theory["final_theory_count"] = theory["final_theories"].map(
        lambda value: len(parse_theory_labels(value))
    )
    theory["theory_assignment_source"] = np.where(
        manual_override, "human_adjudication", "embedding_inference"
    )
    theory["final_theory_is_formal"] = theory["final_theories"].map(
        lambda value: bool(parse_theory_labels(value))
        and all(label in FINAL_FORMAL_THEORY_LABELS for label in parse_theory_labels(value))
    )
    theory["theory_assignment_confidence_final"] = np.where(
        manual_override, "human_confirmed", theory["theory_assignment_confidence"]
    )
    adjudication_audit = adjudications.copy()
    adjudication_audit["present_in_final_corpus"] = adjudication_audit["paper_id"].isin(selection["paper_id"])
    save_csv(adjudication_audit, tables / "manual_adjudications_applied.csv")
    save_csv(theory, tables / "theory_similarity.csv")
    evidence = pd.DataFrame(evidence_rows)
    save_csv(evidence, tables / "theory_evidence_chunks.csv")
    validation = pd.DataFrame(validation_rows)
    save_csv(validation, tables / "theory_explicit_mention_validation.csv")
    primary_evidence = evidence.loc[evidence["evidence_rank"] == 1, [
        "paper_id", "theory", "section_type", "cosine_similarity", "chunk_text"
    ]].rename(
        columns={
            "theory": "embedding_inferred_theory",
            "section_type": "primary_evidence_section",
            "cosine_similarity": "primary_evidence_similarity",
            "chunk_text": "primary_evidence_text",
        }
    )
    review = (
        theory.merge(
            validation[["paper_id", "explicit_theory_mentions", "explicit_theory_count", "top_theory_is_explicit"]],
            on="paper_id",
            how="left",
        )
        .merge(selection[["paper_id", "source_sdg", "title"]], on="paper_id", how="left")
        .merge(primary_evidence, on=["paper_id", "embedding_inferred_theory"], how="left")
    )
    review_reasons: list[str] = []
    review_priorities: list[str] = []
    for item in review.itertuples(index=False):
        reasons: list[str] = []
        if item.embedding_inferred_theory == "Not_inferable":
            reasons.append("no_eligible_section")
        if item.theory_assignment_confidence in {"moderate", "ambiguous"}:
            reasons.append(f"{item.theory_assignment_confidence}_embedding_assignment")
        if item.explicit_theory_count > 0 and not bool(item.top_theory_is_explicit):
            reasons.append("embedding_conflicts_with_explicit_name")
        review_reasons.append(";".join(reasons))
        if "no_eligible_section" in reasons or "embedding_conflicts_with_explicit_name" in reasons:
            review_priorities.append("high")
        elif "ambiguous_embedding_assignment" in reasons:
            review_priorities.append("medium")
        elif reasons:
            review_priorities.append("low")
        else:
            review_priorities.append("")
    review["review_reason"] = review_reasons
    review["review_priority"] = review_priorities
    completed_manual_ids = set(adjudications.loc[adjudications["adjudication_status"] == "complete", "paper_id"])
    review = review.loc[
        (review["review_reason"] != "") & ~review["paper_id"].isin(completed_manual_ids),
        [
        "review_priority", "review_reason", "paper_id", "source_sdg", "title",
        "embedding_inferred_theory", "dominant_theory", "runner_up_theory", "theory_top1_similarity", "theory_margin",
        "theory_bootstrap_stability", "theory_assignment_confidence",
        "explicit_theory_mentions", "primary_evidence_section", "primary_evidence_similarity",
        "primary_evidence_text",
    ]]
    priority_order = pd.Categorical(review["review_priority"], ["high", "medium", "low"], ordered=True)
    review = review.assign(_priority_order=priority_order).sort_values(
        ["_priority_order", "theory_margin", "paper_id"], na_position="first"
    ).drop(columns="_priority_order")
    save_csv(review, tables / "theory_manual_review_queue.csv")
    eligible = theory["embedding_inferred_theory"] != "Not_inferable"
    mentioned = validation["explicit_theory_count"] > 0
    singleton = validation["explicit_theory_count"] == 1
    validation_summary = [
        {"scope": "overall", "metric": "corpus_documents", "value": len(theory)},
        {"scope": "overall", "metric": "documents_with_eligible_sections", "value": int(eligible.sum())},
        {"scope": "overall", "metric": "documents_without_eligible_sections", "value": int((~eligible).sum())},
        {"scope": "overall", "metric": "documents_with_explicit_theory_mentions", "value": int(mentioned.sum())},
        {"scope": "overall", "metric": "documents_with_single_explicit_theory", "value": int(singleton.sum())},
        {
            "scope": "overall",
            "metric": "top1_agreement_with_any_explicit_mention",
            "value": float(validation.loc[mentioned, "top_theory_is_explicit"].astype(bool).mean()) if mentioned.any() else math.nan,
        },
        {
            "scope": "overall",
            "metric": "top1_agreement_with_single_explicit_theory",
            "value": float(validation.loc[singleton, "singleton_top1_match"].astype(bool).mean()) if singleton.any() else math.nan,
        },
    ]
    for name in theory_names:
        theory_mentioned = validation["explicit_theory_mentions"].fillna("").str.split(";").apply(lambda values: name in values)
        validation_summary.extend(
            [
                {"scope": name, "metric": "documents_with_explicit_mention", "value": int(theory_mentioned.sum())},
                {
                    "scope": name,
                    "metric": "top1_recall_among_explicit_mentions",
                    "value": float((validation.loc[theory_mentioned, "embedding_inferred_theory"] == name).mean()) if theory_mentioned.any() else math.nan,
                },
            ]
        )
    save_csv(pd.DataFrame(validation_summary), tables / "theory_validation_summary.csv")

    sdg_names = list(SDG_ANCHORS)
    sdg_similarity = np.einsum(
        "ij,kj->ik", embeddings, anchor_vectors[len(theory_names) :], optimize=False
    )
    if not np.all(np.isfinite(sdg_similarity)):
        raise ValueError("SDG similarities contain non-finite values")
    sdg = pd.DataFrame(sdg_similarity, columns=sdg_names)
    sdg.insert(0, "paper_id", selection["paper_id"])
    sdg["source_sdg"] = selection["source_sdg"]
    sdg["semantic_sdg"] = np.where(sdg["SDG14"] >= sdg["SDG15"], "SDG14", "SDG15")
    sdg["sdg_margin"] = sdg["SDG14"] - sdg["SDG15"]
    sdg["source_semantic_agreement"] = sdg["source_sdg"] == sdg["semantic_sdg"]
    save_csv(sdg, tables / "sdg_similarity.csv")

    anchor_rows = [
        {"anchor_type": "theory", "anchor": name, "text": text}
        for name, text in THEORY_ANCHORS.items()
    ] + [
        {"anchor_type": "sdg", "anchor": name, "text": text}
        for name, text in SDG_ANCHORS.items()
    ]
    save_csv(pd.DataFrame(anchor_rows), tables / "anchor_definitions.csv")
    return theory, sdg


def cluster_documents(
    selection: pd.DataFrame,
    corpus: pd.DataFrame,
    embeddings: np.ndarray,
    tables: Path,
    figures: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    models: dict[int, KMeans] = {}
    for k in range(2, 13):
        model = KMeans(n_clusters=k, random_state=SEED, n_init=100, max_iter=1000, algorithm="lloyd").fit(embeddings)
        models[k] = model
        counts = np.bincount(model.labels_)
        diagnostics.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "cosine_silhouette": float(silhouette_score(embeddings, model.labels_, metric="cosine")),
                "smallest_cluster": int(counts.min()),
                "largest_cluster": int(counts.max()),
            }
        )
    selected_k = int(max(diagnostics, key=lambda row: (row["cosine_silhouette"], -row["k"]))["k"])
    kmeans = models[selected_k]
    labels = kmeans.labels_.astype(int)
    sample_silhouette = silhouette_samples(embeddings, labels, metric="cosine")
    distances = np.linalg.norm(embeddings - kmeans.cluster_centers_[labels], axis=1)

    stability_scores: list[float] = []
    for offset in range(1, 21):
        alternate = KMeans(n_clusters=selected_k, random_state=SEED + offset, n_init=30, max_iter=1000).fit_predict(embeddings)
        stability_scores.append(float(adjusted_rand_score(labels, alternate)))

    hdbscan = HDBSCAN(min_cluster_size=5, min_samples=3, metric="euclidean", cluster_selection_method="eom").fit(embeddings)
    hdb_labels = hdbscan.labels_.astype(int)
    hdb_probability = np.asarray(hdbscan.probabilities_, dtype=float)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words=STOP_WORDS,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.72,
        max_features=30000,
        sublinear_tf=True,
        norm="l2",
    )
    tfidf = vectorizer.fit_transform(corpus["embedding_text"].fillna("").astype(str))
    features = np.asarray(vectorizer.get_feature_names_out())
    term_rows: list[dict[str, Any]] = []
    cluster_terms: dict[int, list[str]] = {}
    for cluster in sorted(np.unique(labels)):
        scores = np.asarray(tfidf[labels == cluster].mean(axis=0)).ravel()
        ranked = np.argsort(scores)[::-1]
        selected = [str(features[index]) for index in ranked[:20] if scores[index] > 0]
        cluster_terms[int(cluster)] = selected
        for rank, index in enumerate(ranked[:20], start=1):
            if scores[index] <= 0:
                continue
            term_rows.append(
                {"cluster": int(cluster), "rank": rank, "term": str(features[index]), "mean_tfidf": float(scores[index])}
            )
    labels_by_cluster = {cluster: automatic_cluster_label(terms) for cluster, terms in cluster_terms.items()}

    pca = PCA(n_components=2, svd_solver="full")
    coordinates = pca.fit_transform(embeddings)
    assignments = selection[["paper_id", "source_sdg", "title", "year", "publication_type"]].copy()
    assignments["cluster"] = labels
    assignments["automatic_theme_label"] = [labels_by_cluster[int(value)] for value in labels]
    assignments["distance_to_centroid"] = distances
    assignments["silhouette_cosine"] = sample_silhouette
    assignments["hdbscan_cluster"] = hdb_labels
    assignments["hdbscan_probability"] = hdb_probability
    assignments["pca_x"] = coordinates[:, 0]
    assignments["pca_y"] = coordinates[:, 1]
    save_csv(assignments, tables / "cluster_assignments.csv")
    save_csv(pd.DataFrame(diagnostics), tables / "clustering_diagnostics.csv")
    save_csv(pd.DataFrame(term_rows), tables / "cluster_terms.csv")

    summary_rows: list[dict[str, Any]] = []
    representatives: list[dict[str, Any]] = []
    similarity = np.einsum("ik,jk->ij", embeddings, embeddings, optimize=False)
    for cluster in sorted(np.unique(labels)):
        indices = np.flatnonzero(labels == cluster)
        nearest = indices[np.argsort(distances[indices])[:3]]
        sub = similarity[np.ix_(indices, indices)]
        upper = sub[np.triu_indices_from(sub, k=1)]
        sdg_counts = Counter(assignments.iloc[indices]["source_sdg"])
        summary_rows.append(
            {
                "cluster": int(cluster),
                "automatic_theme_label": labels_by_cluster[int(cluster)],
                "documents": int(len(indices)),
                "percent_of_corpus": float(100 * len(indices) / len(assignments)),
                "SDG14": int(sdg_counts.get("SDG14", 0)),
                "SDG15": int(sdg_counts.get("SDG15", 0)),
                "mean_silhouette_cosine": float(sample_silhouette[indices].mean()),
                "within_cluster_mean_cosine": float(upper.mean()) if len(upper) else math.nan,
                "top_terms": "; ".join(cluster_terms[int(cluster)][:12]),
                "representative_paper_ids": "; ".join(assignments.iloc[nearest]["paper_id"]),
            }
        )
        for rank, index in enumerate(nearest, start=1):
            representatives.append(
                {
                    "cluster": int(cluster),
                    "automatic_theme_label": labels_by_cluster[int(cluster)],
                    "rank": rank,
                    "paper_id": assignments.iloc[index]["paper_id"],
                    "title": assignments.iloc[index]["title"],
                    "source_sdg": assignments.iloc[index]["source_sdg"],
                    "distance_to_centroid": float(distances[index]),
                }
            )
    summary = pd.DataFrame(summary_rows)
    save_csv(summary, tables / "cluster_summary.csv")
    save_csv(pd.DataFrame(representatives), tables / "representative_papers.csv")

    pair_same = labels[:, None] == labels[None, :]
    upper_triangle = np.triu(np.ones_like(pair_same, dtype=bool), k=1)
    within = similarity[pair_same & upper_triangle]
    between = similarity[(~pair_same) & upper_triangle]
    clustering_metrics = {
        "selected_k": selected_k,
        "selected_cosine_silhouette": float(silhouette_score(embeddings, labels, metric="cosine")),
        "stability_mean_adjusted_rand": float(np.mean(stability_scores)),
        "stability_min_adjusted_rand": float(np.min(stability_scores)),
        "hdbscan_clusters": int(len(set(hdb_labels) - {-1})),
        "hdbscan_noise_records": int(np.sum(hdb_labels == -1)),
        "within_cluster_mean_similarity": float(within.mean()),
        "between_cluster_mean_similarity": float(between.mean()),
        "within_between_difference": float(within.mean() - between.mean()),
        "pca_variance_pc1": float(pca.explained_variance_ratio_[0]),
        "pca_variance_pc2": float(pca.explained_variance_ratio_[1]),
    }

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot([row["k"] for row in diagnostics], [row["cosine_silhouette"] for row in diagnostics], marker="o", color="#174A6E", linewidth=2)
    ax.scatter([selected_k], [clustering_metrics["selected_cosine_silhouette"]], color="#E76F51", s=90, zorder=5)
    style_axes(ax, "Cluster-count selection", "KMeans evaluated with cosine silhouette; maximum determines k")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Cosine silhouette")
    ax.set_xticks(range(2, 13))
    save_figure(fig, figures / "supplement_cluster_selection.png")

    fig, ax = plt.subplots(figsize=(10, 7))
    for cluster in sorted(np.unique(labels)):
        mask = labels == cluster
        ax.scatter(coordinates[mask, 0], coordinates[mask, 1], s=48, alpha=0.82, color=COLORS[int(cluster)], label=f"{cluster}: {labels_by_cluster[int(cluster)]}", edgecolor="white", linewidth=0.4)
    style_axes(ax, "Semantic map of the retained corpus", "PCA display of pooled document embeddings; colors show KMeans groups")
    ax.set_xlabel(f"PC1 ({100 * pca.explained_variance_ratio_[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({100 * pca.explained_variance_ratio_[1]:.1f}% variance)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=8)
    save_figure(fig, figures / "supplement_embedding_clusters.png")
    return assignments, summary, clustering_metrics


def extract_technologies(
    selection: pd.DataFrame,
    corpus: pd.DataFrame,
    tables: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    compiled = {
        category: [(pattern, re.compile(pattern, flags=re.IGNORECASE)) for pattern in patterns]
        for category, patterns in TECHNOLOGY_PATTERNS.items()
    }
    matrix_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    for selection_row, text in zip(selection.to_dict("records"), corpus["embedding_text"].fillna("").astype(str)):
        normalized = normalize_for_matching(text)
        row: dict[str, Any] = {
            "paper_id": selection_row["paper_id"],
            "source_sdg": selection_row["source_sdg"],
        }
        for category, patterns in compiled.items():
            category_count = 0
            matched_terms: list[str] = []
            for raw_pattern, pattern in patterns:
                matches = list(pattern.finditer(normalized))
                if matches:
                    category_count += len(matches)
                    matched_terms.append(matches[0].group(0))
                    evidence_rows.append(
                        {
                            "paper_id": selection_row["paper_id"],
                            "source_sdg": selection_row["source_sdg"],
                            "technology": category,
                            "matched_term": matches[0].group(0),
                            "pattern": raw_pattern,
                            "occurrences": len(matches),
                        }
                    )
            row[category] = int(category_count > 0)
            row[f"{category}_mentions"] = category_count
            row[f"{category}_matched_terms"] = "; ".join(sorted(set(matched_terms)))
        matrix_rows.append(row)
    matrix = pd.DataFrame(matrix_rows)
    evidence = pd.DataFrame(evidence_rows)
    save_csv(matrix, tables / "technology_matrix.csv")
    save_csv(evidence, tables / "technology_evidence.csv")

    categories = list(TECHNOLOGY_PATTERNS)
    distribution = pd.DataFrame(
        {
            "technology": categories,
            "documents": [int(matrix[category].sum()) for category in categories],
            "percent_of_corpus": [float(100 * matrix[category].mean()) for category in categories],
            "total_mentions": [int(matrix[f"{category}_mentions"].sum()) for category in categories],
        }
    ).sort_values(["documents", "technology"], ascending=[False, True])
    save_csv(distribution, tables / "table_technology_distribution.csv")

    cross = matrix.groupby("source_sdg")[categories].sum().T.reindex(columns=["SDG14", "SDG15"], fill_value=0)
    cross.index.name = "technology"
    save_csv(cross.reset_index(), tables / "table_technology_by_sdg.csv")
    cooccurrence = matrix[categories].T.dot(matrix[categories]).astype(int)
    np.fill_diagonal(cooccurrence.values, 0)
    cooccurrence.index.name = "technology"
    save_csv(cooccurrence.reset_index(), tables / "technology_cooccurrence_matrix.csv")
    return matrix, distribution, cooccurrence


def classify_by_lexicon(text: str, lexicon: dict[str, list[str]], fallback: str) -> str:
    normalized = normalize_for_matching(text)
    best: tuple[int, int, str] | None = None
    for priority, (category, terms) in enumerate(lexicon.items()):
        positions = [normalized.find(term) for term in terms if normalized.find(term) >= 0]
        if not positions:
            continue
        candidate = (min(positions), priority, category)
        if best is None or candidate < best:
            best = candidate
    return fallback if best is None else best[2]


def split_sentences(text: str) -> list[str]:
    text = text.replace("\u00ad", "")
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text) if sentence.strip()]


def claim_zones(text: str) -> list[tuple[str, str]]:
    """Return abstract and conclusion text using fixed heading rules."""
    zones: list[tuple[str, str]] = []
    abstract = re.search(
        r"(?is)\babstract\s*:?[ \t]*(.*?)(?=\n\s*(?:keywords?|key\s+words|index\s+terms|"
        r"(?:1\.?\s*)?introduction)\b)",
        text,
    )
    if abstract and len(abstract.group(1).split()) >= 20:
        zones.append(("abstract", abstract.group(1)[:7000]))

    conclusion_heading = re.compile(
        r"(?im)^\s*(?:\d+(?:\.\d+)*\.?\s*)?(?:conclusions?|discussion\s+and\s+conclusions?|"
        r"concluding\s+remarks)\s*$"
    )
    end_heading = re.compile(
        r"(?im)^\s*(?:\d+(?:\.\d+)*\.?\s*)?(?:references|bibliography|acknowledg(?:e)?ments?)\s*$"
    )
    matches = list(conclusion_heading.finditer(text))
    if matches:
        start = matches[-1].end()
        ending = end_heading.search(text, start)
        end = ending.start() if ending else min(len(text), start + 9000)
        conclusion = text[start:end]
        if len(conclusion.split()) >= 20:
            zones.append(("conclusion", conclusion[:9000]))
    return zones


def extract_claims(
    selection: pd.DataFrame,
    corpus: pd.DataFrame,
    assignments: pd.DataFrame,
    tables: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    compiled = {name: re.compile(pattern, flags=re.IGNORECASE) for name, pattern in CLAIM_VERB_PATTERNS.items()}
    excluded_fragments = ("copyright", "all rights reserved", "doi:", "references", "bibliography", "hypothesis")
    rows: list[dict[str, Any]] = []
    for paper_id, source_sdg, text in zip(selection["paper_id"], selection["source_sdg"], corpus["embedding_text"].fillna("").astype(str)):
        for zone_priority, (zone, zone_text) in enumerate(claim_zones(text)):
            for sentence_index, sentence in enumerate(split_sentences(zone_text)):
                words = sentence.split()
                normalized = sentence.lower()
                if len(words) < 8 or len(words) > 80 or any(fragment in normalized for fragment in excluded_fragments):
                    continue
                for canonical, pattern in compiled.items():
                    match = pattern.search(sentence)
                    if not match:
                        continue
                    before = sentence[: match.start()].strip(" ,;:()[]")
                    after = sentence[match.end() :].strip(" ,;:()[]")
                    subject = " ".join(before.split()[-12:])
                    obj = " ".join(after.split()[:18])
                    if len(subject.split()) < 1 or len(obj.split()) < 2:
                        continue
                    subject_category = classify_by_lexicon(subject, SUBJECT_CATEGORIES, "Other_Subject")
                    outcome_category = classify_by_lexicon(obj + " " + sentence, OUTCOME_CATEGORIES, "Other_Outcome")
                    evidence_matches = len(CLAIM_EVIDENCE_PATTERN.findall(sentence))
                    if evidence_matches == 0:
                        continue
                    rows.append(
                        {
                            "paper_id": paper_id,
                            "source_sdg": source_sdg,
                            "zone": zone,
                            "zone_priority": zone_priority,
                            "sentence_index": sentence_index,
                            "evidence_cue_count": evidence_matches,
                            "canonical_verb": canonical,
                            "matched_verb": match.group(0),
                            "subject_surface": subject,
                            "object_surface": obj,
                            "subject_category": subject_category,
                            "outcome_category": outcome_category,
                            "claim_sentence": sentence,
                        }
                    )
    claims = pd.DataFrame(rows)
    if claims.empty:
        claims = pd.DataFrame(columns=["paper_id", "source_sdg", "zone", "zone_priority", "sentence_index", "evidence_cue_count", "canonical_verb", "matched_verb", "subject_surface", "object_surface", "subject_category", "outcome_category", "claim_sentence"])
    else:
        claims["normalized_claim"] = claims["claim_sentence"].map(normalize_for_matching)
        claims.sort_values(
            ["paper_id", "canonical_verb", "evidence_cue_count", "zone_priority", "sentence_index"],
            ascending=[True, True, False, True, True],
            inplace=True,
        )
        claims.drop_duplicates(["paper_id", "canonical_verb", "normalized_claim"], inplace=True)
        claims.drop(columns="normalized_claim", inplace=True)
    save_csv(claims, tables / "rule_based_claims.csv")

    verbs = list(CLAIM_VERB_PATTERNS)
    matrix = pd.DataFrame(0, index=selection["paper_id"], columns=verbs, dtype=int)
    if not claims.empty:
        pivot = claims.groupby(["paper_id", "canonical_verb"]).size().unstack(fill_value=0)
        matrix.loc[pivot.index, pivot.columns] = pivot
    matrix.insert(0, "source_sdg", selection.set_index("paper_id").loc[matrix.index, "source_sdg"])
    matrix.insert(0, "paper_id", matrix.index)
    matrix.reset_index(drop=True, inplace=True)
    save_csv(matrix, tables / "claim_matrix.csv")

    pattern_rows = []
    total_claims = max(1, len(claims))
    for verb in verbs:
        count = int((claims["canonical_verb"] == verb).sum()) if not claims.empty else 0
        pattern_rows.append(
            {
                "canonical_verb": verb,
                "claims": count,
                "percent_of_claims": 100 * count / total_claims,
                "documents": int(claims.loc[claims["canonical_verb"] == verb, "paper_id"].nunique()) if not claims.empty else 0,
            }
        )
    patterns = pd.DataFrame(pattern_rows).sort_values("claims", ascending=False)
    save_csv(patterns, tables / "table_claim_patterns.csv")

    joined = claims.merge(assignments[["paper_id", "cluster"]], on="paper_id", how="left")
    cluster_claims = joined.groupby(["cluster", "canonical_verb"]).size().unstack(fill_value=0).reindex(columns=verbs, fill_value=0)
    all_clusters = sorted(assignments["cluster"].unique())
    cluster_claims = cluster_claims.reindex(all_clusters, fill_value=0)
    cluster_claims.index.name = "cluster"
    save_csv(cluster_claims.reset_index(), tables / "claim_patterns_by_cluster.csv")
    row_percent = cluster_claims.div(cluster_claims.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
    row_percent.index.name = "cluster"
    save_csv(row_percent.reset_index(), tables / "claim_patterns_by_cluster_percent.csv")

    structures = (
        claims.groupby(["subject_category", "canonical_verb", "outcome_category"]).size().reset_index(name="claims")
        if not claims.empty
        else pd.DataFrame(columns=["subject_category", "canonical_verb", "outcome_category", "claims"])
    )
    structures.sort_values("claims", ascending=False, inplace=True)
    save_csv(structures, tables / "claim_structures.csv")
    return claims, matrix, row_percent


def graph_metrics(
    graph: nx.Graph,
    source_by_id: dict[str, str],
    similarity_threshold: float,
) -> dict[str, Any]:
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    components = list(nx.connected_components(graph))
    largest = max((len(component) for component in components), default=0)
    cross_edges = sum(source_by_id[u] != source_by_id[v] for u, v in graph.edges())
    n14 = sum(value == "SDG14" for value in source_by_id.values())
    n15 = sum(value == "SDG15" for value in source_by_id.values())
    source_partition = [
        {node for node in graph if source_by_id[node] == "SDG14"},
        {node for node in graph if source_by_id[node] == "SDG15"},
    ]
    source_modularity = nx.community.modularity(graph, source_partition, weight="weight") if m else math.nan
    assortativity = nx.attribute_assortativity_coefficient(graph, "source_sdg") if m else math.nan
    return {
        "similarity_threshold": similarity_threshold,
        "nodes": n,
        "edges": m,
        "density": float(nx.density(graph)),
        "components": len(components),
        "largest_component_nodes": largest,
        "largest_component_percent": 100 * largest / n if n else math.nan,
        "isolates": int(nx.number_of_isolates(graph)),
        "transitivity": float(nx.transitivity(graph)) if m else 0.0,
        "average_clustering": float(nx.average_clustering(graph, weight="weight")) if m else 0.0,
        "cross_sdg_edges": int(cross_edges),
        "cross_sdg_edge_share": cross_edges / m if m else math.nan,
        "structural_coupling_index": cross_edges / (n14 * n15) if n14 and n15 else math.nan,
        "source_sdg_modularity": float(source_modularity),
        "source_sdg_assortativity": float(assortativity),
    }


def build_networks(
    selection: pd.DataFrame,
    embeddings: np.ndarray,
    theory: pd.DataFrame,
    technology_matrix: pd.DataFrame,
    tables: Path,
) -> tuple[nx.Graph, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ids = selection["paper_id"].tolist()
    source_by_id = dict(zip(selection["paper_id"], selection["source_sdg"]))
    title_by_id = dict(zip(selection["paper_id"], selection["title"]))
    similarity = np.einsum("ik,jk->ij", embeddings, embeddings, optimize=False)
    thresholds = [round(float(value), 3) for value in np.arange(0.50, 0.801, 0.025)]
    sensitivity_rows: list[dict[str, Any]] = []
    primary_graph: nx.Graph | None = None
    for threshold in thresholds:
        graph = nx.Graph()
        for paper_id in ids:
            graph.add_node(paper_id, source_sdg=source_by_id[paper_id], title=title_by_id[paper_id])
        upper_i, upper_j = np.where(np.triu(similarity >= threshold, k=1))
        for i, j in zip(upper_i, upper_j):
            graph.add_edge(ids[i], ids[j], weight=float(similarity[i, j]))
        sensitivity_rows.append(graph_metrics(graph, source_by_id, threshold))
        if abs(threshold - PRIMARY_SIMILARITY_THRESHOLD) < 1e-9:
            primary_graph = graph
    if primary_graph is None:
        raise RuntimeError("Primary similarity threshold was not evaluated")
    sensitivity = pd.DataFrame(sensitivity_rows)
    save_csv(sensitivity, tables / "network_threshold_sensitivity.csv")

    edge_rows = []
    for u, v, data in primary_graph.edges(data=True):
        edge_rows.append(
            {
                "paper_i": u,
                "paper_j": v,
                "cosine_similarity": float(data["weight"]),
                "source_sdg_i": source_by_id[u],
                "source_sdg_j": source_by_id[v],
                "cross_sdg": source_by_id[u] != source_by_id[v],
            }
        )
    edges = pd.DataFrame(edge_rows, columns=["paper_i", "paper_j", "cosine_similarity", "source_sdg_i", "source_sdg_j", "cross_sdg"])
    save_csv(edges, tables / "paper_similarity_edges.csv")

    betweenness = nx.betweenness_centrality(primary_graph, normalized=True, weight=None)
    weighted_degree = dict(primary_graph.degree(weight="weight"))
    bridge_rows: list[dict[str, Any]] = []
    for node in ids:
        cross_neighbors = [neighbor for neighbor in primary_graph.neighbors(node) if source_by_id[neighbor] != source_by_id[node]]
        bridge_rows.append(
            {
                "paper_id": node,
                "source_sdg": source_by_id[node],
                "title": title_by_id[node],
                "degree": int(primary_graph.degree(node)),
                "weighted_degree": float(weighted_degree[node]),
                "betweenness": float(betweenness[node]),
                "cross_sdg_degree": len(cross_neighbors),
                "cross_sdg_weight": float(sum(primary_graph[node][neighbor]["weight"] for neighbor in cross_neighbors)),
                "is_cross_sdg_bridge": bool(cross_neighbors),
            }
        )
    bridges = pd.DataFrame(bridge_rows).sort_values(["is_cross_sdg_bridge", "betweenness", "cross_sdg_degree"], ascending=[False, False, False])
    save_csv(bridges, tables / "bridge_nodes.csv")

    if primary_graph.number_of_edges():
        communities = nx.community.louvain_communities(primary_graph, weight="weight", seed=SEED)
        community_modularity = float(nx.community.modularity(primary_graph, communities, weight="weight"))
    else:
        communities = [{node} for node in ids]
        community_modularity = math.nan
    community_map = {node: index for index, community in enumerate(communities) for node in community}
    community_frame = selection[["paper_id", "source_sdg", "title"]].copy()
    community_frame["louvain_community"] = community_frame["paper_id"].map(community_map)
    save_csv(community_frame, tables / "community_assignments.csv")

    primary_metrics = graph_metrics(primary_graph, source_by_id, PRIMARY_SIMILARITY_THRESHOLD)
    primary_metrics["louvain_communities"] = len(communities)
    primary_metrics["louvain_modularity"] = community_modularity
    null_transitivity = []
    for iteration in range(200):
        null = nx.gnm_random_graph(primary_graph.number_of_nodes(), primary_graph.number_of_edges(), seed=SEED + iteration)
        null_transitivity.append(nx.transitivity(null))
    primary_metrics["er_null_transitivity_mean"] = float(np.mean(null_transitivity))
    primary_metrics["er_null_transitivity_sd"] = float(np.std(null_transitivity, ddof=1))

    prediction_rows: list[dict[str, Any]] = []
    index_by_id = {paper_id: index for index, paper_id in enumerate(ids)}
    for u, v, score in nx.adamic_adar_index(primary_graph):
        if score <= 0:
            continue
        if index_by_id[u] > index_by_id[v]:
            u, v = v, u
        i, j = index_by_id[u], index_by_id[v]
        prediction_rows.append(
            {
                "paper_i": u,
                "paper_j": v,
                "adamic_adar_score": float(score),
                "cosine_similarity": float(similarity[i, j]),
                "source_sdg_i": source_by_id[u],
                "source_sdg_j": source_by_id[v],
                "cross_sdg": source_by_id[u] != source_by_id[v],
            }
        )
    predictions = pd.DataFrame(prediction_rows, columns=["paper_i", "paper_j", "adamic_adar_score", "cosine_similarity", "source_sdg_i", "source_sdg_j", "cross_sdg"])
    if not predictions.empty:
        predictions.sort_values(
            ["adamic_adar_score", "cosine_similarity", "paper_i", "paper_j"],
            ascending=[False, False, True, True],
            inplace=True,
        )
        predictions = predictions.head(100)
    save_csv(predictions, tables / "exploratory_link_prediction_candidates.csv")

    theory_by_id = {
        row.paper_id: {
            label
            for label in parse_theory_labels(row.final_theories)
            if label not in NON_FORMAL_THEORY_LABELS
        }
        for row in theory[["paper_id", "final_theories"]].itertuples(index=False)
    }
    semantic_pairs = {tuple(sorted((u, v))) for u, v in primary_graph.edges()}
    theory_pairs = {
        tuple(sorted((u, v)))
        for u, v in combinations(ids, 2)
        if theory_by_id[u] & theory_by_id[v]
    }
    sdg_pairs = {
        tuple(sorted((u, v)))
        for u, v in combinations(ids, 2)
        if source_by_id[u] == source_by_id[v]
    }
    layers = {"semantic": semantic_pairs, "shared_theory": theory_pairs, "shared_sdg": sdg_pairs}
    total_pairs = len(ids) * (len(ids) - 1) / 2
    layer_summary = pd.DataFrame(
        [
            {"layer": name, "edges": len(pairs), "density": len(pairs) / total_pairs}
            for name, pairs in layers.items()
        ]
    )
    save_csv(layer_summary, tables / "multiplex_layer_summary.csv")
    overlap_rows = []
    for a, b in combinations(layers, 2):
        union = layers[a] | layers[b]
        overlap_rows.append(
            {
                "layer_a": a,
                "layer_b": b,
                "intersection_edges": len(layers[a] & layers[b]),
                "union_edges": len(union),
                "jaccard_overlap": len(layers[a] & layers[b]) / len(union) if union else math.nan,
            }
        )
    save_csv(pd.DataFrame(overlap_rows), tables / "multiplex_layer_overlap.csv")
    for name, pairs in layers.items():
        save_csv(pd.DataFrame(sorted(pairs), columns=["paper_i", "paper_j"]), tables / f"multiplex_{name}_edges.csv")
    return primary_graph, bridges, sensitivity, primary_metrics


def build_knowledge_graph(
    selection: pd.DataFrame,
    theory: pd.DataFrame,
    technology_matrix: pd.DataFrame,
    tables: Path,
) -> tuple[nx.Graph, pd.DataFrame]:
    theory_by_id = {
        row.paper_id: [
            label
            for label in parse_theory_labels(row.final_theories)
            if label not in NON_FORMAL_THEORY_LABELS
        ]
        for row in theory[["paper_id", "final_theories"]].itertuples(index=False)
    }
    theory_source_by_id = dict(zip(theory["paper_id"], theory["theory_assignment_source"]))
    tech_by_id = technology_matrix.set_index("paper_id")
    categories = list(TECHNOLOGY_PATTERNS)
    triples: list[dict[str, Any]] = []
    for row in selection.itertuples(index=False):
        for theory_label in theory_by_id[row.paper_id]:
            triples.append(
                {
                    "subject": row.paper_id,
                    "relation": (
                        "human_adjudicated_theory"
                        if theory_source_by_id[row.paper_id] == "human_adjudication"
                        else "embedding_inferred_theory"
                    ),
                    "object": theory_label,
                    "object_type": "theory",
                }
            )
        triples.append({"subject": row.paper_id, "relation": "targets_sdg", "object": row.source_sdg, "object_type": "sdg"})
        for technology in categories:
            if int(tech_by_id.loc[row.paper_id, technology]):
                triples.append({"subject": row.paper_id, "relation": "uses_technology", "object": technology, "object_type": "technology"})
    triples_frame = pd.DataFrame(triples)
    save_csv(triples_frame, tables / "knowledge_graph_triples.csv")
    graph = nx.Graph()
    for row in selection.itertuples(index=False):
        graph.add_node(row.paper_id, node_type="paper", source_sdg=row.source_sdg)
    for row in triples:
        object_id = f"{row['object_type']}::{row['object']}"
        graph.add_node(object_id, node_type=row["object_type"], label=row["object"])
        graph.add_edge(row["subject"], object_id, relation=row["relation"])
    centrality = nx.degree_centrality(graph)
    centrality_rows = []
    for node, data in graph.nodes(data=True):
        if data.get("node_type") == "paper":
            continue
        centrality_rows.append(
            {
                "node": data.get("label", node),
                "node_type": data.get("node_type"),
                "degree": int(graph.degree(node)),
                "degree_centrality": float(centrality[node]),
            }
        )
    centrality_frame = pd.DataFrame(centrality_rows).sort_values("degree", ascending=False)
    save_csv(centrality_frame, tables / "knowledge_graph_concept_centrality.csv")
    return graph, triples_frame


def statistical_tables(
    selection: pd.DataFrame,
    theory: pd.DataFrame,
    sdg: pd.DataFrame,
    technology_distribution: pd.DataFrame,
    claims: pd.DataFrame,
    claim_matrix: pd.DataFrame,
    clustering_metrics: dict[str, Any],
    network_metrics: dict[str, Any],
    tables: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    theory_memberships = explode_final_theories(theory)
    theory_distribution = theory_memberships["theory"].value_counts().rename_axis("theory").reset_index(name="count")
    theory_distribution["percent"] = 100 * theory_distribution["count"] / len(theory)
    save_csv(theory_distribution, tables / "table_1_theory_distribution.csv")

    source_distribution = selection["source_sdg"].value_counts().reindex(["SDG14", "SDG15"]).rename_axis("sdg").reset_index(name="count")
    source_distribution["percent"] = 100 * source_distribution["count"] / len(selection)
    semantic_counts = sdg["semantic_sdg"].value_counts()
    source_distribution["semantic_assignment_count"] = source_distribution["sdg"].map(semantic_counts).fillna(0).astype(int)
    save_csv(source_distribution, tables / "table_2_sdg_distribution.csv")

    theory_sdg_memberships = theory_memberships.merge(
        selection[["paper_id", "source_sdg"]], on="paper_id", how="left", validate="many_to_one"
    )
    theory_sdg = pd.crosstab(theory_sdg_memberships["theory"], theory_sdg_memberships["source_sdg"]).reindex(columns=["SDG14", "SDG15"], fill_value=0)
    theory_sdg.index.name = "theory"
    save_csv(theory_sdg.reset_index(), tables / "table_theory_by_sdg.csv")

    margin = sdg["sdg_margin"].to_numpy(dtype=float)
    t_result = ttest_1samp(margin, popmean=0.0)
    skew_value = float(skew(margin, bias=False))
    skew_result = skewtest(margin)
    stats_rows = [
        {"statistic": "count", "value": len(margin)},
        {"statistic": "mean", "value": float(np.mean(margin))},
        {"statistic": "standard_deviation", "value": float(np.std(margin, ddof=1))},
        {"statistic": "minimum", "value": float(np.min(margin))},
        {"statistic": "percentile_25", "value": float(np.quantile(margin, 0.25))},
        {"statistic": "median", "value": float(np.median(margin))},
        {"statistic": "percentile_75", "value": float(np.quantile(margin, 0.75))},
        {"statistic": "maximum", "value": float(np.max(margin))},
        {"statistic": "one_sample_t", "value": float(t_result.statistic)},
        {"statistic": "one_sample_t_p_value", "value": float(t_result.pvalue)},
        {"statistic": "cohen_d", "value": float(np.mean(margin) / np.std(margin, ddof=1))},
        {"statistic": "skewness", "value": skew_value},
        {"statistic": "skew_test_p_value", "value": float(skew_result.pvalue)},
        {"statistic": "sarle_bimodality_coefficient", "value": sarle_bimodality(margin)},
        {"statistic": "source_semantic_agreement_percent", "value": float(100 * sdg["source_semantic_agreement"].mean())},
    ]
    margin_stats = pd.DataFrame(stats_rows)
    save_csv(margin_stats, tables / "table_3_sdg_margin_statistics.csv")

    theory_counts = (
        theory_memberships.loc[theory_memberships["theory"].isin(FINAL_FORMAL_THEORY_LABELS), "theory"]
        .value_counts()
        .reindex(FINAL_FORMAL_THEORY_LABELS, fill_value=0)
        .to_numpy(dtype=float)
    )
    technology_counts = technology_distribution["documents"].to_numpy(dtype=float)
    claims_by_verb = claims["canonical_verb"].value_counts().reindex(list(CLAIM_VERB_PATTERNS), fill_value=0).to_numpy(dtype=float)
    numeric_claims = claim_matrix[list(CLAIM_VERB_PATTERNS)].to_numpy(dtype=float)
    covered = np.linalg.norm(numeric_claims, axis=1) > 0
    claim_similarity_mean = math.nan
    if covered.sum() >= 2:
        normalized_claims = numeric_claims[covered] / np.linalg.norm(numeric_claims[covered], axis=1, keepdims=True)
        claim_similarity = np.einsum("ik,jk->ij", normalized_claims, normalized_claims, optimize=False)
        claim_similarity_mean = float(claim_similarity[np.triu_indices_from(claim_similarity, k=1)].mean())

    metric_definitions = [
        ("Corpus records", len(selection), "Retained records analyzed"),
        ("SDG semantic-source agreement", float(sdg["source_semantic_agreement"].mean()), "Share of records whose anchor-based SDG matches the source folder"),
        ("SDG margin Cohen d", float(np.mean(margin) / np.std(margin, ddof=1)), "Mean SDG14-minus-SDG15 similarity divided by sample SD"),
        ("SDG margin bimodality coefficient", sarle_bimodality(margin), "Finite-sample Sarle coefficient; 0.555 is the conventional comparison value"),
        ("Technology Gini", gini(technology_counts), "Gini coefficient across six document-frequency counts"),
        ("Technology normalized entropy", normalized_entropy(technology_counts), "Shannon entropy divided by log of six categories"),
        ("Theory Gini", gini(theory_counts), f"Gini coefficient across {len(FINAL_FORMAL_THEORY_LABELS)} final formal-theory counts; non-formal labels excluded"),
        ("Theory normalized entropy", normalized_entropy(theory_counts), f"Shannon entropy divided by log of {len(FINAL_FORMAL_THEORY_LABELS)} final formal theories; non-formal labels excluded"),
        ("Embedding section-not-inferable share", float((theory["embedding_inferred_theory"] == "Not_inferable").mean()), "Share without a detected Abstract, Results/Findings, or Discussion section"),
        ("Final no-formal-theory share", float((theory["final_theories"] == "No_formal_theory").mean()), f"Share manually classified as having no formal theory in the {len(FINAL_FORMAL_THEORY_LABELS)}-label final taxonomy"),
        ("Theory low-margin assignment share", float(theory["theory_low_margin_flag"].dropna().astype(bool).mean()), "Share of inferable records whose pooled top theory score exceeds the runner-up by less than 0.03"),
        ("Theory stable assignment share", float((theory.loc[theory["embedding_inferred_theory"] != "Not_inferable", "theory_assignment_confidence"] == "stable").mean()), "Share of embedding-inferable records meeting both the 0.03 margin and 0.80 bootstrap-stability thresholds"),
        ("Network density", network_metrics["density"], f"Observed edge share at cosine similarity >= {PRIMARY_SIMILARITY_THRESHOLD}"),
        ("Network components", network_metrics["components"], "Connected components including isolates"),
        ("Largest component share", network_metrics["largest_component_percent"] / 100, "Largest connected component divided by all papers"),
        ("Network transitivity", network_metrics["transitivity"], "Global triangle closure"),
        ("ER-null transitivity mean", network_metrics["er_null_transitivity_mean"], "Mean over 200 G(n,m) graphs with fixed seeds"),
        ("Structural Coupling Index", network_metrics["structural_coupling_index"], "Observed cross-SDG edges divided by all possible cross-SDG pairs"),
        ("Source-SDG modularity", network_metrics["source_sdg_modularity"], "Weighted modularity of the fixed SDG14/SDG15 partition"),
        ("Source-SDG assortativity", network_metrics["source_sdg_assortativity"], "Attribute assortativity of connected paper pairs"),
        ("KMeans cosine silhouette", clustering_metrics["selected_cosine_silhouette"], "Maximum silhouette over k=2 through k=12"),
        ("Within-cluster mean similarity", clustering_metrics["within_cluster_mean_similarity"], "Mean cosine similarity for pairs assigned to the same KMeans cluster"),
        ("Between-cluster mean similarity", clustering_metrics["between_cluster_mean_similarity"], "Mean cosine similarity for pairs assigned to different KMeans clusters"),
        ("HDBSCAN noise share", clustering_metrics["hdbscan_noise_records"] / len(selection), "Records classified as noise with min_cluster_size=5 and min_samples=3"),
        ("Rule-based claim coverage", float(covered.mean()), "Documents containing at least one surface-pattern claim"),
        ("Mean claim-vector similarity", claim_similarity_mean, "Mean cosine similarity among non-empty document claim vectors"),
        ("Claim-verb normalized entropy", normalized_entropy(claims_by_verb), "Shannon entropy across eight canonical claim verbs"),
    ]
    structural = pd.DataFrame(metric_definitions, columns=["metric", "value", "definition"])
    save_csv(structural, tables / "table_4_structural_metrics.csv")

    kmi = theory[["paper_id"] + list(THEORY_ANCHORS)].copy()
    kmi["mean_theory_similarity"] = kmi[list(THEORY_ANCHORS)].mean(axis=1)
    kmi["KMI"] = 1 - kmi["mean_theory_similarity"]
    kmi = kmi.merge(selection[["paper_id", "source_sdg", "title"]], on="paper_id", how="left")
    save_csv(kmi, tables / "kmi_scores.csv")

    results = {
        "corpus_records": len(selection),
        "source_sdg_counts": selection["source_sdg"].value_counts().to_dict(),
        "selected_clusters": clustering_metrics["selected_k"],
        "cluster_silhouette": clustering_metrics["selected_cosine_silhouette"],
        "hdbscan_noise_records": clustering_metrics["hdbscan_noise_records"],
        "network_threshold": PRIMARY_SIMILARITY_THRESHOLD,
        "network_edges": network_metrics["edges"],
        "network_density": network_metrics["density"],
        "network_components": network_metrics["components"],
        "largest_component_percent": network_metrics["largest_component_percent"],
        "cross_sdg_edges": network_metrics["cross_sdg_edges"],
        "structural_coupling_index": network_metrics["structural_coupling_index"],
        "sdg_margin_mean": float(np.mean(margin)),
        "sdg_margin_cohen_d": float(np.mean(margin) / np.std(margin, ddof=1)),
        "sdg_margin_t_p_value": float(t_result.pvalue),
        "sdg_margin_bimodality": sarle_bimodality(margin),
        "technology_gini": gini(technology_counts),
        "technology_normalized_entropy": normalized_entropy(technology_counts),
        "theory_gini": gini(theory_counts),
        "theory_normalized_entropy": normalized_entropy(theory_counts),
        "theory_embedding_not_inferable_records": int((theory["embedding_inferred_theory"] == "Not_inferable").sum()),
        "theory_no_formal_theory_records": int((theory["final_theories"] == "No_formal_theory").sum()),
        "theory_human_adjudicated_records": int((theory["theory_assignment_source"] == "human_adjudication").sum()),
        "theory_stable_assignments": int((theory["theory_assignment_confidence"] == "stable").sum()),
        "theory_moderate_assignments": int((theory["theory_assignment_confidence"] == "moderate").sum()),
        "theory_ambiguous_assignments": int((theory["theory_assignment_confidence"] == "ambiguous").sum()),
        "theory_low_margin_assignments": int(theory["theory_low_margin_flag"].eq(True).sum()),
        "claims_extracted": len(claims),
        "claim_document_coverage": float(covered.mean()),
        "mean_claim_vector_similarity": claim_similarity_mean,
    }
    save_csv(pd.DataFrame([{"metric": key, "value": value} for key, value in results.items() if not isinstance(value, dict)]), tables / "results_summary.csv")
    return theory_distribution, structural, results


def draw_figures(
    selection: pd.DataFrame,
    theory: pd.DataFrame,
    sdg: pd.DataFrame,
    assignments: pd.DataFrame,
    technology_matrix: pd.DataFrame,
    technology_distribution: pd.DataFrame,
    technology_cooccurrence: pd.DataFrame,
    claims: pd.DataFrame,
    claim_cluster_percent: pd.DataFrame,
    paper_graph: nx.Graph,
    bridges: pd.DataFrame,
    knowledge_graph: nx.Graph,
    sensitivity: pd.DataFrame,
    tables: Path,
    figures: Path,
) -> None:
    categories = list(TECHNOLOGY_PATTERNS)
    labels = {"AI_ML": "AI/ML", "IoT": "IoT", "Blockchain": "Blockchain", "Remote_Sensing": "Remote sensing", "Digital_Twin": "Digital twin", "Big_Data": "Big data"}

    ordered = technology_distribution.sort_values("documents")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.barh([labels[value] for value in ordered["technology"]], ordered["documents"], color="#2A9D8F", zorder=3)
    for bar, count in zip(bars, ordered["documents"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, str(int(count)), va="center", color="#17324D", fontweight="bold")
    style_axes(ax, "Technology distribution across the retained corpus", "Document presence from explicit regular-expression rules")
    ax.set_xlabel("Documents")
    save_figure(fig, figures / "figure_03_technology_distribution.png")

    technology_sdg = technology_matrix.groupby("source_sdg")[categories].sum().T.reindex(columns=["SDG14", "SDG15"], fill_value=0)
    annotated_heatmap(technology_sdg.to_numpy(), [labels[value] for value in technology_sdg.index], ["SDG14", "SDG15"], "Technology by source SDG", figures / "figure_04_technology_sdg_heatmap.png", figsize=(7.5, 6.0))

    source_by_id = dict(zip(selection["paper_id"], selection["source_sdg"]))
    if paper_graph.number_of_edges():
        positions = nx.spring_layout(paper_graph, seed=SEED, weight="weight", iterations=300)
    else:
        positions = nx.circular_layout(paper_graph)
    fig, ax = plt.subplots(figsize=(10, 9))
    nx.draw_networkx_edges(paper_graph, positions, ax=ax, alpha=0.25, width=0.65, edge_color="#91A4B5")
    for source in ["SDG14", "SDG15"]:
        nodes = [node for node in paper_graph if source_by_id[node] == source]
        nx.draw_networkx_nodes(paper_graph, positions, nodelist=nodes, node_size=30, node_color=SDG_COLORS[source], alpha=0.85, linewidths=0.2, edgecolors="white", label=source, ax=ax)
    ax.set_title(f"Paper similarity network at cosine similarity ≥ {PRIMARY_SIMILARITY_THRESHOLD}", loc="left", color="#17324D", pad=14)
    ax.legend(frameon=False)
    ax.axis("off")
    save_figure(fig, figures / "figure_05_paper_similarity_network.png")

    top_bridges = bridges.loc[bridges["is_cross_sdg_bridge"]].head(15).sort_values("betweenness")
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    bar_colors = [SDG_COLORS[source] for source in top_bridges["source_sdg"]]
    bars = ax.barh(
        [f"{row.paper_id}  ({int(row.cross_sdg_degree)} cross-SDG links)" for row in top_bridges.itertuples()],
        top_bridges["betweenness"],
        color=bar_colors,
        zorder=3,
    )
    for bar, value in zip(bars, top_bridges["betweenness"]):
        ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8, color="#17324D")
    style_axes(ax, "Cross-SDG bridge papers", "Papers with cross-SDG edges, ranked by unweighted betweenness centrality")
    ax.set_xlabel("Betweenness centrality")
    ax.set_ylabel("")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=SDG_COLORS["SDG14"], label="SDG14"), Patch(color=SDG_COLORS["SDG15"], label="SDG15")], frameon=False, loc="lower right")
    save_figure(fig, figures / "figure_06_bridge_nodes.png")

    concept_nodes = sorted(
        [node for node, data in knowledge_graph.nodes(data=True) if data.get("node_type") != "paper"],
        key=lambda node: (knowledge_graph.nodes[node].get("node_type", ""), knowledge_graph.nodes[node].get("label", node)),
    )
    paper_nodes = sorted(
        [node for node, data in knowledge_graph.nodes(data=True) if data.get("node_type") == "paper"],
        key=lambda node: (knowledge_graph.nodes[node].get("source_sdg", ""), node),
    )
    kg_pos: dict[str, tuple[float, float]] = {}
    for index, node in enumerate(concept_nodes):
        angle = (
            2 * math.pi * index / max(1, len(concept_nodes))
            + math.pi / 2
            + math.pi / max(1, len(concept_nodes))
        )
        kg_pos[node] = (0.43 * math.cos(angle), 0.43 * math.sin(angle))
    for index, node in enumerate(paper_nodes):
        angle = 2 * math.pi * index / max(1, len(paper_nodes)) + math.pi / 2
        kg_pos[node] = (math.cos(angle), math.sin(angle))
    fig, ax = plt.subplots(figsize=(12, 10))
    nx.draw_networkx_edges(knowledge_graph, kg_pos, ax=ax, alpha=0.10, width=0.40, edge_color="#7F8C99")
    node_styles = {
        "paper": ("#C8D3DC", 18),
        "theory": ("#E76F51", 190),
        "sdg": ("#2A9D8F", 260),
        "technology": ("#E9C46A", 190),
    }
    for node_type, (color, size) in node_styles.items():
        nodes = [node for node, data in knowledge_graph.nodes(data=True) if data.get("node_type") == node_type]
        display = "SDG" if node_type == "sdg" else node_type.title()
        nx.draw_networkx_nodes(knowledge_graph, kg_pos, nodelist=nodes, node_color=color, node_size=size, alpha=0.88, edgecolors="white", linewidths=0.4, ax=ax, label=display)
    for node in concept_nodes:
        x, y = kg_pos[node]
        radius = math.hypot(x, y)
        label_x, label_y = (x * 1.16 / radius, y * 1.16 / radius)
        label = knowledge_graph.nodes[node].get("label", node).replace("_", " ")
        ax.text(label_x, label_y, label, fontsize=7, ha="left" if label_x >= 0 else "right", va="center", color="#17324D")
    ax.set_title("Document–theory–SDG–technology knowledge graph", loc="left", color="#17324D", pad=14)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.axis("off")
    save_figure(fig, figures / "figure_07_knowledge_graph.png")

    theory_memberships = explode_final_theories(theory).merge(
        selection[["paper_id", "source_sdg"]], on="paper_id", how="left", validate="many_to_one"
    )
    theory_sdg = pd.crosstab(theory_memberships["theory"], theory_memberships["source_sdg"]).reindex(columns=["SDG14", "SDG15"], fill_value=0)
    annotated_heatmap(theory_sdg.to_numpy(), [value.replace("_", " ") for value in theory_sdg.index], ["SDG14", "SDG15"], "Final theory classification by source SDG", figures / "figure_08_theory_sdg_heatmap.png", figsize=(8.0, 7.0))

    margin_frame = theory_memberships[["paper_id", "theory"]].merge(sdg[["paper_id", "sdg_margin"]], on="paper_id")
    theory_order = theory_memberships["theory"].value_counts().index.tolist()
    fig, ax = plt.subplots(figsize=(11, 6))
    values = [margin_frame.loc[margin_frame["theory"] == name, "sdg_margin"].to_numpy() for name in theory_order]
    box = ax.boxplot(values, tick_labels=[name.replace("_", " ") for name in theory_order], patch_artist=True, showfliers=True, medianprops={"color": "#17324D", "linewidth": 1.5})
    for patch, color in zip(box["boxes"], [COLORS[index % len(COLORS)] for index in range(len(box["boxes"]))]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.axhline(0, color="#D1495B", linestyle="--", linewidth=1)
    style_axes(ax, "SDG semantic margin by final theory classification", "Positive values indicate stronger similarity to the SDG14 anchor")
    ax.set_ylabel("SDG14 similarity minus SDG15 similarity")
    ax.tick_params(axis="x", rotation=42)
    save_figure(fig, figures / "figure_09_sdg_margin_by_theory.png")

    margin = sdg["sdg_margin"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.hist(margin, bins=20, color="#457B9D", alpha=0.78, edgecolor="white", density=True, zorder=3)
    if np.std(margin) > 0:
        grid = np.linspace(margin.min(), margin.max(), 300)
        ax.plot(grid, gaussian_kde(margin)(grid), color="#17324D", linewidth=2)
    ax.axvline(0, color="#D1495B", linestyle="--", linewidth=1.3, label="Equal anchor similarity")
    ax.axvline(np.mean(margin), color="#E9C46A", linewidth=2, label=f"Mean {np.mean(margin):.3f}")
    style_axes(ax, "SDG semantic-margin distribution", "Margin is calculated from fixed SDG14 and SDG15 anchor embeddings")
    ax.set_xlabel("SDG14 similarity minus SDG15 similarity")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    save_figure(fig, figures / "figure_10_sdg_margin_distribution.png")

    claim_values = claim_cluster_percent[list(CLAIM_VERB_PATTERNS)].to_numpy() if not claim_cluster_percent.empty else np.zeros((0, len(CLAIM_VERB_PATTERNS)))
    annotated_heatmap(claim_values, [f"Cluster {int(value)}" for value in claim_cluster_percent.index], [value.replace("_", " ") for value in claim_cluster_percent.columns], "Claim-pattern composition by cluster", figures / "figure_11_claim_patterns_by_cluster.png", fmt=".1f", colorbar_label="Percent of cluster claims", figsize=(12.5, 5.2))

    structures = claims.groupby(["subject_category", "canonical_verb", "outcome_category"]).size().reset_index(name="claims") if not claims.empty else pd.DataFrame(columns=["subject_category", "canonical_verb", "outcome_category", "claims"])
    top_structures = structures.nlargest(20, "claims").sort_values("claims")
    structure_labels = [
        f"{row.subject_category.replace('_', ' ')} → {row.canonical_verb.replace('_', ' ')} → {row.outcome_category.replace('_', ' ')}"
        for row in top_structures.itertuples()
    ]
    verb_colors = {verb: COLORS[index % len(COLORS)] for index, verb in enumerate(CLAIM_VERB_PATTERNS)}
    fig, ax = plt.subplots(figsize=(12.5, 8.5))
    bars = ax.barh(structure_labels, top_structures["claims"], color=[verb_colors[verb] for verb in top_structures["canonical_verb"]], zorder=3)
    for bar, value in zip(bars, top_structures["claims"]):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2, str(int(value)), va="center", fontsize=8, color="#17324D")
    style_axes(ax, "Most frequent rule-based claim structures", "Subject category → canonical verb → outcome category; evidence-cued abstract and conclusion sentences")
    ax.set_xlabel("Claims")
    ax.set_ylabel("")
    save_figure(fig, figures / "figure_12_claim_structures.png")

    sorted_papers = assignments.sort_values(["source_sdg", "cluster", "paper_id"])["paper_id"]
    technology_by_id = technology_matrix.set_index("paper_id").loc[sorted_papers, categories]
    fig, ax = plt.subplots(figsize=(9.5, 12))
    image = ax.imshow(technology_by_id.to_numpy(), aspect="auto", cmap=LinearSegmentedColormap.from_list("binary", ["#F4F6F8", "#174A6E"]), interpolation="nearest")
    ax.set_xticks(range(len(categories)), labels=[labels[value] for value in categories], rotation=35, ha="right")
    ax.set_yticks([])
    ax.set_ylabel(f"{len(technology_by_id)} papers sorted by source SDG and cluster")
    ax.set_title("Paper × technology matrix", loc="left", color="#17324D", pad=16)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(image, ax=ax, ticks=[0, 1], fraction=0.025, pad=0.02, label="Technology absent / present")
    save_figure(fig, figures / "figure_13_paper_technology_matrix.png")

    technology_graph = nx.Graph()
    for technology in categories:
        technology_graph.add_node(technology, documents=int(technology_matrix[technology].sum()))
    for a, b in combinations(categories, 2):
        weight = int(technology_cooccurrence.loc[a, b])
        if weight > 0:
            technology_graph.add_edge(a, b, weight=weight)
    tech_pos = nx.spring_layout(technology_graph, seed=SEED, weight="weight")
    fig, ax = plt.subplots(figsize=(9, 7))
    edge_weights = np.array([data["weight"] for _, _, data in technology_graph.edges(data=True)], dtype=float)
    widths = 0.5 + 7 * edge_weights / edge_weights.max() if len(edge_weights) else []
    nx.draw_networkx_edges(technology_graph, tech_pos, width=widths, alpha=0.42, edge_color="#6B7D8C", ax=ax)
    node_sizes = [250 + 18 * technology_graph.nodes[node]["documents"] for node in technology_graph]
    nx.draw_networkx_nodes(technology_graph, tech_pos, node_size=node_sizes, node_color="#E9C46A", edgecolors="white", linewidths=1, ax=ax)
    nx.draw_networkx_labels(technology_graph, tech_pos, labels={node: labels[node] for node in technology_graph}, font_size=9, ax=ax)
    edge_labels = {(u, v): data["weight"] for u, v, data in technology_graph.edges(data=True)}
    nx.draw_networkx_edge_labels(technology_graph, tech_pos, edge_labels=edge_labels, font_size=7, rotate=False, ax=ax)
    ax.set_title("Technology co-occurrence network", loc="left", color="#17324D", pad=14)
    ax.text(0, 1.01, "Node size shows document frequency; edge labels show document co-occurrence", transform=ax.transAxes, fontsize=9, color="#5B6770")
    ax.axis("off")
    save_figure(fig, figures / "figure_14_technology_cooccurrence_network.png")

    fig, ax1 = plt.subplots(figsize=(9.5, 5.5))
    ax1.plot(sensitivity["similarity_threshold"], sensitivity["edges"], marker="o", color="#174A6E", label="Edges")
    ax1.set_xlabel("Cosine-similarity threshold")
    ax1.set_ylabel("Edges", color="#174A6E")
    ax1.tick_params(axis="y", labelcolor="#174A6E")
    ax1.grid(axis="y", color="#E8EDF2")
    ax2 = ax1.twinx()
    ax2.plot(sensitivity["similarity_threshold"], sensitivity["largest_component_percent"], marker="s", color="#E76F51", label="Largest component")
    ax2.set_ylabel("Largest component (%)", color="#E76F51")
    ax2.tick_params(axis="y", labelcolor="#E76F51")
    ax1.axvline(PRIMARY_SIMILARITY_THRESHOLD, color="#2A9D8F", linestyle="--", linewidth=1.2)
    ax1.set_title("Similarity-network threshold sensitivity", loc="left", color="#17324D", pad=16)
    save_figure(fig, figures / "supplement_network_threshold_sensitivity.png")

    figure_manifest = pd.DataFrame(
        [
            {"figure": path.stem, "file": path.name, "method": "Generated deterministically by regenerate_all_sdg14_15_python.py"}
            for path in sorted(figures.glob("*.png"))
        ]
    )
    save_csv(figure_manifest, tables / "figure_manifest.csv")


def validate_outputs(output: Path, expected_ids: set[str]) -> dict[str, Any]:
    errors: list[str] = []
    tables = output / "tables"
    figures = output / "figures"
    required_tables = [
        "table_1_theory_distribution.csv",
        "table_2_sdg_distribution.csv",
        "table_3_sdg_margin_statistics.csv",
        "table_4_structural_metrics.csv",
        "table_technology_distribution.csv",
        "table_technology_by_sdg.csv",
        "table_theory_by_sdg.csv",
        "cluster_assignments.csv",
        "theory_similarity.csv",
        "theory_section_manifest.csv",
        "theory_section_chunk_manifest.csv",
        "theory_evidence_chunks.csv",
        "theory_explicit_mention_validation.csv",
        "theory_validation_summary.csv",
        "theory_manual_review_queue.csv",
        "manual_adjudications_applied.csv",
        "manual_theory_agreement_records.csv",
        "manual_theory_agreement_summary.csv",
        "technology_matrix.csv",
        "rule_based_claims.csv",
        "paper_similarity_edges.csv",
        "bridge_nodes.csv",
        "knowledge_graph_triples.csv",
        "kmi_scores.csv",
    ]
    required_figures = [f"figure_{number:02d}_{name}.png" for number, name in [
        (3, "technology_distribution"),
        (4, "technology_sdg_heatmap"),
        (5, "paper_similarity_network"),
        (6, "bridge_nodes"),
        (7, "knowledge_graph"),
        (8, "theory_sdg_heatmap"),
        (9, "sdg_margin_by_theory"),
        (10, "sdg_margin_distribution"),
        (11, "claim_patterns_by_cluster"),
        (12, "claim_structures"),
        (13, "paper_technology_matrix"),
        (14, "technology_cooccurrence_network"),
    ]]
    required_snapshot = [
        "input_snapshot/retained_selection.csv",
        "input_snapshot/theory_corpus_manual_adjudications.csv",
        "input_snapshot/embeddings/canonical_corpus.csv",
        "input_snapshot/embeddings/canonical_manifest.csv",
        "input_snapshot/embeddings/chunk_manifest.csv",
        "input_snapshot/embeddings/doc_embeddings.npy",
        "input_snapshot/embeddings/integrity_report.json",
        "input_snapshot/embeddings/run_metadata.json",
        "code/regenerate_all_sdg14_15_python.py",
        "code/build_canonical_corpus_embeddings.py",
        "derived_embeddings/theory_section_chunk_embeddings.npy",
    ]
    for name in required_tables:
        path = tables / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"Missing or empty table: {name}")
    for name in required_snapshot:
        path = output / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"Missing or empty reproducibility file: {name}")
    image_details = {}
    for name in required_figures:
        path = figures / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"Missing or empty figure: {name}")
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image_details[name] = {"width": image.width, "height": image.height}
                if image.width < 1400 or image.height < 1000:
                    errors.append(f"Figure resolution is too small: {name} {image.size}")
        except Exception as exc:
            errors.append(f"Unreadable figure {name}: {exc}")
    expected_count = len(expected_ids)
    for name in ["cluster_assignments.csv", "theory_similarity.csv", "theory_explicit_mention_validation.csv", "technology_matrix.csv", "kmi_scores.csv"]:
        frame = pd.read_csv(tables / name)
        if len(frame) != expected_count or set(frame["paper_id"]) != expected_ids or not frame["paper_id"].is_unique:
            errors.append(f"Paper reconciliation failed for {name}")
    theory = pd.read_csv(tables / "theory_similarity.csv")
    if not set(theory["dominant_theory"]).issubset(set(FINAL_FORMAL_THEORY_LABELS) | NON_FORMAL_THEORY_LABELS):
        errors.append("Theory assignments contain undeclared labels")
    final_labels = {
        label
        for value in theory["final_theories"]
        for label in parse_theory_labels(value)
    }
    if not final_labels.issubset(set(FINAL_FORMAL_THEORY_LABELS) | NON_FORMAL_THEORY_LABELS):
        errors.append("Multi-theory assignments contain undeclared labels")
    inferable = theory["embedding_inferred_theory"] != "Not_inferable"
    if not inferable.any():
        errors.append("No papers were inferable from eligible theory sections")
    if theory.loc[inferable, list(THEORY_ANCHORS)].isna().any().any():
        errors.append("Inferable papers contain missing theory scores")
    section_manifest = pd.read_csv(tables / "theory_section_manifest.csv")
    allowed_sections = {"abstract", "results", "discussion", "results_discussion"}
    if not set(section_manifest["section_type"]).issubset(allowed_sections):
        errors.append("Theory section manifest contains a forbidden section type")
    evidence = pd.read_csv(tables / "theory_evidence_chunks.csv")
    if not set(evidence["section_type"]).issubset(allowed_sections):
        errors.append("Theory evidence includes text outside eligible sections")
    section_vectors = np.load(output / "derived_embeddings" / "theory_section_chunk_embeddings.npy", allow_pickle=False)
    chunk_manifest = pd.read_csv(tables / "theory_section_chunk_manifest.csv")
    if section_vectors.shape != (len(chunk_manifest), 384) or not np.all(np.isfinite(section_vectors)):
        errors.append("Theory-section embeddings do not reconcile with their chunk manifest")
    claims = pd.read_csv(tables / "rule_based_claims.csv")
    if not claims.empty and not set(claims["zone"]).issubset({"abstract", "conclusion"}):
        errors.append("Claim extraction includes text outside Abstract and Conclusion")
    snapshot_selection = pd.read_csv(output / "input_snapshot" / "retained_selection.csv")
    if len(snapshot_selection) != expected_count or set(snapshot_selection["paper_id"]) != expected_ids:
        errors.append(f"Retained-selection input snapshot does not reconcile to {expected_count} paper IDs")
    snapshot_embeddings = np.load(output / "input_snapshot" / "embeddings" / "doc_embeddings.npy", allow_pickle=False)
    if snapshot_embeddings.shape != (expected_count, 384) or not np.all(np.isfinite(snapshot_embeddings)):
        errors.append(f"Embedding input snapshot is not a finite {expected_count} by 384 matrix")
    snapshot_chunks = pd.read_csv(
        output / "input_snapshot" / "embeddings" / "chunk_manifest.csv"
    )
    parameters = json.loads((output / "method_parameters.json").read_text(encoding="utf-8"))
    declared_chunks = int(parameters["embedding_parameters"]["total_chunks"])
    if declared_chunks != len(snapshot_chunks):
        errors.append(
            f"Declared document chunk count {declared_chunks} does not match manifest {len(snapshot_chunks)}"
        )
    manual = pd.read_csv(tables / "manual_adjudications_applied.csv")
    manual_eligible = manual["corpus_eligible"].astype(str).str.lower().eq("true")
    excluded_manual_ids = set(manual.loc[~manual_eligible, "paper_id"])
    if excluded_manual_ids & expected_ids:
        errors.append("A manually excluded paper remains in the final corpus")
    expected_manual_theory = (
        manual.loc[
            manual_eligible & manual["manual_theory"].fillna("").str.strip().ne(""),
            ["paper_id", "manual_theory"],
        ]
        .set_index("paper_id")["manual_theory"]
        .to_dict()
    )
    final_theory = theory.set_index("paper_id")["final_theories"].to_dict()
    for paper_id, expected_theories in expected_manual_theory.items():
        if parse_theory_labels(final_theory.get(paper_id, "")) != parse_theory_labels(expected_theories):
            errors.append(f"Manual theory adjudication was not applied for {paper_id}")
    pending = pd.read_csv(tables / "theory_manual_review_queue.csv")
    completed_ids = set(manual.loc[manual["adjudication_status"] == "complete", "paper_id"])
    if set(pending["paper_id"]) & completed_ids:
        errors.append("Completed manual adjudications remain in the pending review queue")
    return {
        "valid": not errors,
        "checked_at_utc": utc_now(),
        "errors": errors,
        "required_tables": len(required_tables),
        "required_figures": len(required_figures),
        "required_reproducibility_files": len(required_snapshot),
        "figure_dimensions": image_details,
    }


def write_documentation(
    output: Path,
    embedding_metadata: dict[str, Any],
    results: dict[str, Any],
) -> None:
    parameters = {
        "script_version": SCRIPT_VERSION,
        "completed_at_utc": utc_now(),
        "python_only": True,
        "llm_calls": 0,
        "external_api_calls": 0,
        "random_seed": SEED,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn_version,
        "networkx": nx.__version__,
        "sentence_transformer_model": MODEL,
        "sentence_transformer_revision": MODEL_REVISION,
        "embedding_parameters": embedding_metadata["embedding"],
        "theory_eligible_sections": ["Abstract", "Results", "Findings", "Discussion", "combined Results/Findings and Discussion"],
        "theory_excluded_sections": ["Introduction", "Background", "Literature review", "Theoretical background", "Methods", "Conclusion", "References"],
        "theory_missing_section_policy": "Not_inferable; no fallback to whole-document text",
        "theory_mapping": f"for each paper and each of {len(THEORY_ANCHORS)} name-specific theory anchors, maximum cosine similarity across eligible-section chunks; maximum pooling is used because formal theory names commonly occur in only one local passage",
        "theory_chunk_tokens_excluding_special_tokens": embedding_metadata["embedding"]["chunk_tokens_excluding_special_tokens"],
        "theory_chunk_overlap_tokens": embedding_metadata["embedding"]["chunk_overlap_tokens"],
        "theory_top_chunks_pooled": THEORY_TOP_CHUNKS,
        "theory_supporting_chunks_saved_per_anchor": THEORY_EVIDENCE_CHUNKS,
        "theory_secondary_robustness_score": "mean of the three highest eligible-section chunk similarities, reported as top3_mean_<theory> but not used for assignment",
        "theory_bootstrap_replicates": THEORY_BOOTSTRAP_REPLICATES,
        "theory_bootstrap_seed": "314 plus the original canonical embedding row, preserved after exclusions so retained papers do not change solely because earlier rows were removed",
        "theory_assignment_confidence": {
            "stable": f"bootstrap stability >= {THEORY_STABLE_BOOTSTRAP} and top-two margin >= {THEORY_LOW_MARGIN}",
            "moderate": f"bootstrap stability >= {THEORY_MODERATE_BOOTSTRAP} and top-two margin >= {THEORY_MODERATE_MARGIN}",
            "ambiguous": "all other inferable assignments",
        },
        "theory_low_margin_flag": f"top pooled score minus runner-up pooled score below {THEORY_LOW_MARGIN}",
        "theory_explicit_mentions": "exact theory-name patterns searched only within eligible sections and used for validation, never to set the embedding assignment",
        "manual_adjudications": embedding_metadata.get("manual_adjudication", {}),
        "final_theory_policy": "completed human adjudications override the embedding-inferred label; semicolon-separated human codes preserve multi-theory papers; study type and conceptual focus remain separate fields; No_formal_theory is excluded from formal-theory networks and diversity metrics",
        "sdg_mapping": "cosine similarity to two fixed SDG anchors; source folder remains the corpus domain label",
        "technology_extraction": "case-insensitive boundary-aware regular expressions over reference-trimmed full text",
        "claim_extraction": "strict Abstract/Conclusion surface-pattern rules using eight fixed canonical verb classes; no front-matter fallback is used, and every retained sentence must contain an explicit evidence cue",
        "clustering": "KMeans k=2..12 selected by maximum cosine silhouette; HDBSCAN sensitivity",
        "paper_network_threshold": PRIMARY_SIMILARITY_THRESHOLD,
        "paper_network_sensitivity_thresholds": [round(float(value), 3) for value in np.arange(0.50, 0.801, 0.025)],
        "structural_coupling_index": "cross-SDG edges divided by all possible SDG14-by-SDG15 paper pairs",
        "kmi": f"1 minus the unweighted mean of {len(THEORY_ANCHORS)} maximum-pooled theory similarities from eligible-section chunks",
    }
    save_json(parameters, output / "method_parameters.json")

    exclusions = pd.DataFrame(
        [
            {
                "excluded_output": "normative_priority_matrix and top_research_priorities",
                "reason": "The manuscript and surviving code do not define or validate a normative scoring rule.",
            },
            {
                "excluded_output": "supervised SDG/theory predictions and model explanations",
                "reason": "No independent target, train/test design, or validation protocol is defined; anchor assignments are already deterministic.",
            },
            {
                "excluded_output": "old term-level kg_real graph",
                "reason": "Its token/entity extraction rules are undocumented and cannot be reproduced from the declared method.",
            },
            {
                "excluded_output": "manual or analyst-written cluster labels",
                "reason": "The user required Python-only regeneration; cluster labels are generated from the three highest nonredundant TF-IDF terms.",
            },
        ]
    )
    save_csv(exclusions, output / "excluded_analyses.csv")

    readme = f"""# SDG 14–15 Python-only full regeneration

This bundle was generated from the {results['corpus_records']}-paper adjudicated corpus with deterministic Python code. It made zero LLM calls and zero external API calls.

## Corpus and embeddings

- Records: {results['corpus_records']} ({results['source_sdg_counts'].get('SDG14', 0)} SDG14 and {results['source_sdg_counts'].get('SDG15', 0)} SDG15)
- Embeddings: cached `{MODEL}` model at revision `{MODEL_REVISION}`
- Document vectors: chunked, length-weighted mean pooled, and L2 normalized

## Recomputed analyses

- KMeans and HDBSCAN clustering
- Automatic TF-IDF thematic terms and labels
- Chunk-level theory-anchor inference restricted to Abstract, Results/Findings, and Discussion
- Maximum chunk pooling for assignment, the three strongest supporting chunks per anchor, top-three-mean robustness scores, top-two margins, and 200-replicate bootstrap stability
- Explicit theory-name checks inside the same eligible sections, used only to validate embedding assignments
- A deterministic manual-review queue for non-stable assignments, section failures, and conflicts with explicit names
- Completed human adjudications applied after embedding inference, with study type and conceptual focus stored separately from formal theory
- Whole-document SDG anchor similarities
- Rule-based technology extraction and co-occurrence
- Similarity network, Louvain communities, bridge nodes, and threshold sensitivity
- Document–theory–SDG–technology knowledge graph
- Rule-based surface claim patterns and claim structures
- Multiplex layers, exploratory Adamic–Adar candidates, KMI, distribution metrics, and manuscript tables
- Figures 3 through 14 plus diagnostic figures

## Restrictions

The pipeline does not use an LLM for classification, labeling, extraction, interpretation, or writing results. `excluded_analyses.csv` identifies old outputs that lack reproducible definitions and therefore were not recreated.

Theory inference never reads Introduction, Background, Literature Review, theoretical-background, Methods, Conclusion, or References text. A paper without a detectable eligible section is labeled `Not_inferable`; the code does not fall back to its whole document. `theory_evidence_chunks.csv` records the exact eligible section and supporting text for every paper–theory score.

`embedding_inferred_theory` remains the deterministic embedding result. `final_theories` preserves all completed human theory codes; `dominant_theory` retains the first final label for backward compatibility. `No_formal_theory` is used when the user's coding identifies only a study type or conceptual focus. Digitalization and Industry 4.0 are therefore not treated as theories. `theory_explicit_mention_validation.csv` evaluates the embedding result and does not validate or overwrite the human decision.

The 0.65 paper-network threshold is retained for comparability with the manuscript. `network_threshold_sensitivity.csv` reports how network conclusions change between 0.50 and 0.80.

`input_snapshot/` contains the exact filtered selection, canonical text table, embedding matrix, embedding manifests, and manual-adjudication ledger used by this run. `code/` contains the Python analysis and embedding builders plus the pinned dependency file.

## Reproduction

From this bundle's root directory:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r code/requirements.txt
.venv/bin/python code/regenerate_all_sdg14_15_python.py \
  --base-analysis input_snapshot \
  --manual-adjudications input_snapshot/theory_corpus_manual_adjudications.csv \
  --output reproduced_analysis
```

The analysis downloads the pinned SentenceTransformer model revision when it is not already cached. Add `--offline` only when that exact revision is present in the local model cache.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")


def write_input_snapshot(
    base: Path,
    output: Path,
    selection: pd.DataFrame,
    corpus: pd.DataFrame,
    embeddings: np.ndarray,
    embedding_metadata: dict[str, Any],
    adjudications_path: Path,
) -> None:
    snapshot = output / "input_snapshot"
    snapshot_embeddings = snapshot / "embeddings"
    snapshot_embeddings.mkdir(parents=True)
    save_csv(selection, snapshot / "retained_selection.csv")
    shutil.copy2(adjudications_path, snapshot / adjudications_path.name)
    snapshot_corpus = corpus.copy()
    if "source_path" in snapshot_corpus:
        snapshot_corpus["source_path"] = snapshot_corpus["filename"].map(
            lambda name: f"papers/{name}"
        )
    save_csv(snapshot_corpus, snapshot_embeddings / "canonical_corpus.csv")

    ordered_ids = selection["paper_id"].tolist()
    row_by_id = {paper_id: row for row, paper_id in enumerate(ordered_ids)}
    canonical_manifest = pd.read_csv(base / "embeddings" / "canonical_manifest.csv")
    canonical_manifest = canonical_manifest.loc[
        canonical_manifest["paper_id"].isin(ordered_ids)
    ].copy()
    canonical_manifest["_order"] = canonical_manifest["paper_id"].map(row_by_id)
    canonical_manifest.sort_values("_order", inplace=True)
    canonical_manifest.drop(columns="_order", inplace=True)
    canonical_manifest["embedding_row"] = canonical_manifest["paper_id"].map(row_by_id).astype(int)
    if "source_path" in canonical_manifest:
        canonical_manifest["source_path"] = canonical_manifest["filename"].map(
            lambda name: f"papers/{name}"
        )
    save_csv(canonical_manifest, snapshot_embeddings / "canonical_manifest.csv")

    chunk_manifest = pd.read_csv(base / "embeddings" / "chunk_manifest.csv")
    chunk_manifest = chunk_manifest.loc[chunk_manifest["paper_id"].isin(ordered_ids)].copy()
    chunk_manifest["embedding_row"] = chunk_manifest["paper_id"].map(row_by_id).astype(int)
    chunk_manifest.sort_values(["embedding_row", "chunk_index"], inplace=True)
    save_csv(chunk_manifest, snapshot_embeddings / "chunk_manifest.csv")
    np.save(snapshot_embeddings / "doc_embeddings.npy", embeddings.astype(np.float32))

    excluded_ids = embedding_metadata.get("manual_adjudication", {}).get("excluded_paper_ids", [])
    integrity = json.loads((base / "embeddings" / "integrity_report.json").read_text(encoding="utf-8"))
    integrity.update(
        {
            "valid": True,
            "errors": [],
            "document_count": len(selection),
            "source_sdg_counts": selection["source_sdg"].value_counts().to_dict(),
            "manual_exclusions": excluded_ids,
        }
    )
    integrity["expected_index_check"] = {
        "performed": True,
        "path": "input_snapshot/retained_selection.csv",
        "rows": len(selection),
        "missing_from_index": [],
        "missing_from_source": [],
        "non_ok_statuses": [],
    }
    save_json(integrity, snapshot_embeddings / "integrity_report.json")

    metadata = json.loads(json.dumps(embedding_metadata))
    metadata["finished_at_utc"] = utc_now()
    metadata["papers_dir"] = "papers"
    metadata["output_dir"] = "input_snapshot/embeddings"
    metadata["include_ids_csv"] = "input_snapshot/retained_selection.csv"
    if metadata.get("expected_index"):
        metadata["expected_index"] = "input_snapshot/retained_selection.csv"
    metadata["include_ids_csv_sha256"] = sha256_file(snapshot / "retained_selection.csv")
    metadata["corpus"].update(
        {
            "document_count": len(selection),
            "source_sdg_counts": selection["source_sdg"].value_counts().to_dict(),
            "total_full_characters": int(canonical_manifest["full_char_count"].sum()),
            "total_full_words": int(canonical_manifest["full_word_count"].sum()),
            "documents_with_back_matter_removed": int(canonical_manifest["back_matter_removed"].astype(bool).sum()),
        }
    )
    metadata["embedding"].update(
        {
            "embedding_shape": list(embeddings.shape),
            "embedding_dtype": "float32",
            "document_norm_min": float(np.linalg.norm(embeddings, axis=1).min()),
            "document_norm_max": float(np.linalg.norm(embeddings, axis=1).max()),
            "total_chunks": len(chunk_manifest),
        }
    )
    metadata["output_sha256"] = {
        name: sha256_file(snapshot_embeddings / name)
        for name in [
            "canonical_manifest.csv", "canonical_corpus.csv", "integrity_report.json",
            "chunk_manifest.csv", "doc_embeddings.npy",
        ]
    }
    save_json(metadata, snapshot_embeddings / "run_metadata.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-analysis", type=Path, default=BASE_ANALYSIS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manual-adjudications", type=Path, default=MANUAL_ADJUDICATIONS)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require the pinned SentenceTransformer revision to be present in the local cache.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    base = args.base_analysis.resolve()
    output = args.output.resolve()
    adjudications_path = args.manual_adjudications.resolve()
    adjudications = load_manual_adjudications(adjudications_path)
    if output.exists():
        if not args.force:
            raise FileExistsError(f"Output directory already exists: {output}; pass --force to replace it")
        shutil.rmtree(output)
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True)
    figures.mkdir(parents=True)
    set_plot_style()

    selection, corpus, embeddings, embedding_metadata = load_inputs(
        base, adjudications, adjudications_path
    )
    theory, sdg = map_conceptual_anchors(
        selection, corpus, embeddings, embedding_metadata, adjudications, tables, args.offline
    )
    compute_manual_theory_agreement(theory, tables)
    assignments, cluster_summary, clustering_metrics = cluster_documents(selection, corpus, embeddings, tables, figures)
    technology_matrix, technology_distribution, technology_cooccurrence = extract_technologies(selection, corpus, tables)
    claims, claim_matrix, claim_cluster_percent = extract_claims(selection, corpus, assignments, tables)
    paper_graph, bridges, sensitivity, network_metrics = build_networks(selection, embeddings, theory, technology_matrix, tables)
    knowledge_graph, triples = build_knowledge_graph(selection, theory, technology_matrix, tables)
    theory_distribution, structural_metrics, results = statistical_tables(
        selection,
        theory,
        sdg,
        technology_distribution,
        claims,
        claim_matrix,
        clustering_metrics,
        network_metrics,
        tables,
    )
    draw_figures(
        selection,
        theory,
        sdg,
        assignments,
        technology_matrix,
        technology_distribution,
        technology_cooccurrence,
        claims,
        claim_cluster_percent,
        paper_graph,
        bridges,
        knowledge_graph,
        sensitivity,
        tables,
        figures,
    )
    write_documentation(output, embedding_metadata, results)

    write_input_snapshot(
        base, output, selection, corpus, embeddings, embedding_metadata, adjudications_path
    )
    code_dir = output / "code"
    code_dir.mkdir()
    shutil.copy2(Path(__file__).resolve(), code_dir / Path(__file__).name)
    embedding_builder = PROJECT_ROOT / "build_canonical_corpus_embeddings.py"
    if embedding_builder.is_file():
        shutil.copy2(embedding_builder, code_dir / embedding_builder.name)
    requirements = PROJECT_ROOT / "requirements.txt"
    if requirements.is_file():
        shutil.copy2(requirements, code_dir / requirements.name)

    manifest_rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"output_manifest.csv", "validation_report.json"}:
            manifest_rows.append(
                {
                    "file": str(path.relative_to(output)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    save_csv(pd.DataFrame(manifest_rows), output / "output_manifest.csv")
    validation = validate_outputs(output, set(selection["paper_id"]))
    validation["manifest_files_hashed"] = len(manifest_rows)
    save_json(validation, output / "validation_report.json")
    if not validation["valid"]:
        raise RuntimeError("Output validation failed: " + "; ".join(validation["errors"]))

    print(json.dumps({"output": str(output), "valid": True, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*assortativity.*")
    raise SystemExit(main())
