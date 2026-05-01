"""
src/tfidf.py
------------
TF-IDF implementation in two flavours:

  1. From-scratch — pure Python, formula-transparent, educational.
  2. scikit-learn — production-grade, sparse matrix, sublinear scaling.

All public functions return structured Python objects (dicts / DataFrames).
No print statements — presentation is handled by the caller (CLI or Streamlit).

Public API
----------
run_tfidf(texts)            → dict  (used by CLI and Streamlit)
scratch_tfidf(texts, top_n) → pd.DataFrame
sklearn_tfidf(texts, top_n) → pd.DataFrame
compare_top_terms(texts)    → (scratch_df, sklearn_df)
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils import tokenize, remove_stopwords


# ═══════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _preprocess(text: str) -> List[str]:
    """Tokenize and remove stopwords for the scratch implementation."""
    return remove_stopwords(tokenize(text))


def _compute_tf(tokens: List[str]) -> Dict[str, float]:
    """Term frequency: count(t) / total_terms."""
    if not tokens:
        return {}
    tf: Dict[str, float] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    total = len(tokens)
    return {t: count / total for t, count in tf.items()}


def _compute_idf(corpus: List[List[str]]) -> Dict[str, float]:
    """
    Smoothed IDF (sklearn-style):
        IDF(t) = log((1 + N) / (1 + df(t))) + 1
    """
    N = len(corpus)
    df: Dict[str, int] = {}
    for doc in corpus:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1
    return {term: math.log((1 + N) / (1 + freq)) + 1 for term, freq in df.items()}


# ═══════════════════════════════════════════════════════════════════════════════
#  Public functions
# ═══════════════════════════════════════════════════════════════════════════════

def scratch_tfidf(texts: List[str], top_n: int = 10) -> pd.DataFrame:
    """
    Compute TF-IDF from scratch and return a tidy DataFrame.

    Columns: Document | Term | TF | IDF | TF-IDF

    Args:
        texts: List of raw text strings.
        top_n: Maximum number of top-scoring terms to keep per document.

    Returns:
        DataFrame sorted by TF-IDF score descending, top_n rows per document.
    """
    corpus = [_preprocess(t) for t in texts]
    idf = _compute_idf(corpus)
    rows = []
    for doc_idx, tokens in enumerate(corpus):
        tf = _compute_tf(tokens)
        for term, tf_score in tf.items():
            tfidf_score = tf_score * idf.get(term, 0.0)
            rows.append({
                "Document": f"Doc {doc_idx + 1}",
                "Term":     term,
                "TF":       round(tf_score, 5),
                "IDF":      round(idf.get(term, 0.0), 5),
                "TF-IDF":   round(tfidf_score, 5),
            })

    if not rows:
        return pd.DataFrame(columns=["Document", "Term", "TF", "IDF", "TF-IDF"])

    df = pd.DataFrame(rows)
    # Keep top_n terms per document by TF-IDF score
    df = (
        df.sort_values("TF-IDF", ascending=False)
          .groupby("Document")
          .head(top_n)
          .reset_index(drop=True)
    )
    return df


def sklearn_tfidf(texts: List[str], top_n: int = 10) -> pd.DataFrame:
    """
    Compute TF-IDF with scikit-learn's TfidfVectorizer.

    Columns: Document | Term | TF-IDF (sklearn)

    Uses sublinear TF scaling: log(1 + tf) instead of raw tf.

    Args:
        texts: List of raw text strings.
        top_n: Maximum number of top-scoring terms to keep per document.

    Returns:
        DataFrame sorted by score descending, top_n rows per document.
    """
    if not texts or all(t.strip() == "" for t in texts):
        return pd.DataFrame(columns=["Document", "Term", "TF-IDF (sklearn)"])

    vectorizer = TfidfVectorizer(
        stop_words="english",
        sublinear_tf=True,          # log(1 + tf) — better for long docs
        ngram_range=(1, 1),
    )
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return pd.DataFrame(columns=["Document", "Term", "TF-IDF (sklearn)"])

    feature_names = vectorizer.get_feature_names_out()
    dense = matrix.toarray()

    rows = []
    for doc_idx, row in enumerate(dense):
        top_indices = np.argsort(row)[::-1][:top_n]
        for idx in top_indices:
            if row[idx] > 0:
                rows.append({
                    "Document":        f"Doc {doc_idx + 1}",
                    "Term":            feature_names[idx],
                    "TF-IDF (sklearn)": round(float(row[idx]), 5),
                })

    return pd.DataFrame(rows).reset_index(drop=True)


def compare_top_terms(
    texts: List[str],
    top_n: int = 8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper — returns (scratch_df, sklearn_df).

    Args:
        texts:  List of raw text strings.
        top_n:  Top-N terms per document.

    Returns:
        Tuple of (scratch DataFrame, sklearn DataFrame).
    """
    return scratch_tfidf(texts, top_n), sklearn_tfidf(texts, top_n)


def run_tfidf(texts: List[str], top_n: int = 8) -> dict:
    """
    Master entry point — runs both implementations and returns a structured dict.

    This is the function called by both main.py (CLI) and app.py (Streamlit).

    Args:
        texts:  List of text documents. Accepts a single string too.
        top_n:  Top-N terms to extract per document.

    Returns:
        {
            "scratch": pd.DataFrame,   # scratch TF-IDF results
            "sklearn": pd.DataFrame,   # sklearn TF-IDF results
            "doc_count": int,
            "top_n": int,
        }
    """
    if isinstance(texts, str):
        texts = [texts]

    scratch_df, sklearn_df = compare_top_terms(texts, top_n=top_n)
    return {
        "scratch":   scratch_df,
        "sklearn":   sklearn_df,
        "doc_count": len(texts),
        "top_n":     top_n,
    }
