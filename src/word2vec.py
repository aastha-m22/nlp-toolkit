"""
src/word2vec.py
---------------
Word2Vec training (gensim skip-gram) with similarity queries and
optional PCA-based 2-D visualisation.

All heavy objects (the trained model) are cached at module level so
subsequent calls within the same process reuse the same weights.
In Streamlit, use @st.cache_resource on top of train_model().

Public API
----------
run_word2vec(word)                  → dict  (used by CLI and Streamlit)
train_model(sentences, …)          → gensim Word2Vec
get_similar_words(model, word, …)  → pd.DataFrame
get_vocab(model)                   → List[str]
pca_plot_data(model, words)        → pd.DataFrame
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.utils import get_word2vec_corpus, PCA_WORDS, WORD2VEC_QUERY_EXAMPLES

# Semantic cluster labels used for PCA colouring
_CLUSTERS: dict[str, set[str]] = {
    "royalty":   {"king", "queen", "prince", "princess", "kingdom", "ruler"},
    "ml/ai":     {"machine", "learning", "deep", "neural", "networks",
                  "transformer", "language", "processing", "convolutional",
                  "classification"},
    "science":   {"scientist", "researcher", "doctor", "nurse", "laboratory",
                  "molecule", "chemistry", "findings", "journal", "patient"},
    "education": {"teacher", "student", "students", "lesson", "university",
                  "examinations", "library", "books"},
}

_cached_model = None  # Module-level cache (avoids retraining on every call)


# ═══════════════════════════════════════════════════════════════════════════════
#  Model training
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(
    sentences: Optional[List[List[str]]] = None,
    vector_size: int = 64,
    window: int = 4,
    min_count: int = 1,
    epochs: int = 200,
    seed: int = 42,
):
    """
    Train a Word2Vec (skip-gram) model on the given tokenized corpus.

    Results are cached at module level — subsequent calls with the same
    arguments return instantly. Pass sentences=None to use the built-in corpus.

    Args:
        sentences:   List of tokenized sentences (list of word lists).
        vector_size: Dimensionality of word embeddings.
        window:      Context window size.
        min_count:   Minimum word frequency threshold.
        epochs:      Training iterations.
        seed:        Random seed for reproducibility.

    Returns:
        Trained gensim Word2Vec model.
    """
    global _cached_model

    if sentences is None:
        sentences = get_word2vec_corpus()

    if _cached_model is not None:
        return _cached_model

    from gensim.models import Word2Vec

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=1,      # deterministic with fixed seed
        sg=1,           # skip-gram
        epochs=epochs,
        seed=seed,
    )
    _cached_model = model
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  Query utilities
# ═══════════════════════════════════════════════════════════════════════════════

def get_similar_words(model, word: str, topn: int = 8) -> pd.DataFrame:
    """
    Return a DataFrame of words most similar to the query word.

    Columns: Word | Cosine Similarity

    Args:
        model: Trained Word2Vec model.
        word:  Query word (case-insensitive).
        topn:  Number of similar words to return.

    Returns:
        DataFrame, or empty DataFrame if the word is out-of-vocabulary.
    """
    word = word.strip().lower()
    if word not in model.wv:
        return pd.DataFrame(columns=["Word", "Cosine Similarity"])
    results = model.wv.most_similar(word, topn=topn)
    df = pd.DataFrame(results, columns=["Word", "Cosine Similarity"])
    df["Cosine Similarity"] = df["Cosine Similarity"].round(4)
    return df


def get_vocab(model) -> List[str]:
    """Return the sorted vocabulary list of a trained model."""
    return sorted(model.wv.index_to_key)


def pca_plot_data(
    model,
    words: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Project word vectors to 2-D via PCA for scatter-plot visualisation.

    Columns: word | x | y | cluster

    Args:
        model: Trained Word2Vec model.
        words: Words to include. Defaults to PCA_WORDS from utils.

    Returns:
        DataFrame with PCA coordinates and semantic cluster labels.
    """
    from sklearn.decomposition import PCA

    if words is None:
        words = PCA_WORDS

    valid = [w for w in words if w in model.wv]
    if len(valid) < 3:
        return pd.DataFrame(columns=["word", "x", "y", "cluster"])

    vectors = np.array([model.wv[w] for w in valid])
    coords = PCA(n_components=2, random_state=42).fit_transform(vectors)

    def _cluster(w: str) -> str:
        for name, members in _CLUSTERS.items():
            if w in members:
                return name
        return "other"

    return pd.DataFrame({
        "word":    valid,
        "x":       coords[:, 0],
        "y":       coords[:, 1],
        "cluster": [_cluster(w) for w in valid],
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  Master entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run_word2vec(word: str, topn: int = 8) -> dict:
    """
    Master entry point — trains (or reuses) the model and returns a
    structured result dict.

    This is the function called by both main.py (CLI) and app.py (Streamlit).

    Args:
        word:  Query word to find similar terms for.
        topn:  Number of similar words to retrieve.

    Returns:
        {
            "word":          str,
            "in_vocab":      bool,
            "similar_words": pd.DataFrame,   # columns: Word, Cosine Similarity
            "vocab_size":    int,
            "vector_size":   int,
            "pca_data":      pd.DataFrame,   # for visualisation
            "query_examples": List[str],
        }
    """
    model = train_model()
    word = word.strip().lower()

    similar_df = get_similar_words(model, word, topn=topn)
    pca_df = pca_plot_data(model)

    return {
        "word":           word,
        "in_vocab":       word in model.wv,
        "similar_words":  similar_df,
        "vocab_size":     len(model.wv),
        "vector_size":    model.vector_size,
        "pca_data":       pca_df,
        "query_examples": WORD2VEC_QUERY_EXAMPLES,
    }
