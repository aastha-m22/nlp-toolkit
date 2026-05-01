"""
src/sentiment.py
----------------
Three sentiment-analysis approaches evaluated side-by-side:

    1. VADER      — rule-based, lexicon-driven (nltk)
    2. TextBlob   — pattern-based, lightweight
    3. DistilBERT — fine-tuned transformer (HuggingFace)

All analysers return a uniform dict so the caller doesn't have to
special-case each backend.

Heavy models (HuggingFace pipeline) are lazy-loaded and cached at
module level to avoid re-downloading on every call.

Public API
----------
run_sentiment(text, use_transformer=True) → dict  (CLI & Streamlit)
analyse_vader(text)                        → dict
analyse_textblob(text)                     → dict
analyse_transformer(text)                  → dict
run_all(text, use_transformer)             → pd.DataFrame
batch_compare(texts, use_transformer)      → pd.DataFrame
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List

import pandas as pd

warnings.filterwarnings("ignore")

_hf_pipeline = None  # Module-level HuggingFace pipeline cache


# ═══════════════════════════════════════════════════════════════════════════════
#  VADER
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_vader(text: str) -> Dict[str, Any]:
    """
    Classify sentiment with NLTK's VADER analyser.

    Best for short, social-media-style text with punctuation and caps.

    Args:
        text: Input sentence.

    Returns:
        {
            label:    "POSITIVE" | "NEGATIVE" | "NEUTRAL",
            score:    float  (abs compound, 0–1),
            pos:      float,
            neg:      float,
            neu:      float,
            compound: float  (-1 to +1),
        }
    """
    try:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer

        nltk.download("vader_lexicon", quiet=True)
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(text)
        compound = scores["compound"]

        if compound >= 0.05:
            label = "POSITIVE"
        elif compound <= -0.05:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"

        return {
            "label":    label,
            "score":    round(abs(compound), 4),
            "pos":      round(scores["pos"], 4),
            "neg":      round(scores["neg"], 4),
            "neu":      round(scores["neu"], 4),
            "compound": round(compound, 4),
        }
    except Exception as exc:
        return {"label": "ERROR", "score": 0.0, "error": str(exc)}


# ═══════════════════════════════════════════════════════════════════════════════
#  TextBlob
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_textblob(text: str) -> Dict[str, Any]:
    """
    Classify sentiment with TextBlob's pattern-based analyser.

    Polarity:     -1.0 (very negative) → +1.0 (very positive)
    Subjectivity:  0.0 (objective)     →  1.0 (subjective)

    Args:
        text: Input sentence.

    Returns:
        {
            label:        "POSITIVE" | "NEGATIVE" | "NEUTRAL",
            score:        float  (abs polarity, 0–1),
            polarity:     float,
            subjectivity: float,
        }
    """
    try:
        from textblob import TextBlob

        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        if polarity > 0.05:
            label = "POSITIVE"
        elif polarity < -0.05:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"

        return {
            "label":        label,
            "score":        round(abs(polarity), 4),
            "polarity":     round(polarity, 4),
            "subjectivity": round(subjectivity, 4),
        }
    except Exception as exc:
        return {"label": "ERROR", "score": 0.0, "error": str(exc)}


# ═══════════════════════════════════════════════════════════════════════════════
#  HuggingFace Transformer (DistilBERT)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_hf_pipeline():
    """Lazy-load and cache the HuggingFace sentiment pipeline."""
    global _hf_pipeline
    if _hf_pipeline is None:
        from transformers import pipeline
        _hf_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512,
        )
    return _hf_pipeline


def analyse_transformer(text: str) -> Dict[str, Any]:
    """
    Classify sentiment with a fine-tuned DistilBERT transformer.

    Model: distilbert-base-uncased-finetuned-sst-2-english
    (~260 MB download on first use; cached locally afterwards.)

    Args:
        text: Input sentence.

    Returns:
        {
            label: "POSITIVE" | "NEGATIVE",
            score: float  (model confidence, 0–1),
        }
    """
    try:
        pipe = _load_hf_pipeline()
        result = pipe(text)[0]
        return {
            "label": result["label"],
            "score": round(result["score"], 4),
        }
    except Exception as exc:
        return {"label": "ERROR", "score": 0.0, "error": str(exc)}


# ═══════════════════════════════════════════════════════════════════════════════
#  Comparison helpers
# ═══════════════════════════════════════════════════════════════════════════════

def run_all(text: str, use_transformer: bool = True) -> pd.DataFrame:
    """
    Run all analysers on a single text and return a comparison DataFrame.

    Columns: Analyser | Label | Confidence | Note

    Args:
        text:            Input sentence.
        use_transformer: Whether to include the DistilBERT analyser.

    Returns:
        DataFrame with one row per analyser.
    """
    vader = analyse_vader(text)
    blob  = analyse_textblob(text)

    rows = [
        {
            "Analyser":   "VADER",
            "Label":      vader.get("label", "—"),
            "Confidence": vader.get("score", 0.0),
            "Note":       f"compound={vader.get('compound', '—')}",
        },
        {
            "Analyser":   "TextBlob",
            "Label":      blob.get("label", "—"),
            "Confidence": blob.get("score", 0.0),
            "Note":       f"subjectivity={blob.get('subjectivity', '—')}",
        },
    ]

    if use_transformer:
        trans = analyse_transformer(text)
        rows.append({
            "Analyser":   "DistilBERT",
            "Label":      trans.get("label", "—"),
            "Confidence": trans.get("score", 0.0),
            "Note":       "Fine-tuned on SST-2",
        })

    return pd.DataFrame(rows)


def batch_compare(
    texts: List[str],
    use_transformer: bool = False,
) -> pd.DataFrame:
    """
    Compare all analysers across multiple sentences.

    Columns: Text | VADER | TextBlob [| DistilBERT]

    Args:
        texts:           List of input sentences.
        use_transformer: Whether to include DistilBERT (slower).

    Returns:
        DataFrame with one row per sentence and one column per analyser.
    """
    rows = []
    for text in texts:
        vader = analyse_vader(text)
        blob  = analyse_textblob(text)
        row = {
            "Text":     text[:55] + "…" if len(text) > 55 else text,
            "VADER":    vader.get("label", "—"),
            "TextBlob": blob.get("label", "—"),
        }
        if use_transformer:
            trans = analyse_transformer(text)
            row["DistilBERT"] = trans.get("label", "—")
        rows.append(row)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  Master entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run_sentiment(text: str, use_transformer: bool = False) -> dict:
    """
    Master entry point — runs all analysers and returns a structured dict.

    This is the function called by both main.py (CLI) and app.py (Streamlit).

    Args:
        text:            Input sentence.
        use_transformer: Include DistilBERT (requires ~260 MB download first run).

    Returns:
        {
            "text":          str,
            "vader":         dict,
            "textblob":      dict,
            "transformer":   dict | None,
            "comparison_df": pd.DataFrame,
            "use_transformer": bool,
        }
    """
    vader = analyse_vader(text)
    blob  = analyse_textblob(text)
    trans = analyse_transformer(text) if use_transformer else None

    comparison_df = run_all(text, use_transformer=use_transformer)

    return {
        "text":            text,
        "vader":           vader,
        "textblob":        blob,
        "transformer":     trans,
        "comparison_df":   comparison_df,
        "use_transformer": use_transformer,
    }
