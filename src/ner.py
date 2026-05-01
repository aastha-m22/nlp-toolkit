"""
src/ner.py
----------
Named Entity Recognition via spaCy's en_core_web_sm model.

Returns structured DataFrames and colour-highlighted HTML suitable for
both terminal output (CLI) and st.markdown() / st.dataframe() (Streamlit).

The spaCy model is cached at module level to avoid reloading on every call.

Public API
----------
run_ner(text)         → dict  (used by CLI and Streamlit)
load_model()          → (nlp, error_string | None)
extract_entities(text) → (pd.DataFrame, html_string)
get_entity_summary(df) → pd.DataFrame
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import pandas as pd

warnings.filterwarnings("ignore")


# ── Entity metadata ────────────────────────────────────────────────────────────

LABEL_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "PERSON":      ("👤", "People, including fictional"),
    "ORG":         ("🏢", "Companies, agencies, institutions"),
    "GPE":         ("🌍", "Countries, cities, states"),
    "LOC":         ("📍", "Non-GPE locations (mountains, rivers)"),
    "DATE":        ("📅", "Absolute or relative dates"),
    "TIME":        ("⏰", "Times smaller than a day"),
    "MONEY":       ("💰", "Monetary values"),
    "PRODUCT":     ("📦", "Objects, vehicles, foods"),
    "EVENT":       ("🎉", "Named events (wars, sports, etc.)"),
    "WORK_OF_ART": ("🎨", "Titles of books, songs, etc."),
    "FAC":         ("🏗️",  "Buildings, airports, highways"),
    "NORP":        ("🏛️",  "Nationalities, religious groups"),
    "LAW":         ("⚖️",  "Named laws and documents"),
    "LANGUAGE":    ("🗣️",  "Named languages"),
    "PERCENT":     ("📊", "Percentage values"),
    "QUANTITY":    ("📐", "Measurements, weights"),
    "ORDINAL":     ("🔢", "Ordinal numbers (first, second)"),
    "CARDINAL":    ("🔣", "Numerals not covered elsewhere"),
}

# Background / foreground colour pairs for HTML highlighting
LABEL_COLOURS: dict[str, tuple[str, str]] = {
    "PERSON":      ("#D4E6F1", "#1A5276"),
    "ORG":         ("#D5F5E3", "#1E8449"),
    "GPE":         ("#FDEBD0", "#A04000"),
    "LOC":         ("#FAD7A0", "#7D6608"),
    "DATE":        ("#E8DAEF", "#6C3483"),
    "TIME":        ("#D1F2EB", "#148F77"),
    "MONEY":       ("#FDEDEC", "#922B21"),
    "PRODUCT":     ("#EBF5FB", "#1A5276"),
    "EVENT":       ("#FEF9E7", "#9A7D0A"),
    "WORK_OF_ART": ("#F9EBEA", "#7B241C"),
    "NORP":        ("#E8F8F5", "#0E6655"),
    "LAW":         ("#F4F6F7", "#424949"),
    "LANGUAGE":    ("#EAF2FF", "#1F4E79"),
    "PERCENT":     ("#FFF3E0", "#8D4E00"),
    "QUANTITY":    ("#F3E5F5", "#6A1B9A"),
    "ORDINAL":     ("#E3F2FD", "#0D47A1"),
    "CARDINAL":    ("#E8EAF6", "#283593"),
}

_nlp = None  # Module-level spaCy model cache


# ═══════════════════════════════════════════════════════════════════════════════
#  Model loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(model_name: str = "en_core_web_sm") -> Tuple[object, Optional[str]]:
    """
    Load (and cache) the spaCy language model. Downloads automatically if missing.

    Args:
        model_name: spaCy model identifier.

    Returns:
        (nlp_model, error_string) — error_string is None on success.
    """
    global _nlp

    if _nlp is not None:
        return _nlp, None

    try:
        import spacy
        _nlp = spacy.load(model_name)
        return _nlp, None
    except OSError:
        # Auto-download the model
        try:
            import subprocess
            import sys
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", model_name],
                check=True,
                capture_output=True,
            )
            import spacy
            _nlp = spacy.load(model_name)
            return _nlp, None
        except Exception as exc:
            return None, str(exc)
    except Exception as exc:
        return None, str(exc)


# ═══════════════════════════════════════════════════════════════════════════════
#  Entity extraction
# ═══════════════════════════════════════════════════════════════════════════════

def _escape(s: str) -> str:
    """HTML-escape special characters."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _build_highlighted_html(text: str, doc) -> str:
    """
    Produce an inline colour-highlighted HTML string for the input text,
    with entity labels shown as superscripts.
    """
    output: List[str] = []
    last = 0
    for ent in doc.ents:
        output.append(_escape(text[last:ent.start_char]))
        bg, fg = LABEL_COLOURS.get(ent.label_, ("#F0F0F0", "#333"))
        emoji, _ = LABEL_DESCRIPTIONS.get(ent.label_, ("🔖", ""))
        output.append(
            f'<mark style="background:{bg};color:{fg};padding:2px 6px;'
            f'border-radius:4px;margin:0 2px;font-weight:500;">'
            f"{_escape(ent.text)}"
            f'<sup style="font-size:0.65em;margin-left:3px;opacity:0.8;">'
            f"{emoji}{ent.label_}</sup></mark>"
        )
        last = ent.end_char
    output.append(_escape(text[last:]))
    body = "".join(output)
    return (
        f'<p style="font-size:1.05rem;line-height:1.9;'
        f'font-family:Georgia,serif;color:#1a1a1a;">{body}</p>'
    )


def extract_entities(text: str) -> Tuple[pd.DataFrame, str]:
    """
    Run NER on a text string.

    Args:
        text: Input text.

    Returns:
        Tuple of:
          - entities_df: DataFrame with columns
                         [Entity, Label, Type, Start, End]
          - html_string: Colour-highlighted HTML for st.markdown()
    """
    nlp, error = load_model()
    if nlp is None:
        return (
            pd.DataFrame(),
            f"<p style='color:red'>spaCy error: {error}</p>",
        )

    doc = nlp(text)
    rows = []
    for ent in doc.ents:
        emoji, desc = LABEL_DESCRIPTIONS.get(ent.label_, ("🔖", ent.label_))
        rows.append({
            "Entity": ent.text,
            "Label":  ent.label_,
            "Type":   f"{emoji} {desc}",
            "Start":  ent.start_char,
            "End":    ent.end_char,
        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["Entity", "Label", "Type", "Start", "End"]
    )
    html = _build_highlighted_html(text, doc)
    return df, html


def get_entity_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group extracted entities by label to produce a summary table.

    Columns: Label | Type | Entities Found

    Args:
        df: Output from extract_entities().

    Returns:
        Summary DataFrame, or empty DataFrame if input is empty.
    """
    if df.empty:
        return pd.DataFrame()

    return (
        df.groupby("Label")["Entity"]
          .apply(lambda x: ", ".join(x.unique()))
          .reset_index()
          .rename(columns={"Entity": "Entities Found"})
          .merge(
              pd.DataFrame([
                  {"Label": k, "Type": f"{v[0]} {v[1]}"}
                  for k, v in LABEL_DESCRIPTIONS.items()
              ]),
              on="Label",
              how="left",
          )
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Master entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run_ner(text: str) -> dict:
    """
    Master entry point — runs NER and returns a structured result dict.

    This is the function called by both main.py (CLI) and app.py (Streamlit).

    Args:
        text: Input string.

    Returns:
        {
            "text":            str,
            "entities_df":     pd.DataFrame,    # raw entity rows
            "summary_df":      pd.DataFrame,    # grouped by label
            "highlighted_html": str,            # for st.markdown()
            "entity_count":    int,
            "model_loaded":    bool,
            "error":           str | None,
        }
    """
    nlp, error = load_model()
    if nlp is None:
        return {
            "text":              text,
            "entities_df":       pd.DataFrame(),
            "summary_df":        pd.DataFrame(),
            "highlighted_html":  f"<p style='color:red'>spaCy error: {error}</p>",
            "entity_count":      0,
            "model_loaded":      False,
            "error":             error,
        }

    entities_df, html = extract_entities(text)
    summary_df = get_entity_summary(entities_df)

    return {
        "text":              text,
        "entities_df":       entities_df,
        "summary_df":        summary_df,
        "highlighted_html":  html,
        "entity_count":      len(entities_df),
        "model_loaded":      True,
        "error":             None,
    }
