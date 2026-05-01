"""
src/utils.py
------------
Shared utilities used by ALL modules — CLI and Streamlit alike.

Provides:
  - Text preprocessing  (clean_text, tokenize, remove_stopwords)
  - Bundled sample data (get_corpus, get_sentiment_sentences, …)
  - CLI display helpers (print_section, print_divider)
  - Streamlit UI helpers (page_header, info_expander, sentiment_badge, …)

No print statements in business logic — helpers are kept separate so the
backend functions can be called from any context without side effects.
"""

from __future__ import annotations

import re
import string
from typing import List, Optional

# ── Stop-word list ─────────────────────────────────────────────────────────────

STOPWORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "and", "or", "but", "not", "this",
    "that", "it", "its", "i", "we", "you", "he", "she", "they", "my",
    "your", "our", "their", "as", "if", "so", "than", "then", "about",
    "which", "who", "what", "when", "where", "how", "all", "any", "each",
}


# ── Text preprocessing ─────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Lowercase, remove punctuation, and collapse whitespace.

    Args:
        text: Raw input string.

    Returns:
        Cleaned string.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """
    Simple whitespace tokenizer after cleaning.

    Args:
        text: Input string.

    Returns:
        List of tokens.
    """
    return clean_text(text).split()


def remove_stopwords(
    tokens: List[str],
    stopwords: Optional[set] = None,
) -> List[str]:
    """
    Remove stopwords from a token list.

    Args:
        tokens:    List of word tokens.
        stopwords: Custom stop-word set. Falls back to the built-in STOPWORDS.

    Returns:
        Filtered token list.
    """
    sw = stopwords if stopwords is not None else STOPWORDS
    return [t for t in tokens if t not in sw and len(t) > 1]


# ── Bundled sample data ────────────────────────────────────────────────────────

SAMPLE_CORPUS: List[str] = [
    "Machine learning is a subset of artificial intelligence.",
    "Natural language processing enables computers to understand human language.",
    "Deep learning models require large amounts of training data.",
    "Neural networks are inspired by the structure of the human brain.",
    "Text classification is a fundamental task in NLP.",
    "Sentiment analysis helps determine the emotional tone of text.",
    "Named entity recognition identifies proper nouns in text.",
    "Word embeddings capture semantic relationships between words.",
    "Transfer learning has revolutionized the field of NLP.",
    "Transformers use self-attention mechanisms to process sequences.",
    "BERT is a bidirectional transformer pre-trained on large corpora.",
    "GPT models generate coherent and contextually relevant text.",
    "Data preprocessing is a critical step in any machine learning pipeline.",
    "Tokenization splits text into individual words or subwords.",
    "Stop words are common words that are often removed during preprocessing.",
]

TFIDF_SAMPLES: List[str] = [
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    "Natural language processing allows computers to understand and generate human language effectively.",
    "Deep learning models require large amounts of labelled training data and significant compute.",
    "Transformer architectures have revolutionised the way we approach sequence modelling tasks.",
    "Data preprocessing and feature engineering are critical steps in any machine learning pipeline.",
]

SENTIMENT_SENTENCES: List[str] = [
    "I absolutely love this product — it exceeded all my expectations!",
    "This was a terrible experience. I will never buy from them again.",
    "The item arrived on time. It works as described.",
    "Honestly, it's okay. Not great, not terrible.",
    "Outstanding customer service! They resolved my issue immediately.",
    "The quality is disappointingly poor for the price I paid.",
    "I'm very happy with this purchase. Highly recommended!",
    "Worst decision I ever made. Complete waste of money.",
]

NER_TEXTS: List[str] = [
    "Apple Inc. was founded by Steve Jobs and Steve Wozniak in Cupertino, California in 1976.",
    "Elon Musk, the CEO of Tesla and SpaceX, visited Berlin, Germany last Tuesday.",
    "The United Nations headquarters is located in New York City, United States.",
    "Amazon acquired Whole Foods Market for $13.7 billion in 2017.",
    "Dr. Sarah Johnson from MIT published a paper on quantum computing in Nature journal.",
]

# Rich corpus for more meaningful Word2Vec embeddings
WORD2VEC_RAW: List[str] = [
    "the king ruled the kingdom with great wisdom and power",
    "the queen was beloved by all the people in the land",
    "the prince studied hard to become a great ruler one day",
    "the princess loved to read books in the royal library",
    "the king and queen governed the kingdom together peacefully",
    "machine learning algorithms learn patterns from large datasets",
    "deep learning is a powerful technique in artificial intelligence",
    "neural networks are used to solve complex classification problems",
    "natural language processing helps machines understand human speech",
    "convolutional networks excel at image recognition and vision tasks",
    "transformer models revolutionised natural language processing",
    "the doctor examined the patient carefully in the hospital ward",
    "nurses and doctors work together to provide excellent patient care",
    "the scientist discovered a new molecule in the chemistry laboratory",
    "researchers published findings in a peer reviewed science journal",
    "the teacher explained the lesson clearly to all the students",
    "students studied hard to pass their final university examinations",
    "the dog chased the cat around the house and garden",
    "cats and dogs are popular domestic animals kept as pets",
    "the economy grew steadily despite global market uncertainty",
    "stock markets react quickly to changes in government economic policy",
    "python is a popular programming language used for data science",
    "javascript is widely used for web development and front end design",
]

WORD2VEC_QUERY_EXAMPLES: List[str] = [
    "king", "machine", "learning", "doctor", "neural", "student",
]

PCA_WORDS: List[str] = [
    "king", "queen", "prince", "princess", "kingdom",
    "machine", "learning", "neural", "language", "processing",
    "doctor", "nurse", "patient", "hospital", "scientist",
    "teacher", "student", "lesson", "university",
    "dog", "cat", "animal",
]


def get_corpus() -> List[str]:
    """Return the general NLP sample corpus."""
    return SAMPLE_CORPUS


def get_tfidf_samples() -> List[str]:
    """Return extended sample texts for TF-IDF demos."""
    return TFIDF_SAMPLES


def get_sentiment_sentences() -> List[str]:
    """Return sentences for sentiment analysis comparison."""
    return SENTIMENT_SENTENCES


def get_ner_texts() -> List[str]:
    """Return texts for named entity recognition."""
    return NER_TEXTS


def get_word2vec_corpus() -> List[List[str]]:
    """Return tokenized sentences for Word2Vec training."""
    return [sentence.split() for sentence in WORD2VEC_RAW]


# ── CLI display helpers ────────────────────────────────────────────────────────

def print_section(title: str) -> None:
    """Print a formatted section header to stdout (CLI only)."""
    width = 62
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def print_divider(char: str = "·", width: int = 62) -> None:
    """Print a lightweight divider to stdout (CLI only)."""
    print(char * width)


# ── Streamlit UI helpers ───────────────────────────────────────────────────────

# Colour map for sentiment labels: label → (icon, text-color, background-color)
SENTIMENT_COLOURS: dict[str, tuple[str, str, str]] = {
    "POSITIVE": ("🟢", "#28a745", "#d4edda"),
    "NEGATIVE": ("🔴", "#dc3545", "#f8d7da"),
    "NEUTRAL":  ("🟡", "#856404", "#fff3cd"),
    "ERROR":    ("⚠️",  "#721c24", "#f8d7da"),
}


def sentiment_badge(label: str) -> str:
    """Return an HTML badge string for a sentiment label."""
    icon, color, bg = SENTIMENT_COLOURS.get(label, ("❓", "#333", "#eee"))
    return (
        f'<span style="background:{bg};color:{color};padding:3px 10px;'
        f'border-radius:12px;font-weight:600;font-size:0.85em;">'
        f"{icon} {label}</span>"
    )


def confidence_bar(score: float, color: str = "#4A90D9") -> str:
    """Render a mini HTML progress bar for a 0–1 confidence score."""
    pct = int(score * 100)
    return (
        f'<div style="background:#e9ecef;border-radius:4px;height:10px;width:100%;">'
        f'<div style="background:{color};width:{pct}%;height:10px;border-radius:4px;"></div>'
        f"</div><small style=\"color:#666;\">{pct}%</small>"
    )


def page_header(title: str, subtitle: str, icon: str = "") -> None:
    """
    Render a consistent section header in Streamlit.

    Args:
        title:    Section title text.
        subtitle: Brief description shown below the title.
        icon:     Optional emoji prefix.
    """
    try:
        import streamlit as st
        st.markdown(f"## {icon} {title}")
        st.markdown(
            f'<p style="color:#6c757d;font-size:1rem;margin-top:-0.5rem;">{subtitle}</p>',
            unsafe_allow_html=True,
        )
        st.divider()
    except ImportError:
        pass  # No-op outside Streamlit context


def info_expander(title: str, content: str) -> None:
    """
    Render a collapsible info box in Streamlit.

    Args:
        title:   Expander label (without the ℹ️ icon).
        content: Markdown body.
    """
    try:
        import streamlit as st
        with st.expander(f"ℹ️ {title}"):
            st.markdown(content)
    except ImportError:
        pass
