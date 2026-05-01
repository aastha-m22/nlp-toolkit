"""
src/__init__.py
---------------
NLP Toolkit — shared backend package.

Exposes the four master entry points so callers can do:

    from src.tfidf     import run_tfidf
    from src.word2vec  import run_word2vec
    from src.ner       import run_ner
    from src.sentiment import run_sentiment
"""

from src.tfidf     import run_tfidf
from src.word2vec  import run_word2vec
from src.ner       import run_ner
from src.sentiment import run_sentiment

__all__ = ["run_tfidf", "run_word2vec", "run_ner", "run_sentiment"]
