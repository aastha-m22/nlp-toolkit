#!/usr/bin/env python3
"""
main.py
-------
NLP Toolkit — Command-Line Interface
=====================================
Calls all four backend modules in sequence (or a single module via --module)
and prints structured, human-readable output to the terminal.

Usage
-----
    python main.py                              # run all modules
    python main.py --module tfidf              # run TF-IDF only
    python main.py --module word2vec --word king
    python main.py --module ner --text "Apple was founded by Steve Jobs."
    python main.py --module sentiment --text "I love this!"
    python main.py --skip-transformer          # skip HuggingFace (offline/low-RAM)
"""

from __future__ import annotations

import argparse
import sys
import time

from src.utils import (
    get_tfidf_samples,
    get_sentiment_sentences,
    get_ner_texts,
    print_section,
    print_divider,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Argument parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NLP Toolkit — run all or individual modules.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--module",
        choices=["tfidf", "word2vec", "ner", "sentiment"],
        help="Run a specific module only.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Custom input text for NER or Sentiment (overrides built-in samples).",
    )
    parser.add_argument(
        "--word",
        type=str,
        default="king",
        help="Query word for Word2Vec similarity (default: king).",
    )
    parser.add_argument(
        "--skip-transformer",
        action="store_true",
        help="Skip the HuggingFace transformer in sentiment analysis.",
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-module display logic
# ═══════════════════════════════════════════════════════════════════════════════

def display_tfidf() -> None:
    """Run TF-IDF on sample corpus and print results."""
    from src.tfidf import run_tfidf

    texts = get_tfidf_samples()
    result = run_tfidf(texts, top_n=6)

    print_section("TF-IDF  |  From-Scratch vs scikit-learn")

    print("\n[Scratch Implementation]\n")
    scratch_df = result["scratch"]
    for doc in scratch_df["Document"].unique():
        doc_df = scratch_df[scratch_df["Document"] == doc]
        first_text = texts[int(doc.split()[-1]) - 1]
        print(f'  {doc}: "{first_text[:65]}…"')
        for _, row in doc_df.iterrows():
            print(f"    {row['Term']:<25}  TF={row['TF']:.4f}  IDF={row['IDF']:.4f}  TF-IDF={row['TF-IDF']:.4f}")
        print_divider()

    print("\n[scikit-learn Implementation]\n")
    sklearn_df = result["sklearn"]
    for doc in sklearn_df["Document"].unique():
        doc_df = sklearn_df[sklearn_df["Document"] == doc]
        first_text = texts[int(doc.split()[-1]) - 1]
        print(f'  {doc}: "{first_text[:65]}…"')
        for _, row in doc_df.iterrows():
            print(f"    {row['Term']:<25}  {row['TF-IDF (sklearn)']:.4f}")
        print_divider()

    print(
        "\n[Note] Scratch uses raw TF; sklearn applies sublinear scaling log(1+tf).\n"
        "       Minor differences are expected — trends are consistent."
    )


def display_word2vec(word: str) -> None:
    """Train Word2Vec, query similar words, and print results."""
    from src.word2vec import run_word2vec

    print_section("Word2Vec  |  gensim skip-gram")
    print(f"\n  Training model on built-in corpus …")

    result = run_word2vec(word, topn=6)

    print(f"\n  Vocab size:   {result['vocab_size']} tokens")
    print(f"  Vector dims:  {result['vector_size']}")
    print()

    if not result["in_vocab"]:
        print(f'  ⚠  "{word}" is out-of-vocabulary.')
        print(f"     Try one of: {', '.join(result['query_examples'])}")
    else:
        print(f'[Similar words for "{word}"]\n')
        for _, row in result["similar_words"].iterrows():
            bar = "█" * int(row["Cosine Similarity"] * 30)
            print(f"    {row['Word']:<18}  {row['Cosine Similarity']:.4f}  {bar}")
        print_divider()

    # Show a few quick comparisons for context
    from src.word2vec import train_model, get_similar_words
    model = train_model()
    print("\n[Quick demos]\n")
    for demo_word in ["king", "machine", "doctor"]:
        if demo_word == word:
            continue
        sim_df = get_similar_words(model, demo_word, topn=3)
        if not sim_df.empty:
            pairs = "  →  ".join(f"{r['Word']} ({r['Cosine Similarity']:.2f})"
                                  for _, r in sim_df.iterrows())
            print(f"  {demo_word:<12}  {pairs}")
    print()


def display_ner(custom_text: str | None = None) -> None:
    """Run NER on sample texts (or a custom text) and print results."""
    from src.ner import run_ner

    print_section("Named Entity Recognition  |  spaCy en_core_web_sm")

    texts = [custom_text] if custom_text else get_ner_texts()

    for i, text in enumerate(texts, 1):
        print(f"\n  [{i}] {text}\n")
        result = run_ner(text)

        if not result["model_loaded"]:
            print(f"  ✗  spaCy unavailable: {result['error']}")
            print("     Install: python -m spacy download en_core_web_sm")
            return

        if result["entity_count"] == 0:
            print("     No entities detected.")
        else:
            # Group by label for clean CLI output
            grouped: dict[str, list[str]] = {}
            for _, row in result["entities_df"].iterrows():
                grouped.setdefault(row["Label"], []).append(row["Entity"])
            for label, items in grouped.items():
                entity_list = ",  ".join(f'"{e}"' for e in items)
                desc = result["entities_df"].loc[
                    result["entities_df"]["Label"] == label, "Type"
                ].iloc[0]
                print(f"    {label:<14}  ({desc})")
                print(f"                   {entity_list}")
        print_divider()


def display_sentiment(custom_text: str | None = None, use_transformer: bool = True) -> None:
    """Run all sentiment analysers and print a comparison table."""
    from src.sentiment import run_sentiment

    print_section("Sentiment Analysis  |  VADER · TextBlob · DistilBERT")

    texts = [custom_text] if custom_text else get_sentiment_sentences()[:4]
    if not use_transformer:
        print("  [info] HuggingFace transformer skipped (--skip-transformer)\n")

    for text in texts:
        print(f'\n  "{text}"\n')
        result = run_sentiment(text, use_transformer=use_transformer)

        v = result["vader"]
        b = result["textblob"]

        def label_str(lbl: str) -> str:
            icons = {"POSITIVE": "✅", "NEGATIVE": "❌", "NEUTRAL": "➖", "ERROR": "⚠"}
            return f"{icons.get(lbl, '?')} {lbl}"

        print(f"    VADER     {label_str(v['label']):<20}  confidence={v['score']:.2%}"
              f"  compound={v.get('compound', '—')}")
        print(f"    TextBlob  {label_str(b['label']):<20}  confidence={b['score']:.2%}"
              f"  subjectivity={b.get('subjectivity', '—')}")

        if use_transformer and result["transformer"]:
            t = result["transformer"]
            if t.get("label") != "ERROR":
                print(f"    DistilBERT {label_str(t['label']):<19}  confidence={t['score']:.2%}")
            else:
                print(f"    DistilBERT ⚠ {t.get('error', 'Error')}")

        print_divider()


# ═══════════════════════════════════════════════════════════════════════════════
#  Banner & runner
# ═══════════════════════════════════════════════════════════════════════════════

def banner() -> None:
    print(
        "\n"
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║          NLP Toolkit — Core Techniques Implemented          ║\n"
        "║     TF-IDF  ·  Word2Vec  ·  NER  ·  Sentiment Analysis     ║\n"
        "╚══════════════════════════════════════════════════════════════╝"
    )


def _run(name: str, fn, *args, **kwargs) -> None:
    """Execute a module function with timing and graceful error handling."""
    t0 = time.perf_counter()
    try:
        fn(*args, **kwargs)
    except KeyboardInterrupt:
        print(f"\n  [interrupted] {name} stopped.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n  [error] {name}: {type(exc).__name__}: {exc}")
    elapsed = time.perf_counter() - t0
    print(f"\n  ⏱  {name} completed in {elapsed:.2f}s")


def main() -> None:
    args = parse_args()
    banner()

    modules = {
        "tfidf":     (display_tfidf,     [],                             {}),
        "word2vec":  (display_word2vec,  [args.word],                    {}),
        "ner":       (display_ner,       [args.text],                    {}),
        "sentiment": (display_sentiment, [args.text],
                      {"use_transformer": not args.skip_transformer}),
    }

    if args.module:
        fn, a, kw = modules[args.module]
        _run(args.module, fn, *a, **kw)
    else:
        for name, (fn, a, kw) in modules.items():
            _run(name, fn, *a, **kw)

    print("\n" + "═" * 64)
    print("  All modules complete. Run `streamlit run app.py` for the web UI.")
    print("═" * 64 + "\n")


if __name__ == "__main__":
    main()
