"""
app.py
------
NLP Toolkit — Interactive Streamlit Application
================================================
All business logic lives in src/ — this file is presentation only.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is on the path when run from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Page config (must be the first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="NLP Toolkit",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0f1117; }
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.95rem !important;
    padding: 4px 0 !important;
}
[data-testid="metric-container"] {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 10px;
    padding: 12px 16px;
}
[data-testid="stDataFrame"] { border-radius: 8px; }
.stCodeBlock { border-radius: 8px; }
.stTabs [role="tab"] { font-weight: 500; }
.stTabs [aria-selected="true"] { color: #4A90D9 !important; }
.section-title {
    font-size: 1.4rem; font-weight: 700; color: #1a1a2e;
    border-left: 4px solid #4A90D9;
    padding-left: 12px; margin: 1rem 0 0.5rem;
}
.result-box {
    background: #f0f4ff; border: 1px solid #c9d7f8;
    border-radius: 10px; padding: 16px 20px; margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Shared backend imports ────────────────────────────────────────────────────
from src.tfidf     import run_tfidf
from src.word2vec  import run_word2vec, train_model, get_similar_words, pca_plot_data
from src.ner       import run_ner
from src.sentiment import run_sentiment, batch_compare
from src.utils     import (
    get_tfidf_samples, get_sentiment_sentences, get_ner_texts,
    WORD2VEC_QUERY_EXAMPLES, SENTIMENT_COLOURS,
    sentiment_badge, page_header, info_expander,
)

# ── Cached heavy objects ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training Word2Vec model…")
def _cached_w2v_model():
    return train_model()


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 NLP Toolkit")
    st.markdown(
        '<p style="font-size:0.8rem;color:#aaa;margin-top:-10px;">'
        "Core techniques · Interactive demos</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    page = st.radio(
        "Navigate",
        options=[
            "🏠 Home",
            "📊 TF-IDF",
            "🔤 Word2Vec",
            "🏷️ Named Entity Recognition",
            "💬 Sentiment Analysis",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.75rem;color:#888;text-align:center;">'
        "spaCy · gensim · HuggingFace<br>sklearn · Streamlit · Python</p>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  HOME PAGE
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown(
        '<h1 style="font-size:2.8rem;font-weight:800;color:#1a1a2e;margin-bottom:0;">'
        "🧠 NLP Toolkit</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size:1.15rem;color:#555;margin-top:4px;">'
        "Four core NLP techniques — from-scratch implementations vs production "
        "libraries, unified under one interactive interface.</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    cards = [
        (col1, "📊", "TF-IDF",     "Scratch + sklearn comparison", "#EBF5FB"),
        (col2, "🔤", "Word2Vec",   "gensim embeddings + PCA",       "#EAF7EA"),
        (col3, "🏷️",  "NER",        "spaCy named entities",          "#FEF9E7"),
        (col4, "💬", "Sentiment",  "VADER · TextBlob · DistilBERT", "#F9EBEA"),
    ]
    for col, icon, title, desc, bg in cards:
        with col:
            st.markdown(
                f'<div style="background:{bg};border-radius:12px;padding:20px;'
                f'text-align:center;border:1px solid rgba(0,0,0,0.06);">'
                f'<div style="font-size:2.2rem;">{icon}</div>'
                f'<div style="font-weight:700;font-size:1.05rem;margin:8px 0 4px;">{title}</div>'
                f'<div style="color:#666;font-size:0.85rem;">{desc}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🛠 Tech Stack")
    tech_cols = st.columns(5)
    techs = ["Python 3.10+", "scikit-learn", "gensim", "spaCy", "HuggingFace 🤗"]
    for col, tech in zip(tech_cols, techs):
        with col:
            st.code(tech, language=None)

    st.markdown("### 🚀 How to Run")
    run_col1, run_col2 = st.columns(2)
    with run_col1:
        st.markdown("**CLI**")
        st.code("python main.py\npython main.py --module tfidf\npython main.py --skip-transformer", language="bash")
    with run_col2:
        st.markdown("**Web UI**")
        st.code("streamlit run app.py", language="bash")


# ═══════════════════════════════════════════════════════════════════════════════
#  TF-IDF PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 TF-IDF":
    page_header("TF-IDF", "Term Frequency–Inverse Document Frequency", "📊")

    info_expander("What is TF-IDF?", """
**TF-IDF** scores how important a word is to a document *relative to a corpus*.

| Component | Formula | Intuition |
|-----------|---------|-----------|
| TF  | count(t, d) / total_words(d) | Frequency in this document |
| IDF | log((1+N)/(1+df(t))) + 1    | Rarity across all documents |
| TF-IDF | TF × IDF | High = common here, rare elsewhere |

**From-scratch** uses raw TF. **sklearn** applies `log(1+tf)` (sublinear scaling) for better handling of long documents.
""")

    samples = get_tfidf_samples()

    st.markdown("#### 📝 Enter Your Documents")
    st.caption("One document per text area below, or use the built-in samples.")

    use_custom = st.toggle("Use custom documents", value=False)
    if use_custom:
        raw = st.text_area(
            "Enter documents (one per line)",
            value="\n".join(samples[:3]),
            height=140,
        )
        texts = [line.strip() for line in raw.splitlines() if line.strip()]
    else:
        texts = samples
        with st.expander("📚 Built-in corpus"):
            for i, t in enumerate(texts, 1):
                st.markdown(f"**Doc {i}:** {t}")

    top_n = st.slider("Top-N terms per document", 3, 15, 8)

    if st.button("🔍 Compute TF-IDF", type="primary", use_container_width=True):
        if not texts:
            st.warning("Please enter at least one document.")
        else:
            with st.spinner("Computing…"):
                result = run_tfidf(texts, top_n=top_n)

            tab1, tab2 = st.tabs(["🧮 From-Scratch", "⚙️ scikit-learn"])

            with tab1:
                st.markdown("#### From-Scratch Results")
                st.caption("Pure Python — educational, transparent formula.")
                df = result["scratch"]
                if df.empty:
                    st.info("No terms found. Try longer documents.")
                else:
                    st.dataframe(
                        df.style.background_gradient(subset=["TF-IDF"], cmap="Blues"),
                        use_container_width=True,
                        hide_index=True,
                    )
                    # Bar chart of top terms across all docs
                    fig, ax = plt.subplots(figsize=(8, 3.5))
                    top10 = df.nlargest(10, "TF-IDF")
                    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top10)))
                    ax.barh(top10["Term"], top10["TF-IDF"], color=colors)
                    ax.set_xlabel("TF-IDF Score")
                    ax.set_title("Top 10 Terms (Scratch) — All Documents", pad=8)
                    ax.invert_yaxis()
                    ax.spines[["top", "right"]].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

            with tab2:
                st.markdown("#### scikit-learn Results")
                st.caption("Sublinear TF scaling — production-grade.")
                df_sk = result["sklearn"]
                if df_sk.empty:
                    st.info("No terms found.")
                else:
                    st.dataframe(
                        df_sk.style.background_gradient(subset=["TF-IDF (sklearn)"], cmap="Greens"),
                        use_container_width=True,
                        hide_index=True,
                    )

            st.info(
                "💡 Minor differences between implementations are expected: sklearn uses "
                "`log(1+tf)` instead of raw TF, and has its own English stop-word list."
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  WORD2VEC PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔤 Word2Vec":
    page_header("Word2Vec", "Word embeddings via gensim skip-gram", "🔤")

    info_expander("What is Word2Vec?", """
**Word2Vec** trains dense vector representations of words so that *semantically
similar words have similar vectors*.

- **Skip-gram** (used here): predict context words from a target word.
- **Cosine similarity** measures how close two vectors are (1 = identical direction).
- **PCA** projects 50-D vectors to 2-D for visualisation.

Words in similar contexts (king/queen, doctor/nurse) cluster together in vector space.
""")

    model = _cached_w2v_model()

    st.markdown("#### 🔍 Similarity Query")
    col_in, col_btn = st.columns([4, 1])
    with col_in:
        word_input = st.text_input(
            "Enter a word",
            value="king",
            placeholder="king, machine, doctor…",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("Search 🔎", type="primary", use_container_width=True)

    st.caption(f"Quick examples: {' · '.join(WORD2VEC_QUERY_EXAMPLES)}")

    topn = st.slider("Number of similar words", 3, 15, 8)

    if run_btn or word_input:
        result = run_word2vec(word_input.strip().lower(), topn=topn)

        if not result["in_vocab"]:
            st.warning(
                f'"{word_input}" is not in the vocabulary. '
                f"Try: {', '.join(WORD2VEC_QUERY_EXAMPLES)}"
            )
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Vocab Size", f"{result['vocab_size']:,}")
            c2.metric("Vector Dims", result["vector_size"])
            c3.metric("Similar Words Found", len(result["similar_words"]))

            st.markdown(f"#### 🔗 Words Most Similar to **{result['word']}**")
            sim_df = result["similar_words"]

            # Styled table + horizontal bar chart side by side
            tbl_col, chart_col = st.columns([1, 1])
            with tbl_col:
                st.dataframe(
                    sim_df.style.background_gradient(
                        subset=["Cosine Similarity"], cmap="YlGn"
                    ).format({"Cosine Similarity": "{:.4f}"}),
                    use_container_width=True,
                    hide_index=True,
                )
            with chart_col:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                colors = plt.cm.YlGn(np.linspace(0.4, 0.9, len(sim_df)))
                ax.barh(sim_df["Word"], sim_df["Cosine Similarity"],
                        color=colors[::-1])
                ax.set_xlabel("Cosine Similarity")
                ax.set_xlim(0, 1)
                ax.set_title(f'Similar to "{result["word"]}"', pad=8)
                ax.invert_yaxis()
                ax.spines[["top", "right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    # PCA Visualisation
    st.markdown("---")
    st.markdown("#### 📈 PCA Embedding Space (2-D)")
    st.caption("Word vectors projected from high-dimensional space — similar words cluster together.")

    pca_df = pca_plot_data(model)
    if not pca_df.empty:
        cluster_colors = {
            "royalty":   "#E74C3C",
            "ml/ai":     "#3498DB",
            "science":   "#2ECC71",
            "education": "#F39C12",
            "other":     "#95A5A6",
        }
        fig, ax = plt.subplots(figsize=(9, 6))
        for cluster, group_df in pca_df.groupby("cluster"):
            ax.scatter(
                group_df["x"], group_df["y"],
                label=cluster,
                color=cluster_colors.get(cluster, "#95A5A6"),
                s=80, alpha=0.85, zorder=3,
            )
            for _, row in group_df.iterrows():
                ax.annotate(
                    row["word"], (row["x"], row["y"]),
                    fontsize=8, ha="right", va="bottom",
                    color="#333", zorder=4,
                )
        ax.axhline(0, color="#ddd", linewidth=0.6)
        ax.axvline(0, color="#ddd", linewidth=0.6)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
        ax.set_title("Word2Vec Embeddings — PCA (2D)", pad=10)
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  NER PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏷️ Named Entity Recognition":
    page_header("Named Entity Recognition", "Entity extraction via spaCy en_core_web_sm", "🏷️")

    info_expander("What is NER?", """
**Named Entity Recognition** identifies and classifies *named entities* in text:

| Label | Meaning |
|-------|---------|
| PERSON | People (real or fictional) |
| ORG | Companies, agencies, institutions |
| GPE | Countries, cities, states |
| DATE / TIME | Temporal expressions |
| MONEY | Monetary values |
| … | 13+ more entity types |

spaCy's `en_core_web_sm` is a small, fast model trained on web text.
For higher accuracy, use `en_core_web_lg` or `en_core_web_trf`.
""")

    ner_samples = get_ner_texts()
    tab1, tab2 = st.tabs(["✏️ Single Text", "📋 Sample Gallery"])

    # ── Tab 1: Single Text ────────────────────────────────────────────────────
    with tab1:
        use_sample_ner = st.toggle("Use a sample text", value=True)
        if use_sample_ner:
            selected = st.selectbox("Choose sample", ner_samples, label_visibility="collapsed")
            input_text = selected
        else:
            input_text = st.text_area(
                "Enter text",
                height=110,
                placeholder="Type or paste any text to extract named entities…",
            )

        if st.button("🏷️ Extract Entities", type="primary", use_container_width=True):
            if not input_text.strip():
                st.warning("Please enter some text.")
            else:
                with st.spinner("Loading spaCy model and extracting entities…"):
                    result = run_ner(input_text)

                if not result["model_loaded"]:
                    st.error(f"spaCy error: {result['error']}")
                    st.code("python -m spacy download en_core_web_sm", language="bash")
                elif result["entity_count"] == 0:
                    st.info("No named entities detected in this text.")
                else:
                    # Highlighted text
                    st.markdown("#### 📝 Highlighted Text")
                    st.markdown(result["highlighted_html"], unsafe_allow_html=True)

                    # Entity table + summary
                    st.markdown("#### 📊 Entities Found")
                    e_col, s_col = st.columns([3, 2])

                    with e_col:
                        st.dataframe(
                            result["entities_df"][["Entity", "Label", "Type"]],
                            use_container_width=True,
                            hide_index=True,
                        )
                    with s_col:
                        st.metric("Total Entities", result["entity_count"])
                        label_counts = result["entities_df"]["Label"].value_counts()
                        fig, ax = plt.subplots(figsize=(4, 3))
                        colors = plt.cm.Set3(np.linspace(0, 1, len(label_counts)))
                        ax.bar(label_counts.index, label_counts.values, color=colors)
                        ax.set_ylabel("Count")
                        ax.set_title("Entities by Type", pad=6)
                        plt.xticks(rotation=35, ha="right", fontsize=8)
                        ax.spines[["top", "right"]].set_visible(False)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

    # ── Tab 2: Sample Gallery ─────────────────────────────────────────────────
    with tab2:
        st.markdown("#### Batch NER on all sample texts")
        if st.button("🚀 Run on All Samples", type="primary", use_container_width=True):
            with st.spinner("Processing…"):
                for i, text in enumerate(ner_samples, 1):
                    result = run_ner(text)
                    with st.expander(f"[{i}] {text[:75]}…"):
                        if result["entity_count"] > 0:
                            st.markdown(result["highlighted_html"], unsafe_allow_html=True)
                            st.dataframe(
                                result["entities_df"][["Entity", "Label", "Type"]],
                                use_container_width=True,
                                hide_index=True,
                            )
                        else:
                            st.info("No entities detected.")


# ═══════════════════════════════════════════════════════════════════════════════
#  SENTIMENT ANALYSIS PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💬 Sentiment Analysis":
    page_header(
        "Sentiment Analysis",
        "VADER · TextBlob · DistilBERT — side-by-side comparison",
        "💬",
    )

    info_expander("About the three analysers", """
| Analyser | Approach | Best For |
|----------|---------|---------|
| **VADER** | Lexicon + rules | Social media, short informal text |
| **TextBlob** | Pattern matching | General prose, reviews |
| **DistilBERT** | Fine-tuned transformer | High-accuracy classification |

All three return a **label** (POSITIVE / NEGATIVE / NEUTRAL) and a
**confidence score** (0–1). Scores come from different scales, so treat
them comparatively rather than absolutely.
""")

    sent_samples = get_sentiment_sentences()
    tab1, tab2 = st.tabs(["✏️ Single Text", "📋 Batch Analysis"])

    # ── Tab 1: Single Text ────────────────────────────────────────────────────
    with tab1:
        use_sample_sent = st.toggle("Use a sample sentence", value=True)
        if use_sample_sent:
            selected_sent = st.selectbox(
                "Choose sample", sent_samples, label_visibility="collapsed"
            )
            input_sent = selected_sent
        else:
            input_sent = st.text_area(
                "Enter a sentence",
                height=100,
                placeholder="Type any sentence to analyse its sentiment…",
            )

        use_transformer = st.toggle(
            "Include DistilBERT transformer",
            value=False,
            help="Requires ~260 MB model download on first run.",
        )

        if st.button("💬 Analyse Sentiment", type="primary", use_container_width=True):
            if not input_sent.strip():
                st.warning("Please enter a sentence.")
            else:
                with st.spinner("Analysing…"):
                    result = run_sentiment(input_sent, use_transformer=use_transformer)

                vader_r = result["vader"]
                blob_r  = result["textblob"]
                trans_r = result["transformer"]

                # ── Summary cards ──────────────────────────────────────────
                st.markdown("#### Results")
                n_cols = 3 if use_transformer and trans_r else 2
                cols = st.columns(n_cols)

                def render_card(col, title: str, res: dict) -> None:
                    label = res.get("label", "—")
                    score = res.get("score", 0.0)
                    icon, fg, bg = SENTIMENT_COLOURS.get(label, ("❓", "#333", "#eee"))
                    with col:
                        st.markdown(
                            f'<div style="background:{bg};border-radius:12px;'
                            f'padding:18px 20px;border:1px solid rgba(0,0,0,0.08);">'
                            f'<div style="font-weight:700;font-size:0.85rem;color:#555;'
                            f'margin-bottom:8px;text-transform:uppercase;">{title}</div>'
                            f'<div style="font-size:2rem;margin-bottom:4px;">{icon}</div>'
                            f'<div style="font-weight:800;font-size:1.3rem;color:{fg};">{label}</div>'
                            f'<div style="margin-top:8px;">'
                            f'<div style="background:rgba(0,0,0,0.1);border-radius:4px;height:8px;">'
                            f'<div style="background:{fg};width:{int(score*100)}%;height:8px;border-radius:4px;"></div>'
                            f'</div>'
                            f'<small style="color:{fg};font-weight:600;">{int(score*100)}% confidence</small>'
                            f'</div></div>',
                            unsafe_allow_html=True,
                        )

                render_card(cols[0], "VADER",    vader_r)
                render_card(cols[1], "TextBlob", blob_r)
                if use_transformer and trans_r:
                    render_card(cols[2], "DistilBERT", trans_r)

                # ── Detailed breakdown ─────────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### Detailed Scores")
                d_cols = st.columns(n_cols)

                with d_cols[0]:
                    st.markdown("**VADER**")
                    vader_detail = {
                        "Positive":  vader_r.get("pos", 0),
                        "Negative":  vader_r.get("neg", 0),
                        "Neutral":   vader_r.get("neu", 0),
                        "Compound":  vader_r.get("compound", 0),
                    }
                    st.dataframe(
                        pd.DataFrame(vader_detail.items(), columns=["Metric", "Value"])
                          .style.format({"Value": "{:.4f}"})
                          .background_gradient(subset=["Value"], cmap="RdYlGn"),
                        hide_index=True, use_container_width=True,
                    )

                with d_cols[1]:
                    st.markdown("**TextBlob**")
                    blob_detail = {
                        "Polarity":     blob_r.get("polarity", 0),
                        "Subjectivity": blob_r.get("subjectivity", 0),
                    }
                    st.dataframe(
                        pd.DataFrame(blob_detail.items(), columns=["Metric", "Value"])
                          .style.format({"Value": "{:.4f}"}),
                        hide_index=True, use_container_width=True,
                    )

                if use_transformer and trans_r and len(d_cols) > 2:
                    with d_cols[2]:
                        st.markdown("**DistilBERT**")
                        st.dataframe(
                            pd.DataFrame({
                                "Metric": ["Label", "Confidence"],
                                "Value":  [trans_r.get("label", "—"),
                                           trans_r.get("score", 0)],
                            }),
                            hide_index=True, use_container_width=True,
                        )

                # ── Confidence chart ───────────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### Confidence Comparison")
                chart_data = {
                    "VADER":    vader_r.get("score", 0),
                    "TextBlob": blob_r.get("score", 0),
                }
                if use_transformer and trans_r:
                    chart_data["DistilBERT"] = trans_r.get("score", 0)

                fig, ax = plt.subplots(figsize=(6, 2.5))
                bar_colors = ["#4A90D9", "#28a745", "#8e44ad"]
                bars = ax.bar(
                    list(chart_data.keys()),
                    list(chart_data.values()),
                    color=bar_colors[:len(chart_data)],
                    width=0.45,
                )
                ax.set_ylim(0, 1.15)
                ax.set_ylabel("Confidence Score")
                ax.set_title("Confidence by Analyser", pad=8)
                ax.spines[["top", "right"]].set_visible(False)
                for bar in bars:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f"{bar.get_height():.2f}",
                        ha="center", fontsize=9, color="#333",
                    )
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    # ── Tab 2: Batch ──────────────────────────────────────────────────────────
    with tab2:
        st.markdown("#### Batch Comparison across Sample Sentences")
        st.caption("Runs VADER and TextBlob on all sample sentences simultaneously.")

        incl_transformer_batch = st.toggle(
            "Include DistilBERT (slower)",
            value=False,
            key="batch_transformer",
        )

        if st.button("🚀 Run Batch Analysis", type="primary", use_container_width=True):
            with st.spinner("Analysing all sentences…"):
                batch_df = batch_compare(sent_samples, use_transformer=incl_transformer_batch)

            def _colour_label(val: str) -> str:
                return {
                    "POSITIVE": "background-color:#d4edda;color:#155724",
                    "NEGATIVE": "background-color:#f8d7da;color:#721c24",
                    "NEUTRAL":  "background-color:#fff3cd;color:#856404",
                }.get(val, "")

            label_cols = [c for c in batch_df.columns if c != "Text"]
            st.dataframe(
                batch_df.style.applymap(_colour_label, subset=label_cols),
                use_container_width=True,
                hide_index=True,
                height=320,
            )

            if "VADER" in batch_df.columns and "TextBlob" in batch_df.columns:
                agree = (batch_df["VADER"] == batch_df["TextBlob"]).sum()
                total = len(batch_df)
                st.metric(
                    "VADER ↔ TextBlob Agreement",
                    f"{agree}/{total} sentences",
                    delta=f"{int(agree / total * 100)}%",
                )

            # Distribution pie charts
            st.markdown("#### Label Distribution per Analyser")
            pie_cols = st.columns(len(label_cols))
            for i, col_name in enumerate(label_cols):
                counts = batch_df[col_name].value_counts()
                fig, ax = plt.subplots(figsize=(3, 3))
                pie_colors = [
                    "#28a745" if lbl == "POSITIVE"
                    else "#dc3545" if lbl == "NEGATIVE"
                    else "#ffc107"
                    for lbl in counts.index
                ]
                ax.pie(
                    counts.values,
                    labels=counts.index,
                    autopct="%1.0f%%",
                    colors=pie_colors,
                    startangle=90,
                    textprops={"fontsize": 8},
                )
                ax.set_title(col_name, fontsize=10, pad=6)
                with pie_cols[i]:
                    st.pyplot(fig)
                plt.close(fig)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center;color:#aaa;font-size:0.8rem;">'
    "NLP Toolkit · Built with Streamlit · spaCy · gensim · HuggingFace · scikit-learn"
    "</p>",
    unsafe_allow_html=True,
)
