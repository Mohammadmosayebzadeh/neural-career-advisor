"""
Neural Career Advisor — Streamlit chat interface.

Standalone script (not a notebook) — Streamlit needs `streamlit run app.py` to work.
Expects data/facts.csv and data/facts.faiss to already exist (built by Notebooks 3-4),
and Ollama running locally with the model below pulled.
"""

from pathlib import Path

import faiss
import ollama
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(r"C:\\Users\\ASUS\\Desktop\\neural_career_advisor\\data")
print("DATA_DIR =", DATA_DIR)
print("facts.csv =", (DATA_DIR / "facts.csv").exists())
print("facts.faiss =", (DATA_DIR / "facts.faiss").exists())
OLLAMA_MODEL = "llama3.2"
MIN_SCORE = 0.35
TOP_K = 7

SYSTEM_PROMPT = (
    "You are a career advisor for software developers, answering strictly from the survey "
    "facts provided below. Rules:\n"
    "1. Use ONLY these facts — never add outside knowledge or general opinions about "
    "languages, tools, or careers.\n"
    "2. Start with a direct, clear answer in 1-2 sentences, then support it with specific facts.\n"
    "3. Weave each sample size naturally into its sentence (e.g. 'based on 200 respondents') "
    "— never list several sample sizes together as a bare group of numbers.\n"
    "4. If part of the question isn't covered by the facts (for example, a specific technology "
    "with no matching fact), say plainly which part is missing rather than talking around it.\n"
    "5. Keep the whole answer under 150 words."
)

EXAMPLE_QUESTIONS = [
    "I'm a junior dev in India learning Python. Should I learn Rust or Go next?",
    "Is a CS degree worth it compared to being self-taught?",
    "Does moving to Europe pay more?",
    "Should I become a manager or stay a developer?",
]

# ---------------------------------------------------------------------------
# Page setup + styling
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Neural Career Advisor", page_icon="\U0001F9ED", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .nca-hero {
        padding: 0.25rem 0 1.25rem 0;
        border-bottom: 1px solid #2A3140;
        margin-bottom: 1.25rem;
    }
    .nca-hero h1 {
        font-size: 1.6rem;
        font-weight: 600;
        color: #F2F0EA;
        margin-bottom: 0.15rem;
    }
    .nca-hero p {
        color: #8B93A1;
        font-size: 0.92rem;
        margin: 0;
    }
    .nca-mono {
        font-family: 'JetBrains Mono', monospace;
        color: #D9A544;
        font-weight: 600;
    }
    .nca-source {
        border-left: 2px solid #D9A544;
        padding: 0.4rem 0 0.4rem 0.75rem;
        margin-bottom: 0.5rem;
        font-size: 0.88rem;
    }
    .nca-source .nca-cat {
        color: #8B93A1;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    div.stButton > button {
        border: 1px solid #2A3140;
        background-color: #1B212B;
        color: #E8E6E1;
        text-align: left;
        border-radius: 8px;
    }
    div.stButton > button:hover {
        border-color: #D9A544;
        color: #D9A544;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="nca-hero">
        <h1>\U0001F9ED Neural Career Advisor</h1>
        <p>Grounded in the 2025 Stack Overflow Developer Survey — real answers from real developers, never invented.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load resources (cached — only runs once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the knowledge base...")
def load_resources():
    facts_path = DATA_DIR / "facts.csv"
    index_path = DATA_DIR / "facts.faiss"
    if not facts_path.exists() or not index_path.exists():
        return None, None, None
    facts_df = pd.read_csv(facts_path)
    index = faiss.read_index(str(index_path))
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return facts_df, index, embed_model


facts_df, index, embed_model = load_resources()

if facts_df is None:
    st.error(
        "Couldn't find `facts.csv` or `facts.faiss` in the `data/` folder. "
        "Run Notebooks 3 and 4 first — this app only displays what they build, it doesn't build it."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Retrieval + generation (same logic as Notebook 5)
# ---------------------------------------------------------------------------
def search(query, k=TOP_K, min_score=MIN_SCORE):
    q_vec = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_vec)
    scores, ids = index.search(q_vec, k)
    return [
        {
            "text": facts_df.iloc[i]["text"],
            "category": facts_df.iloc[i]["category"],
            "n": int(facts_df.iloc[i]["n"]),
            "score": float(s),
        }
        for i, s in zip(ids[0], scores[0])
        if s >= min_score
    ]


def build_user_message(question, retrieved_facts):
    context = "\n".join(f"- {f['text']}" for f in retrieved_facts)
    return f"Context (2025 Stack Overflow Developer Survey):\n{context}\n\nQuestion: {question}"


def ask(question):
    retrieved = search(question)
    if not retrieved:
        return {
            "answer": (
                "I don't have survey data that speaks to that directly — I can only answer "
                "from the 2025 Stack Overflow Developer Survey facts I've indexed."
            ),
            "sources": [],
        }
    user_message = build_user_message(question, retrieved)
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        answer = response["message"]["content"]
    except Exception as exc:
        answer = (
            f"Couldn't reach Ollama. Make sure it's installed and running "
            f"(`ollama serve`), and that the model is pulled (`ollama pull {OLLAMA_MODEL}`).\n\n"
            f"Error detail: {exc}"
        )
    return {"answer": answer, "sources": retrieved}


# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("### About this project")
    st.write(
        "Every answer here is retrieved from pre-computed survey statistics, then explained "
        "by a local LLM instructed to use only that retrieved data — not its own general knowledge."
    )
    c1, c2 = st.columns(2)
    c1.metric("Facts indexed", len(facts_df))
    c2.metric("Survey responses", "49,191")
    st.caption(f"Model: `{OLLAMA_MODEL}` (local, via Ollama)")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Empty-state: example questions to act on
if not st.session_state.messages:
    st.markdown("**Try asking:**")
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        if cols[i % 2].button(q, use_container_width=True, key=f"example_{i}"):
            st.session_state.pending_question = q

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    st.markdown(
                        f"""<div class="nca-source">
                        <span class="nca-cat">{s['category'].replace('_', ' ')}</span>
                        · <span class="nca-mono">n={s['n']}</span>
                        · match {s['score']:.2f}<br>{s['text']}
                        </div>""",
                        unsafe_allow_html=True,
                    )

# Get new input — typed, or from an example button clicked this run
typed = st.chat_input("Ask about salary, skills, or career moves...")
pending = st.session_state.pop("pending_question", None)
user_input = typed or pending

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Checking the survey data..."):
            result = ask(user_input)
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander(f"Sources ({len(result['sources'])})"):
                for s in result["sources"]:
                    st.markdown(
                        f"""<div class="nca-source">
                        <span class="nca-cat">{s['category'].replace('_', ' ')}</span>
                        · <span class="nca-mono">n={s['n']}</span>
                        · match {s['score']:.2f}<br>{s['text']}
                        </div>""",
                        unsafe_allow_html=True,
                    )

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )
