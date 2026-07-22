# Neural Career Advisor (NCA)

A retrieval-augmented generation (RAG) chatbot that answers developer career questions — salary expectations, which skills to learn next, role fit — grounded in real data from the [Stack Overflow Developer Survey 2025](https://survey.stackoverflow.co/2025), instead of a language model's guesswork.

Built for **EE-404 Advanced Data Science**.

## How it works

1. **Preprocessing** — clean the raw survey responses, parse multi-select fields, handle missing values
2. **Fact generation** — compute aggregated statistics (salary by role/country/experience, tech adoption, job satisfaction, AI sentiment) and turn each into a natural-language fact sentence
3. **FAISS vector store** — embed the fact sentences and index them for semantic search
4. **RAG + LLM** — retrieve the most relevant facts for a user's question, then generate a grounded natural-language answer using *only* that retrieved context
5. **Chat interface** — a Streamlit app for asking questions in plain English

## Project structure

```
neural-career-advisor/
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_cleaning_eda.ipynb
│   ├── 03_fact_generation.ipynb
│   ├── 04_build_faiss_index.ipynb
│   └── 05_rag_chat_logic.ipynb
├── app/
│   ├── .streamlit/config.toml   # theme
│   └── app.py                   # Streamlit chat interface
├── data/                    # raw survey CSVs go here — not committed, see below
├── requirements.txt
├── .gitignore
└── README.md
```

## Running the chatbot

1. Run Notebooks 1 through 5 once, in order, so `data/facts.csv` and `data/facts.faiss` exist
2. Make sure Ollama is running locally with the model pulled (see Setup below)
3. From the project root:

```bash
streamlit run app/app.py
```

## Getting the data

This project uses the Stack Overflow Developer Survey 2025 — `results.txt` (response data) and `schema.txt` (question reference) — licensed under the [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/1-0/). Both are plain CSVs despite the `.txt` extension (that's how they come from the GitHub archive). Download from the [official survey site](https://survey.stackoverflow.co/2025), the [official GitHub archive](https://github.com/StackExchange/Survey), or a Kaggle mirror, and place both files in `data/`. The raw data isn't committed to this repo (file size + license terms).

## Setup

```bash
pip install -r requirements.txt
```

This project uses [Ollama](https://ollama.com/download) to run the LLM locally and for free — install it separately from the pip packages above, then pull the model once:

```bash
ollama pull llama3.2
```


Ollama needs to be running in the background whenever you use Notebook 5 or the app (the desktop app does this automatically).
```bash
ollama run llama3.2
```
for runing app.py you need to use streamlit:

```bash
streamlit run app.py
```


## Roadmap

- [x] Step 0 — Project setup & GitHub
- [x] Step 1 — Data understanding
- [x] Step 2 — Cleaning & EDA
- [x] Step 3 — Fact generation
- [x] Step 4 — Build FAISS index
- [x] Step 5 — RAG chat logic
- [x] Step 6 — Streamlit interface
- [ ] Step 7 — Polish & scalability

## Team

| Role | Name |
|---|---|
| Data Engineering & RAG Architecture | Mohammad Mosayeb zadeh |
| Vectorization & FAISS Implementation | Iliya Barary |
| LLM Integration & Interface Design | Mani Gholampour |

EE-404 Advanced Data Science — Dr. Amirhosain Salavati
