# Neural Career Advisor (NCA)

A retrieval-augmented generation (RAG) chatbot that answers developer career questions — salary expectations, which skills to learn next, role fit — grounded in real data from the [Stack Overflow Developer Survey 2025](https://survey.stackoverflow.co/2025), instead of a language model's guesswork.

Built for **CS-404 Advanced Data Science**.

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
│   └── app.py              # Streamlit chat interface (Step 6)
├── data/                    # raw survey CSVs go here — not committed, see below
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting the data

This project uses the Stack Overflow Developer Survey 2025 — `survey_results_public.csv` and `survey_results_schema.csv` — licensed under the [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/1-0/). Download it from the [official survey site](https://survey.stackoverflow.co/2025), the [official GitHub archive](https://github.com/StackExchange/Survey), or a Kaggle mirror, and place both CSVs in `data/`. The raw data isn't committed to this repo (file size + license terms).

## Setup

```bash
pip install -r requirements.txt
```

## Roadmap

- [x] Step 0 — Project setup & GitHub
- [x] Step 1 — Data understanding
- [ ] Step 2 — Cleaning & EDA
- [ ] Step 3 — Fact generation
- [ ] Step 4 — Build FAISS index
- [ ] Step 5 — RAG chat logic
- [ ] Step 6 — Streamlit interface
- [ ] Step 7 — Polish & scalability

## Team

| Role | Name |
|---|---|
| Data Engineering & RAG Architecture | Mohammad Mosayeb zadeh |
| Vectorization & FAISS Implementation | Iliya Barary |
| LLM Integration & Interface Design | Mani Gholampour |

EE-404 Advanced Data Science — Dr.Amirhosain Salavati
