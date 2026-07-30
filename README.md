# Neural Career Advisor (NCA)

A hybrid AI career copilot for software developers. NCA combines a **Retrieval-Augmented Generation (RAG) chatbot** with an **XGBoost Machine Learning Predictive Engine**. Instead of relying on a language model's guesswork, it grounds all answers—salary expectations, skill trends, and role fit—in over 360,000 real data points from 5 years of [Stack Overflow Developer Surveys (2021–2025)](https://survey.stackoverflow.co/).

Built for **EE-404 Advanced Data Science** under the supervision of **Dr. Amirhosain Salavati**.

---

## How It Works

1. **Preprocessing** — Merge and clean raw survey responses from 2021 to 2025, parse multi-select fields, and handle outliers (e.g., capping extreme salary inputs).
2. **Fact Generation** — Compute aggregated, time-stamped statistics (salary by role/country/experience, tech adoption, job satisfaction) and generate over 12,000 natural-language fact sentences.
3. **Machine Learning** — Train an XGBoost regression model on historical data to predict future developer salaries based on economic trends and specific user inputs.
4. **FAISS Vector Store** — Embed the generated fact sentences using `sentence-transformers` and index them for rapid semantic search.
5. **RAG + LLM Orchestration** — Retrieve the most relevant facts for a user's question, passing them to a local LLM to generate a strict, data-backed answer without hallucinating numbers.
6. **System Evaluation** — Validate the pipeline using RAG Hit Rate, Faithfulness checks, and ML regression metrics (MAE, RMSE, $R^2$).
7. **Interactive UI** — A modern Streamlit application featuring live typing effects, predictive forms, and data transparency links.

---

## Project Structure

```text
neural-career-advisor-main/
├── 01_data_understanding.ipynb    # Initial data exploration
├── 02_cleaning_eda.ipynb          # Multi-year merging, cleaning, and EDA
├── 03_fact_generation.ipynb       # Temporal Knowledge Base generation
├── 04_build_faiss_index.ipynb     # Embedding and FAISS indexing
├── 05_rag_chat_logic.ipynb        # LLM orchestration and RAG testing
├── 06_system_evaluation.ipynb     # RAG & ML metrics (MAE, R2, Hit Rate)
├── app.py                         # Streamlit application (Chat, Predictor, Data)
├── salary_utils.py                # Helper functions for the ML pipeline
├── config.py                      # Global configuration (Paths, Model names, Thresholds)
├── config.toml                    # Streamlit UI theme config
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── .gitignore                     
├── .gitattributes                 
└── data/                          # Ignored in version control
    ├── 2021.txt, 2022.csv, ...    # Raw multi-year datasets
    ├── results.txt, schema.txt    # 2025 Dataset
    ├── merged_results.csv         # Unified historical dataset
    ├── facts.csv                  # 12,000+ generated text facts
    ├── facts.faiss                # Vectorized index for search
    ├── salary_predictor.pkl       # Trained XGBoost model
    └── evaluation_results.csv     # Exported metrics
```
## Getting the Data

This project relies on the official Stack Overflow Developer Survey datasets from **2021 to 2025**. These files (`results.txt`, `schema.txt`, and the annual CSV/TSV exports) are publicly available but are **not included** in this repository due to size constraints.

👉 **Download** the archives from either:
- The [official Stack Overflow survey page](https://survey.stackoverflow.co/), or
- The [GitHub repository](https://github.com/StackOverflow/StackOverflow-Developer-Survey) (look for the yearly releases).

After downloading, extract all files and place them directly inside the `data/` directory at the project root. The raw data is licensed under the **Open Database License (ODbL)** – feel free to use it for research and educational purposes.

---

## Setup

### 1. Install Python Dependencies

Make sure you have Python 3.9+ installed, then run:

```bash
pip install -r requirements.txt
```

### 2. Install and Run Ollama (Local LLM)

This project uses [Ollama](https://ollama.com/) to serve the language model **locally, securely, and free of charge**.

- Download and install Ollama from [ollama.com](https://ollama.com/).
- Once installed, pull the required model (default is `qwen2.5`, as defined in `config.py`):

    ```bash
    ollama pull qwen2.5
    ```

- **Keep Ollama running** in the background before you start the application or execute the RAG notebooks.

---

## Running the Application

Follow these steps to run the full pipeline:

1. Execute the Jupyter notebooks (from `01` to `06`) in order – this will:
   - Clean and merge the raw survey data.
   - Generate over 12,000 fact sentences.
   - Build the FAISS vector index.
   - Train the XGBoost salary predictor.
   - Evaluate system performance.

2. Verify that Ollama is active (it should be listening on `http://localhost:11434` by default).

3. Launch the Streamlit interface from the project root:


The app will open in your browser, providing interactive chat, salary prediction, and data exploration features.

---

## Roadmap

- [x] Step 0 — Project setup & GitHub
- [x] Step 1 — Data understanding
- [x] Step 2 — Cleaning & EDA (2021–2025 Integration)
- [x] Step 3 — Time‑Aware Fact Generation
- [x] Step 4 — Build FAISS Index
- [x] Step 5 — RAG Chat Logic & Orchestration
- [x] Step 6 — Machine Learning Predictive Engine
- [x] Step 7 — Comprehensive System Evaluation
- [x] Step 8 — Streamlit Interface & Polish

---

## Team

| Role | Name |
| :--- | :--- |
| Data Engineering & RAG Architecture | Mohammad Mosayeb zadeh |
| Vectorization & FAISS Implementation | Iliya Barary |
| LLM Integration & Interface Design | Mani Gholampour |

---

**EE‑404 Advanced Data Science — Dr. Amirhosain Salavati**

