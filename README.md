# Neural Career Advisor (NCA)

A hybrid AI career copilot for software developers. NCA combines a **Retrieval-Augmented Generation (RAG) chatbot** with an **XGBoost Machine Learning Predictive Engine**. Instead of relying on a language model's guesswork, it grounds all answers—salary expectations, skill trends, and role fit—in over 360,000 real data points from 5 years of [Stack Overflow Developer Surveys (2021–2025)](https://survey.stackoverflow.co/).

Built for **EE-404 Advanced Data Science** under the supervision of **Dr. Amirhosain Salavati**.

## How it works

1. **Preprocessing** — Merge and clean raw survey responses from 2021 to 2025, parse multi-select fields, and handle outliers (e.g., capping extreme salary inputs).
2. **Fact Generation** — Compute aggregated, time-stamped statistics (salary by role/country/experience, tech adoption, job satisfaction) and generate over 12,000 natural-language fact sentences.
3. **Machine Learning** — Train an XGBoost regression model on historical data to predict future developer salaries based on economic trends and specific user inputs.
4. **FAISS Vector Store** — Embed the generated fact sentences using `sentence-transformers` and index them for rapid semantic search.
5. **RAG + LLM Orchestration** — Retrieve the most relevant facts for a user's question, passing them to a local LLM to generate a strict, data-backed answer without hallucinating numbers.
6. **System Evaluation** — Validate the pipeline using RAG Hit Rate, Faithfulness checks, and ML regression metrics (MAE, RMSE, $R^2$).
7. **Interactive UI** — A modern Streamlit application featuring live typing effects, predictive forms, and data transparency links.

## Project structure

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