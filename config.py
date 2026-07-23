"""
Shared configuration for the Neural Career Advisor.

This is the one file to touch when swapping in a new year's survey data —
everything downstream (notebooks 1-5 and app.py) reads from here instead of
repeating these values. See README.md, "Swapping in a new year's data."
"""

# --- Data source ---
SURVEY_YEAR = 2025
RAW_RESULTS_FILE = "results.txt"
RAW_SCHEMA_FILE = "schema.txt"

# --- Fact generation (Notebook 3) ---
MIN_N = 30  # never generate a fact from fewer respondents than this

# --- Retrieval (Notebooks 4-5, app.py) ---
EMBED_MODEL = "all-MiniLM-L6-v2"
MIN_SCORE = 0.35  # minimum FAISS similarity before a retrieved fact is trusted
TOP_K = 7  # how many facts to retrieve per question
MAX_CONTEXT_CHARS = 3000  # defensive cap on retrieved-context size handed to the LLM

# --- LLM (Notebook 5, app.py) ---
OLLAMA_MODEL = "llama3.2"

# --- Schema check (Notebook 1) ---
# Columns the whole pipeline depends on existing, by exact name, in the raw data.
# If next year's survey adds, removes, or renames any of these, Notebook 1's
# validation check will flag it immediately, rather than a confusing failure
# three notebooks later.
REQUIRED_COLUMNS = [
    "ResponseId", "DevType", "Country", "ConvertedCompYearly", "WorkExp", "YearsCode",
    "EdLevel", "ICorPM", "OrgSize", "RemoteWork", "JobSat", "AISent", "AIThreat",
    "AIAcc", "AIComplex", "AIAgents", "NewRole",
    "LanguageHaveWorkedWith", "DatabaseHaveWorkedWith", "PlatformHaveWorkedWith",
    "WebframeHaveWorkedWith", "AIModelsHaveWorkedWith",
]
