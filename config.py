from pathlib import Path

ROOT = Path(__file__).resolve().parent


RAW_CSV_CANDIDATES = [
    ROOT.parent / "Dataset 3 dhe 1 FINAL Replies Normalized Ready (without profilename+url).csv",
    ROOT / "data" / "raw.csv",
]
DATA_DIR = ROOT / "data"
FOLDS_CSV = DATA_DIR / "phs_folds.csv"
COUNTERFACTUAL_CSV = DATA_DIR / "counterfactual.csv"
NATURAL_CF_CSV = DATA_DIR / "natural_cf.csv"
GPU_PACKAGE_DATA = ROOT / "gpu_baselines" / "data"


NON_HS_CLASS = 0
PHS_CLASS = 3
N_NON_HS = 1000
SAMPLE_SEED = 42
N_FOLDS = 5
FOLD_SEED = 42


RESULTS = ROOT / "results"
CLASSIFIER_PREDS = RESULTS / "classifier_preds"
LLM_RUNS = RESULTS / "llm_runs"
LLM_CACHE = RESULTS / "llm_cache"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
LOGS = ROOT / "logs"
KB_DIR = ROOT / "kb"


EXISTING_MBERT_DIR = ROOT / "Political_HS_BERT" / "Political_HS_BERT"


AGENT_CLASSIFIER = "mbert_orig+jev"


CLASSIFIER_DESCRIPTIONS = {
    "mbert_orig": "multilingual BERT fine-tuned on labelled Albanian comments (tends to over-react to names "
                  "of politicians, parties and institutions)",
    "jev": "Jev, a zero-shot System One decision model (never trained on these labels; less sure on short "
           "dialect comments)",
    "jev_ex": "Jev with a few labelled examples",
    "tfidf_lr": "TF-IDF logistic regression trained on labelled Albanian comments",
    "mbert_post": "multilingual BERT fine-tuned on labelled Albanian comments (tends to over-react to names "
                  "of politicians, parties and institutions)",
}

REV2_DIR = DATA_DIR / "rev2"
REV2_CLASSIFIER = "mbert_post+jev"

S1_VARIANTS = ["mbert_orig", "jev", "jev_ex", "agree"]


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


LLM_MODELS = {
    "qwen3.8-27b": "qwen/qwen3.8-27b",
    "gemini-3.8-flash": "google/gemini-3.8-flash",
    "gpt-5.6-luna": "openai/gpt-5.6-luna",
}

STRONG_MODEL = {"claude-sonnet-5.5": "anthropic/claude-sonnet-5.5"}


DEFAULT_EXTRA = {"reasoning": {"enabled": False}}


MODEL_EXTRA = {}


CHEAP_MODELS = {
    "qwen3.7-flash": "qwen/qwen3.7-flash",
    "qwen3.8-flash": "qwen/qwen3.8-flash",
    "gemma-4-31b": "google/gemma-4-31b-it",
}


FREE_MODELS = {
    "gemma-4-31b-free": "google/gemma-4-31b-it:free",
    "qwen3.8-27b-free": "qwen/qwen3.8-27b:free",
    "nemotron-3-ultra-free": "nvidia/nemotron-3-ultra-550b-a55b:free",
}
FREE_MIN_INTERVAL_S = 3.2


FREE_PLAN = [
    ("main", "zero_shot", "nemotron-3-ultra-free"), ("main", "agent_full", "nemotron-3-ultra-free"),
]


REVISION_PLAN = [
    ("ncf", "zero_shot", "qwen3.7-flash"), ("ncf", "agent_full", "qwen3.7-flash"),
    ("ncf", "agent_no_classifier", "qwen3.7-flash"), ("ncf", "agent_no_entities", "qwen3.7-flash"),
    ("cf", "zero_shot", "gemma-4-31b"), ("cf", "agent_full", "gemma-4-31b"),
    ("ncf", "zero_shot", "gemma-4-31b"), ("ncf", "agent_full", "gemma-4-31b"),
    ("ncf", "zero_shot", "qwen3.8-flash"), ("ncf", "agent_full", "qwen3.8-flash"),
    ("ncf", "agent_no_entities", "qwen3.8-flash"),
]

REVISION_FREE_PLAN = [
    ("cf", "zero_shot", "nemotron-3-ultra-free"), ("ncf", "zero_shot", "nemotron-3-ultra-free"),
    ("cf", "agent_full", "nemotron-3-ultra-free"), ("ncf", "agent_full", "nemotron-3-ultra-free"),
]


_R37 = "qwen3.7-flash"
REV2_PLAN = (   
    [(ds, "agent_full", m) for m in ("qwen3.8-flash", _R37, "gemma-4-31b") for ds in ("pval", "ptest")]
    + [("ptest", "single_all", m) for m in ("qwen3.8-flash", _R37, "gemma-4-31b")]
    + [("ptest", "zero_shot", m) for m in ("qwen3.8-flash", _R37, "gemma-4-31b")]
    + [("ptest", c, _R37) for c in ("agent_no_classifier", "single_no_classifier")]
    + [(ds, c, _R37) for ds in ("ncf2", "hn") for c in ("agent_full", "single_all", "zero_shot", "agent_no_classifier", "single_no_classifier")]
    + [(ds, c, "gemma-4-31b") for ds in ("ncf2", "hn") for c in ("agent_full", "single_all", "zero_shot")]
    + [("pval", c, m) for m in ("qwen3.8-flash", _R37, "gemma-4-31b") for c in ("single_all", "zero_shot")]
    + [("pval", c, _R37) for c in ("agent_no_classifier", "single_no_classifier")]
    + [("gen", c, _R37) for c in ("zero_shot", "agent_full", "single_all", "agent_no_classifier")]
    + [("cf", "agent_full", _R37), ("cf", "single_all", _R37)]
)
REV2_FREE_PLAN = ([("pval", "agent_full", "nemotron-3-ultra-free"), ("ptest", "agent_full", "nemotron-3-ultra-free"),
                   ("ptest", "single_all", "nemotron-3-ultra-free"), ("ptest", "zero_shot", "nemotron-3-ultra-free"),
                   ("ncf2", "agent_full", "nemotron-3-ultra-free"), ("hn", "agent_full", "nemotron-3-ultra-free"),
                   ("ncf2", "single_all", "nemotron-3-ultra-free"), ("hn", "single_all", "nemotron-3-ultra-free"),
                   ("pval", "single_all", "nemotron-3-ultra-free"), ("pval", "zero_shot", "nemotron-3-ultra-free")])

NEMOTRON_PAID_ID = "nvidia/nemotron-3-ultra-550b-a55b"
REV2_BIAS_PLAN = [("hn", "agent_full", "qwen3.8-flash"), ("ncf2", "agent_full", "qwen3.8-flash"), ("cf", "agent_full", "qwen3.8-flash"),
                  ("cf", "agent_full", "gemma-4-31b")]
REV2_NEMO_PAID_PLAN = [("ptest", "single_all", "nemotron-3-ultra-free"), ("hn", "agent_full", "nemotron-3-ultra-free"),
                       ("hn", "single_all", "nemotron-3-ultra-free"), ("ptest", "zero_shot", "nemotron-3-ultra-free")]


MODEL_SETS = {
    "poc": ["qwen3.7-flash", "gemma-4-31b"],
    "budget": ["qwen3.7-flash", "qwen3.8-flash", "gemma-4-31b"],
    "paper": list(LLM_MODELS),
}


BUDGET_USD = 6.00


PAPER_HARD_CAP_USD = 6.00


BUDGET_PLAN = [

    ("main", "zero_shot", "qwen3.8-flash"), ("main", "agent_full", "qwen3.8-flash"),
    ("cf", "zero_shot", "qwen3.8-flash"), ("cf", "agent_full", "qwen3.8-flash"),

    ("main", "zero_shot", "gemma-4-31b"), ("main", "agent_full", "gemma-4-31b"),

    ("main", "few_shot", "qwen3.8-flash"), ("main", "agent_no_classifier", "qwen3.8-flash"),
    ("main", "agent_no_entities", "qwen3.8-flash"), ("main", "agent_no_context", "qwen3.8-flash"),
    ("cf", "agent_no_entities", "qwen3.8-flash"),

    ("main", "zero_shot", "qwen3.7-flash"), ("main", "few_shot", "qwen3.7-flash"),
    ("main", "agent_full", "qwen3.7-flash"), ("main", "agent_no_classifier", "qwen3.7-flash"),
    ("main", "agent_no_entities", "qwen3.7-flash"), ("main", "agent_no_context", "qwen3.7-flash"),
    ("cf", "zero_shot", "qwen3.7-flash"), ("cf", "agent_full", "qwen3.7-flash"),
    ("cf", "agent_no_entities", "qwen3.7-flash"),

]

TEMPERATURE = 0.0
MAX_TOKENS = 1000
REQUEST_TIMEOUT = 120
MAX_RETRIES = 8
N_WORKERS = 20
AGENT_MAX_STEPS = 6
FEW_SHOT_K = 8
FEW_SHOT_SEED = 13


LLM_FOLDS = ["orig"]


import os as _os
if _os.environ.get("PHS_MOCK_LLM") == "1":
    LLM_RUNS = RESULTS / "mock" / "llm_runs"
    LLM_CACHE = RESULTS / "mock" / "llm_cache"
    TABLES = RESULTS / "mock" / "tables"
    FIGURES = RESULTS / "mock" / "figures"
