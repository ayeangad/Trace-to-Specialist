from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = 20261008

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR_KAGGLE = "/kaggle/working/qwen25-3b-sft"       # Phase 1 output
STALE_ADAPTER_DIR_KAGGLE = "/kaggle/working/qwen25-3b-sft/checkpoint-100"  # Phase 8 "stale" adapter

TAIL_DIR = ROOT / "data" / "tail"
TAIL_FILINGS = TAIL_DIR / "filings.json"
TAIL_TASKS = TAIL_DIR / "tasks.jsonl"            # fit + cal + test (all slices except D1)
TAIL_DRIFT_TASKS = TAIL_DIR / "drift_tasks.jsonl"  # D1 only (Phase 7)
EXP = ROOT / "experiments" / "tailguard"

SLICES = ["S0", "T1", "T2", "T3", "T4", "T5", "T6"]  # D1 is drift-only
SPLITS = ["fit", "cal", "test"]
N_TASKS = {"fit": 500, "cal": 700, "test": 800}
SLICE_SHARE = {"S0": 0.64, "T1": 0.06, "T2": 0.06, "T3": 0.06, "T4": 0.06, "T5": 0.06, "T6": 0.06}
N_DRIFT = 200

MAX_TURNS = 6
MAX_NEW_TOKENS = 256
PROMPT_CHAR_WINDOW = 12000   # same truncation as evals/run_model_baseline.py
K_SAMPLES = 5
SAMPLE_TEMPERATURE = 0.7
SAMPLE_TOP_P = 0.95

EPSILONS = [0.01, 0.02, 0.05]
DELTA = 0.10
PRIMARY_EPS = 0.02
LTT_MIN_COVERAGE = 0.10      # fixed-sequence grid starts here (label-free)
LTT_GRID_STEP = 0.01
N_RESPLITS = 200

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
KNN_K = 10

# OpenAI (parameters per docs/tailguard/openai_caps.json — do not assume)
OPENAI_FRONTIER_MODEL = "gpt-5.4-mini"   # escalation target + frontier baseline (small pool)
OPENAI_JUDGE_MODEL = "gpt-4.1-mini"      # judge signal; chosen because it is non-reasoning and should return logprobs (VERIFY)
OPENAI_STRONG_MODEL = "gpt-5.4"          # optional 100-item sanity check only (large pool)
POOL_SMALL = {"gpt-5.4-mini", "gpt-5.4-nano", "gpt-5-mini", "gpt-5-nano", "gpt-4.1-mini",
              "gpt-4.1-nano", "gpt-4o-mini", "o3-mini", "o4-mini"}
POOL_LARGE = {"gpt-5.4", "gpt-5.2", "gpt-5.1", "gpt-5", "gpt-4.1", "gpt-4o", "o1", "o3"}
DAILY_CAP = {"small": 2_300_000, "large": 230_000}   # 8% safety margin under the free limits
