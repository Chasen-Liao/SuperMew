"""评测配置"""
from pathlib import Path

# eval_config.py 在 backend/eval/ 下，项目根目录在其 parent.parent.parent
BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATASET_PATH = BASE_DIR / "data" / "cmrc2019_dev.json"
EVAL_COLLECTION = "eval_cmrc2019"
PARENT_CHUNK_STORE_PATH = BASE_DIR / "data" / "eval_parent_chunks.json"

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128
TOP_K = 5
OVERLAP_THRESHOLD = 0.3
RESULTS_DIR = BASE_DIR / "eval_results"
RESULTS_DIR.mkdir(exist_ok=True)
