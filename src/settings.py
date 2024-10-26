from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATE = datetime.now().strftime("%Y-%m-%d")
PROJECT_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_PATH / "data"
DATASETS_PATH = DATA_PATH / "datasets"
CACHE_PATH = DATA_PATH / ".cache"
CACHE_PATH.mkdir(parents=True, exist_ok=True)
