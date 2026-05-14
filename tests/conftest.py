import sys
from pathlib import Path


ROOT_PATH: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_PATH))
