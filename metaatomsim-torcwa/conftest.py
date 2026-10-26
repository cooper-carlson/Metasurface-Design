"""Put the shared engines in core/ on sys.path for every project's tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'core'))
