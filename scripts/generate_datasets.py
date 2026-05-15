from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.problem.benchmark_generator import generate_all


if __name__ == "__main__":
    out = ROOT / "datasets"
    generate_all(out, seeds=range(10))
    print(f"Generated benchmark instances under {out}")
