from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.problem.wireless_generator import generate_all_wireless


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="archive/data/datasets_wireless")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--window-width", type=float, default=30.0)
    parser.add_argument("--rate-threshold-mbps", type=float, default=1.0)
    args = parser.parse_args()

    out = ROOT / args.out
    generate_all_wireless(
        out,
        seeds=args.seeds,
        window_width=args.window_width,
        rate_threshold_mbps=args.rate_threshold_mbps,
    )
    print(f"Generated wireless benchmark instances under {out}")


if __name__ == "__main__":
    main()
