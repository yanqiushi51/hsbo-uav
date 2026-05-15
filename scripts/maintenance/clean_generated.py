"""Clean ignored local build and runtime artifacts.

The script is conservative by design: it only considers files and directories
that Git already reports as ignored. It never removes tracked result snapshots.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def ignored_paths() -> list[Path]:
    result = subprocess.run(
        ["git", "status", "--ignored", "--short", "--untracked-files=normal"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    paths: list[Path] = []
    for line in result.stdout.splitlines():
        if not line.startswith("!! "):
            continue
        raw_path = line[3:].rstrip("/")
        paths.append(ROOT / raw_path)
    return sorted(paths)


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print ignored files that would be removed. This is the default.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually remove ignored files and directories.",
    )
    args = parser.parse_args()

    if args.dry_run and args.apply:
        parser.error("Choose either --dry-run or --apply, not both.")

    apply = args.apply
    removable = ignored_paths()

    if not removable:
        print("No ignored generated files found.")
        return

    action = "Removing" if apply else "Would remove"
    for path in removable:
        print(f"{action}: {path.relative_to(ROOT)}")
        if apply:
            remove_path(path)

    if not apply:
        print("\nDry run only. Re-run with --apply to remove these paths.")


if __name__ == "__main__":
    main()
