"""Phase 10 inference utility entry point."""

from pathlib import Path


def main() -> None:
    """Report Phase 10 inference status without starting a science campaign."""

    priors_path = Path("paper/tables/priors.csv")
    if not priors_path.exists():
        raise SystemExit("Missing frozen prior table: paper/tables/priors.csv")
    rows = priors_path.read_text(encoding="utf-8").strip().splitlines()
    prior_count = max(0, len(rows) - 1)
    print("Phase 10 inference API is implemented.")
    print(f"Frozen prior table: {priors_path} ({prior_count} parameters)")
    print("Phase 11 screening campaigns are not started by this script.")


if __name__ == "__main__":
    main()
