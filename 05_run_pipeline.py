"""
05_run_pipeline.py — Master pipeline runner
============================================
MODES:
  --mode synthetic   Generate fake data and run full analysis (free, instant)
  --mode ingest      Use JSON files you downloaded from Apify (recommended now)
  --mode live        Live scraping (requires working scraper + API key)

YOUR CURRENT WORKFLOW:
  # Step 1: put Apify JSON files in data/raw/ as treatment_*.json / control_*.json
  # Step 2: ingest + augment + analyse in one command:
  python3 05_run_pipeline.py --mode ingest

  # After first run, re-run analysis only (no re-ingestion):
  python3 05_run_pipeline.py --mode ingest --skip-scrape

LOG TO FILE:
  python3 05_run_pipeline.py --mode ingest | tee "outputs/pipeline_$(date +%Y%m%d_%H%M%S).log"
"""

import subprocess, sys, argparse, glob
from pathlib import Path
from datetime import datetime

# Create output dirs at import time so tee can write before Python fully starts
for d in ["outputs/figures","outputs/tables",
          "data/raw","data/processed","data/synthetic"]:
    Path(d).mkdir(parents=True, exist_ok=True)

PYTHON = sys.executable   # always uses the same python3 that launched this script


def run(cmd: str, description: str = ""):
    cmd = cmd.replace("python ", f'"{PYTHON}" ', 1)
    print(f"\n{'='*60}")
    if description:
        print(f"  STEP: {description}")
    print(f"  CMD:  {cmd}")
    print(f"  TIME: {datetime.now():%H:%M:%S}")
    print(f"{'='*60}")
    subprocess.run(cmd, shell=True, check=True)
    print(f"  Done at {datetime.now():%H:%M:%S}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-NCII pipeline")
    parser.add_argument(
        "--mode",
        choices=["synthetic", "ingest", "live"],
        default="synthetic",
        help=(
            "synthetic = fake data, instant, free | "
            "ingest    = parse your Apify JSON files from data/raw/ | "
            "live      = real-time scraping (requires API key)"
        )
    )
    parser.add_argument("--use-clip", action="store_true",
                        help="Enable CLIP image classifier in step 02")
    parser.add_argument(
        "--skip-scrape", action="store_true",
        help="Skip step 1 (data collection/ingestion) and reuse existing all_posts.csv"
    )
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════╗
║  AI-NCII on X: Full Analysis Pipeline                   ║
║  Data Science & Public Policy  Spring 2026              ║
║  Mode: {args.mode.upper():<51}║
║  Skip step 1: {str(args.skip_scrape):<44}║
║  Started: {datetime.now():%Y-%m-%d %H:%M:%S:<49}║
╚══════════════════════════════════════════════════════════╝
""")

    # ── STEP 1: DATA COLLECTION / INGESTION ──────────────────────────────────
    DATA = ""

    if args.mode == "synthetic":
        if not args.skip_scrape:
            run("python 01_data_collection.py --mode synthetic",
                "Generate synthetic dataset (calibrated to real data distributions)")
        DATA = "data/synthetic/synthetic_posts.csv"

    elif args.mode == "ingest":
        DATA = "data/processed/all_posts.csv"
        if args.skip_scrape:
            # Reuse whatever is already in data/processed/all_posts.csv
            print("\n[--skip-scrape] Skipping ingestion — using existing data/processed/all_posts.csv")
            if not Path(DATA).exists():
                print("[ERROR] data/processed/all_posts.csv not found.")
                print("  Run without --skip-scrape first to ingest your JSON files.")
                sys.exit(1)
            import pandas as pd
            print(f"  Found {len(pd.read_csv(DATA))} posts — proceeding to analysis.")
        else:
            # Ingest real JSON files from data/raw/, then augment to analysis size
            run("python 01_data_collection.py --mode ingest",
                "Ingest Apify JSON files from data/raw/")
            run("python 01_data_collection.py --mode augment",
                "Augment real posts with calibrated synthetic data (400+400 target)")

    elif args.mode == "live":
        DATA = "data/processed/all_posts.csv"
        if args.skip_scrape:
            print("\n[--skip-scrape] Skipping live scraping — using existing data.")
            if not Path(DATA).exists():
                print("[ERROR] No data found. Run without --skip-scrape first.")
                sys.exit(1)
        else:
            run("python 01_data_collection.py --mode treatment",
                "Scrape treatment posts via live scraper")
            run("python 01_data_collection.py --mode control",
                "Scrape control posts via live scraper")
            import pandas as pd
            csv_files = glob.glob("data/processed/*_posts.csv")
            merged = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
            merged.to_csv(DATA, index=False)
            print(f"  Merged {len(csv_files)} CSVs → {len(merged)} posts")

    # ── STEP 2: AI DETECTION ─────────────────────────────────────────────────
    clip_flag = "--use-clip" if args.use_clip else ""
    run(
        f"python 02_ai_detection.py --input {DATA} "
        f"--output data/processed/classified_posts.csv {clip_flag}".strip(),
        "Classify posts as AI-generated vs human-created"
    )

    # ── STEP 3: NETWORK ANALYSIS ─────────────────────────────────────────────
    run(f"python 03_network_analysis.py --input {DATA}",
        "Build account-hashtag network + community detection")

    # ── STEP 4: ENGAGEMENT ANALYSIS ──────────────────────────────────────────
    run(f"python 04_engagement_analysis.py --input {DATA}",
        "PSM matching + OLS regression for algorithmic amplification")

    print(f"""
╔══════════════════════════════════════════════════════════╗
║  Pipeline complete!                                     ║
╠══════════════════════════════════════════════════════════╣
║  Key outputs:                                           ║
║    outputs/figures/network_bipartite.png                ║
║    outputs/figures/engagement_violin.png                ║
║    outputs/figures/regression_coef_views.png            ║
║    outputs/tables/ttest_results.csv                     ║
║    outputs/tables/ols_regression_results.csv            ║
║    outputs/tables/centralization_test.csv               ║
╠══════════════════════════════════════════════════════════╣
║  Finished: {datetime.now():%Y-%m-%d %H:%M:%S:<47}║
╚══════════════════════════════════════════════════════════╝
""")
