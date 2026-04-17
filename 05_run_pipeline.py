"""
05_run_pipeline.py
==================
PURPOSE:
  Master orchestration script — runs all 4 analysis scripts in sequence.

PIPELINE STEPS:
  Step 1 → 01_data_collection.py     : Collect/generate posts
  Step 2 → 02_ai_detection.py        : Label which posts are AI-generated
  Step 3 → 03_network_analysis.py    : Build account-hashtag network
  Step 4 → 04_engagement_analysis.py : PSM + regression for amplification

USAGE:
  python3 05_run_pipeline.py --mode synthetic
  python3 05_run_pipeline.py --mode synthetic | tee "outputs/pipeline_$(date +%Y%m%d_%H%M%S).log"
  python3 05_run_pipeline.py --mode live

HOW THE LOG COMMAND WORKS:
  | tee "outputs/pipeline_TIMESTAMP.log"
    tee splits the stream: prints to screen AND writes to the log file.
    The outputs/ directory must exist before tee runs — we create it below
    at module load time (before any function runs) for exactly this reason.
"""

import subprocess
import sys
import argparse
import glob
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CREATE OUTPUT DIRECTORIES AT IMPORT TIME (before tee tries to write)
# ─────────────────────────────────────────────────────────────────────────────
# When you pipe this script to tee, the shell creates the tee file BEFORE
# Python starts. So outputs/ must already exist. We create it here.
Path("outputs/figures").mkdir(parents=True, exist_ok=True)
Path("outputs/tables").mkdir(parents=True, exist_ok=True)
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("data/synthetic").mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# PYTHON INTERPRETER PATH
# ─────────────────────────────────────────────────────────────────────────────
# sys.executable = full path to the Python binary running this script.
# e.g. /opt/homebrew/bin/python3 on an M-chip Mac.
# This avoids "python: command not found" since modern Macs only have python3.
PYTHON = sys.executable


def run(cmd: str, description: str = ""):
    """
    Runs a shell command, replacing 'python ' with the real interpreter path.
    Stops the whole pipeline immediately if any step fails (check=True).
    """
    cmd = cmd.replace("python ", f'"{PYTHON}" ', 1)
    print(f"\n{'='*60}")
    if description:
        print(f"  STEP: {description}")
    print(f"  CMD:  {cmd}")
    print(f"  TIME: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, check=True)
    print(f"  Done at {datetime.now().strftime('%H:%M:%S')}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full AI-NCII pipeline")
    parser.add_argument(
        "--mode", choices=["synthetic", "live"], default="synthetic",
        help="synthetic=fake data, live=real scraping via snscrape (free, no API key)"
    )
    parser.add_argument(
        "--use-clip", action="store_true",
        help="Enable CLIP image classifier in step 02 (requires torch+transformers)"
    )
    parser.add_argument(
        "--skip-scrape", action="store_true",
        help=(
            "Skip Apify scraping (Step 1) and reuse already-collected data. "
            "Use this after a successful --mode live run to avoid spending more credits. "
            "Example: python3 05_run_pipeline.py --mode live --skip-scrape"
        )
    )
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════╗
║  AI-NCII on X: Full Analysis Pipeline                   ║
║  Data Science & Public Policy  Spring 2026              ║
║  Mode: {args.mode.upper():<51}║
║  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<49}║
╚══════════════════════════════════════════════════════════╝
""")

    # ── STEP 1: DATA COLLECTION ───────────────────────────────────────────────
    if args.mode == "synthetic":
        # No API key needed. Generates 800 treatment + 800 control fake posts.
        run(
            "python 01_data_collection.py --mode synthetic",
            "Generate synthetic dataset (800 AI + 800 human posts)"
        )
        DATA = "data/synthetic/synthetic_posts.csv"

    else:
        DATA = "data/processed/all_posts.csv"

        if args.skip_scrape:
            # ─────────────────────────────────────────────────────────────
            # --skip-scrape: REUSE DATA ALREADY ON DISK
            # Use this after a successful live scrape to re-run only the
            # analysis steps (AI detection, network, engagement) without
            # triggering Apify again and spending more credits.
            # Command: python3 05_run_pipeline.py --mode live --skip-scrape
            # ─────────────────────────────────────────────────────────────
            print("\n[--skip-scrape] Skipping Apify calls. Reusing existing data.")
            if not Path(DATA).exists():
                import pandas as pd
                csv_files = glob.glob("data/processed/*_posts.csv")
                if not csv_files:
                    print("[ERROR] No scraped data found. Run without --skip-scrape first.")
                    sys.exit(1)
                merged = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
                merged.to_csv(DATA, index=False)
                print(f"  Rebuilt all_posts.csv from {len(csv_files)} CSVs ({len(merged)} posts)")
            else:
                import pandas as pd
                print(f"  Using existing {DATA} ({len(pd.read_csv(DATA))} posts)")

        else:
            # ─────────────────────────────────────────────────────────────
            # LIVE SCRAPING via Apify — only runs if no --skip-scrape flag.
            # Requires APIFY_TOKEN env var. Uses compute-unit actor (not
            # pay-per-result) to stay within the $5 free monthly credit.
            # ─────────────────────────────────────────────────────────────
            run(
                "python 01_data_collection.py --mode treatment",
                "Scrape treatment posts (AI/Grok hashtags) via snscrape (free)"
            )
            run(
                "python 01_data_collection.py --mode control",
                "Scrape control posts (human hashtags) via snscrape (free)"
            )
            print(f"\nMerging scraped CSVs into {DATA}...")
            import pandas as pd
            csv_files = glob.glob("data/processed/*_posts.csv")
            if not csv_files:
                print("[ERROR] No CSVs in data/processed/ — scraping may have failed.")
                sys.exit(1)
            merged = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
            merged.to_csv(DATA, index=False)
            print(f"  Merged {len(csv_files)} files → {len(merged)} posts")

    # ── STEP 2: AI DETECTION ──────────────────────────────────────────────────
    # Labels each post: is it AI-generated? Uses keyword matching + EXIF
    # watermark detection + optionally CLIP zero-shot image classifier.
    clip_flag = "--use-clip" if args.use_clip else ""
    run(
        f"python 02_ai_detection.py --input {DATA} "
        f"--output data/processed/classified_posts.csv {clip_flag}".strip(),
        "Classify posts as AI-generated vs human-created"
    )

    # ── STEP 3: NETWORK ANALYSIS ──────────────────────────────────────────────
    # Builds bipartite account-hashtag network, runs community detection,
    # computes Gini coefficient to test centralization hypothesis.
    run(
        f"python 03_network_analysis.py --input {DATA}",
        "Build account-hashtag network and detect communities"
    )

    # ── STEP 4: ENGAGEMENT ANALYSIS ───────────────────────────────────────────
    # PSM balances confounders (followers, account age, media, posting time).
    # Then t-tests and OLS regression test the algorithmic amplification hypothesis.
    run(
        f"python 04_engagement_analysis.py --input {DATA}",
        "PSM matching + OLS regression for algorithmic amplification test"
    )

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
║  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<47}║
╚══════════════════════════════════════════════════════════╝
""")
