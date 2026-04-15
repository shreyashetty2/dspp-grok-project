"""
05_run_pipeline.py
==================
Master script: runs the full analysis pipeline end-to-end.

Modes:
  --mode synthetic   → Use generated synthetic data (no API key needed)
  --mode live        → Use real Apify-scraped data (requires APIFY_TOKEN)

Usage:
  python 05_run_pipeline.py --mode synthetic
  python 05_run_pipeline.py --mode live
"""

import subprocess
import sys
import argparse
from pathlib import Path

def run(cmd: str):
    print(f"\n{'='*60}")
    print(f"▶  {cmd}")
    print('='*60)
    result = subprocess.run(cmd, shell=True, check=True)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic", "live"], default="synthetic")
    parser.add_argument("--use-clip", action="store_true")
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════╗
║  AI-NCII on X: Full Analysis Pipeline                   ║
║  Data Science & Public Policy — Spring 2026             ║
╚══════════════════════════════════════════════════════════╝
""")

    if args.mode == "synthetic":
        # Step 1: Generate synthetic data
        run("python 01_data_collection.py --mode synthetic")
        DATA = "data/synthetic/synthetic_posts.csv"
    else:
        # Step 1: Scrape live data
        run("python 01_data_collection.py --mode treatment")
        run("python 01_data_collection.py --mode control")
        DATA = "data/processed/all_posts.csv"
        # Merge treatment + control
        import pandas as pd
        import glob
        dfs = [pd.read_csv(f) for f in glob.glob("data/processed/*_posts.csv")]
        if dfs:
            pd.concat(dfs, ignore_index=True).to_csv(DATA, index=False)

    # Step 2: AI Detection
    clip_flag = "--use-clip" if args.use_clip else ""
    run(f"python 02_ai_detection.py --input {DATA} --output data/processed/classified_posts.csv {clip_flag}")

    # Step 3: Network Analysis
    run(f"python 03_network_analysis.py --input {DATA}")

    # Step 4: Engagement Analysis
    run(f"python 04_engagement_analysis.py --input {DATA}")

    print("""
╔══════════════════════════════════════════════════════════╗
║  ✓ Pipeline complete!                                   ║
║                                                          ║
║  Outputs:                                                ║
║    outputs/figures/  → All charts                       ║
║    outputs/tables/   → CSV result tables                ║
╚══════════════════════════════════════════════════════════╝
""")
