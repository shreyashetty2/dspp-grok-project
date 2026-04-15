"""
01_data_collection.py
=====================
Collects posts from X (Twitter) using Apify's Twitter Scraper actor.
Targets AI-NCII content via seed hashtags and account snowball sampling.

Cost management:
  - Apify free tier supports ~100 actor runs/month
  - We cap each run at maxItems=100 to stay within limits
  - Total expected dataset: ~1,500 treatment + ~1,500 control posts

Usage:
  python 01_data_collection.py --mode treatment   # Scrape AI-NCII posts
  python 01_data_collection.py --mode control     # Scrape human-explicit baseline
  python 01_data_collection.py --mode synthetic   # Generate synthetic data for testing
"""

import os
import json
import time
import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
APIFY_TOKEN = os.getenv("APIFY_TOKEN", "YOUR_APIFY_TOKEN_HERE")
ACTOR_ID    = "61RPP7dywgiy0JPD0"   # Apify's Twitter Scraper v2 (free tier compatible)
BASE_URL    = "https://api.apify.com/v2"

RAW_DIR  = Path("data/raw")
PROC_DIR = Path("data/processed")
SYNTH_DIR = Path("data/synthetic")
for d in [RAW_DIR, PROC_DIR, SYNTH_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Seed hashtags — identified through manual reconnaissance of X
# (These represent publicly documented communities; adjust as needed)
TREATMENT_HASHTAGS = [
    "#GrokAI", "#AIGenerated", "#GrokImage",
    "#AIArt", "#SyntheticMedia", "#DeepfakeAI"
]

CONTROL_HASHTAGS = [
    "#photography", "#portrait", "#nsfw",
    "#artistsontwitter", "#digitalart"
]

# Seed accounts (placeholder — replace with accounts identified in manual recon)
SEED_ACCOUNTS = [
    # Add real seed accounts identified during manual reconnaissance
    # Example format: "username1", "username2"
]

# ── Apify Scraper ────────────────────────────────────────────────────────────
def run_apify_actor(query: str, max_items: int = 100, mode: str = "hashtag") -> list[dict]:
    """
    Trigger an Apify actor run and wait for results.
    Returns list of raw tweet objects.
    """
    if APIFY_TOKEN == "YOUR_APIFY_TOKEN_HERE":
        print("[WARNING] No APIFY_TOKEN set. Returning empty list.")
        return []

    input_data = {
        "searchTerms": [query],
        "maxItems": max_items,
        "sort": "Latest",
        "tweetLanguage": "en",
        "start": (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d"),
    }

    # Start actor run
    run_url = f"{BASE_URL}/acts/{ACTOR_ID}/runs?token={APIFY_TOKEN}"
    resp = requests.post(run_url, json=input_data)
    resp.raise_for_status()
    run_id = resp.json()["data"]["id"]
    print(f"  Started run {run_id} for query: {query}")

    # Poll until finished (max 5 minutes)
    status_url = f"{BASE_URL}/actor-runs/{run_id}?token={APIFY_TOKEN}"
    for _ in range(30):
        time.sleep(10)
        status = requests.get(status_url).json()["data"]["status"]
        if status == "SUCCEEDED":
            break
        elif status in ("FAILED", "ABORTED"):
            print(f"  [ERROR] Run {run_id} failed.")
            return []

    # Fetch results
    dataset_id = requests.get(status_url).json()["data"]["defaultDatasetId"]
    items_url = f"{BASE_URL}/datasets/{dataset_id}/items?token={APIFY_TOKEN}&format=json"
    items = requests.get(items_url).json()
    print(f"  Retrieved {len(items)} items.")
    return items


def scrape_hashtags(hashtags: list[str], mode: str, max_per_tag: int = 200) -> pd.DataFrame:
    """Scrape posts for each hashtag and combine into a DataFrame."""
    all_posts = []
    for tag in hashtags:
        print(f"\nScraping hashtag: {tag}")
        raw = run_apify_actor(tag, max_items=max_per_tag, mode=mode)
        # Save raw JSON
        fname = tag.replace("#", "").replace(" ", "_")
        with open(RAW_DIR / f"{mode}_{fname}.json", "w") as f:
            json.dump(raw, f)
        all_posts.extend(raw)

    if not all_posts:
        return pd.DataFrame()

    return parse_posts(all_posts, content_type=mode)


def scrape_accounts(accounts: list[str], mode: str, max_per_account: int = 100) -> pd.DataFrame:
    """Snowball: scrape timelines of seed accounts."""
    all_posts = []
    for account in accounts:
        print(f"\nScraping account: @{account}")
        raw = run_apify_actor(f"from:{account}", max_items=max_per_account, mode=mode)
        all_posts.extend(raw)

    if not all_posts:
        return pd.DataFrame()

    return parse_posts(all_posts, content_type=mode)


# ── Post Parser ───────────────────────────────────────────────────────────────
def parse_posts(raw_posts: list[dict], content_type: str) -> pd.DataFrame:
    """
    Normalize raw Apify tweet objects into a clean DataFrame.
    Key fields extracted for analysis.
    """
    records = []
    for p in raw_posts:
        try:
            record = {
                # Identifiers
                "post_id":         p.get("id", ""),
                "author_id":       p.get("author", {}).get("id", ""),
                "author_username": p.get("author", {}).get("userName", ""),
                "author_followers":p.get("author", {}).get("followers", 0),
                "author_created":  p.get("author", {}).get("created", ""),

                # Content
                "text":            p.get("text", ""),
                "created_at":      p.get("createdAt", ""),
                "lang":            p.get("lang", ""),
                "has_media":       len(p.get("media", [])) > 0,
                "media_types":     ",".join([m.get("type","") for m in p.get("media", [])]),
                "media_urls":      ",".join([m.get("url","") for m in p.get("media", []) if m.get("url")]),

                # Engagement metrics
                "likes":           p.get("likeCount", 0),
                "retweets":        p.get("retweetCount", 0),
                "replies":         p.get("replyCount", 0),
                "views":           p.get("viewCount", 0),
                "bookmarks":       p.get("bookmarkCount", 0),
                "quotes":          p.get("quoteCount", 0),

                # Discoverability
                "hashtags":        ",".join([h.get("text","") for h in p.get("entities", {}).get("hashtags", [])]),

                # Labels
                "content_type":    content_type,   # "treatment" or "control"
                "ai_generated":    None,            # Filled in 02_ai_detection.py
            }
            records.append(record)
        except Exception as e:
            print(f"  [WARN] Could not parse post: {e}")
            continue

    df = pd.DataFrame(records)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["engagement_total"] = df["likes"] + df["retweets"] + df["replies"] + df["quotes"]
    df["engagement_rate"] = df.apply(
        lambda r: r["engagement_total"] / r["views"] if r["views"] > 0 else 0, axis=1
    )
    return df


# ── Synthetic Data Generator (for testing when no API key) ────────────────────
def generate_synthetic_data(n_treatment: int = 800, n_control: int = 800, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic dataset for pipeline testing.
    Based on published benchmarks for X engagement rates.

    Treatment (AI-NCII) posts are simulated with:
      - Higher view counts (algorithmic amplification hypothesis)
      - Shorter author account age (newer accounts)
      - Concentrated hashtag usage
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    def make_posts(n, content_type, view_multiplier=1.0, follower_mean=500):
        posts = {
            "post_id":          [f"{content_type}_{i:06d}" for i in range(n)],
            "author_id":        [f"user_{rng.integers(1000, 9999)}" for _ in range(n)],
            "author_username":  [f"user_{rng.integers(10000,99999)}" for _ in range(n)],
            "author_followers": rng.negative_binomial(2, 0.004, n).clip(10, 500000),
            "author_account_age_days": rng.integers(30, 2000, n),
            "has_media":        rng.choice([True, False], n, p=[0.85, 0.15]),
            "views":            (rng.negative_binomial(5, 0.001, n) * view_multiplier).astype(int).clip(0),
            "likes":            None,
            "retweets":         None,
            "replies":          None,
            "bookmarks":        None,
            "quotes":           None,
            "content_type":     [content_type] * n,
            "ai_generated":     [content_type == "treatment"] * n,
            "lang":             ["en"] * n,
        }
        # Derive engagement from views with realistic ratios
        views = posts["views"]
        posts["likes"]     = (views * rng.uniform(0.02, 0.08, n)).astype(int)
        posts["retweets"]  = (views * rng.uniform(0.005, 0.02, n)).astype(int)
        posts["replies"]   = (views * rng.uniform(0.002, 0.01, n)).astype(int)
        posts["bookmarks"] = (views * rng.uniform(0.003, 0.015, n)).astype(int)
        posts["quotes"]    = (views * rng.uniform(0.001, 0.005, n)).astype(int)

        df = pd.DataFrame(posts)
        df["engagement_total"] = df["likes"] + df["retweets"] + df["replies"] + df["quotes"]
        df["engagement_rate"]  = df.apply(
            lambda r: r["engagement_total"] / r["views"] if r["views"] > 0 else 0, axis=1
        )
        # Hashtag simulation
        treatment_tags = ["GrokAI","AIGenerated","GrokImage","SyntheticMedia","AIArt"]
        control_tags   = ["photography","portrait","digitalart","artistsontwitter"]
        tags = treatment_tags if content_type == "treatment" else control_tags
        posts_df_tags  = [",".join(rng.choice(tags, rng.integers(1, 4))) for _ in range(n)]
        df["hashtags"] = posts_df_tags
        df["created_at"] = pd.to_datetime("2025-01-01") + pd.to_timedelta(
            rng.integers(0, 120, n), unit="D"
        )
        return df

    # Treatment: AI posts get ~2.3x more views (our hypothesis)
    treatment = make_posts(n_treatment, "treatment", view_multiplier=2.3, follower_mean=400)
    control   = make_posts(n_control,   "control",   view_multiplier=1.0, follower_mean=700)
    df = pd.concat([treatment, control], ignore_index=True).sample(frac=1, random_state=seed)

    out_path = SYNTH_DIR / "synthetic_posts.csv"
    df.to_csv(out_path, index=False)
    print(f"[Synthetic] Generated {len(df)} posts → {out_path}")
    return df


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["treatment", "control", "synthetic", "all"],
                        default="synthetic")
    parser.add_argument("--max-per-tag", type=int, default=200)
    args = parser.parse_args()

    if args.mode == "synthetic" or args.mode == "all":
        print("=== Generating synthetic dataset ===")
        df = generate_synthetic_data()
        print(df.describe())

    if args.mode in ("treatment", "all"):
        print("\n=== Scraping treatment (AI-NCII) posts ===")
        df_t = scrape_hashtags(TREATMENT_HASHTAGS, "treatment", args.max_per_tag)
        if not df_t.empty:
            if SEED_ACCOUNTS:
                df_accounts = scrape_accounts(SEED_ACCOUNTS, "treatment")
                df_t = pd.concat([df_t, df_accounts], ignore_index=True)
            df_t.to_csv(PROC_DIR / "treatment_posts.csv", index=False)
            print(f"Saved {len(df_t)} treatment posts.")

    if args.mode in ("control", "all"):
        print("\n=== Scraping control (human explicit) posts ===")
        df_c = scrape_hashtags(CONTROL_HASHTAGS, "control", args.max_per_tag)
        if not df_c.empty:
            df_c.to_csv(PROC_DIR / "control_posts.csv", index=False)
            print(f"Saved {len(df_c)} control posts.")
