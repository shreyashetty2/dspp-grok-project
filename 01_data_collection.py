"""
01_data_collection.py
=====================
Collects posts from X (Twitter) using snscrape — a FREE, no-API-key-needed
Python library that scrapes X search results directly from your machine.

WHY snscrape instead of Apify:
  Every Apify Twitter actor charges per result ($0.20-$0.40/1K tweets).
  There is no free compute-unit-only Twitter scraper on Apify.
  snscrape scrapes X the same way a browser would — completely free.

INSTALL:
  pip3 install snscrape pandas

USAGE:
  python3 01_data_collection.py --mode synthetic    # fake data, instant, free
  python3 01_data_collection.py --mode treatment    # scrape AI hashtags (free)
  python3 01_data_collection.py --mode control      # scrape human hashtags (free)
  python3 01_data_collection.py --mode all          # both treatment + control

NOTES:
  - snscrape works by mimicking a browser — no API key, no cost
  - X occasionally rate-limits aggressive scraping; --max-per-tag 200 is safe
  - If you get 0 results, wait 30 min and retry (X rate limit, not a code bug)
  - Results saved to data/processed/ as CSVs
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# ── Folder setup ──────────────────────────────────────────────────────────────
RAW_DIR   = Path("data/raw")
PROC_DIR  = Path("data/processed")
SYNTH_DIR = Path("data/synthetic")
for d in [RAW_DIR, PROC_DIR, SYNTH_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Seed hashtags ─────────────────────────────────────────────────────────────
# TREATMENT: AI-generation markers (our research target)
TREATMENT_HASHTAGS = [
    "#GrokAI", "#AIGenerated", "#GrokImage",
    "#AIArt", "#SyntheticMedia", "#DeepfakeAI",
]

# CONTROL: human creative communities (baseline comparison)
CONTROL_HASHTAGS = [
    "#photography", "#portrait", "#nsfw",
    "#artistsontwitter", "#digitalart",
]

# ── snscrape scraping functions ───────────────────────────────────────────────

def check_snscrape():
    """Check snscrape is installed; give a clear install message if not."""
    try:
        import snscrape.modules.twitter as sntwitter
        return sntwitter
    except ImportError:
        print("\n[ERROR] snscrape is not installed.")
        print("  Fix: pip3 install snscrape")
        print("  Then re-run this script.\n")
        sys.exit(1)


def scrape_hashtag(tag: str, max_results: int, days_back: int = 60):
    """
    Scrape posts for one hashtag using snscrape.

    HOW IT WORKS:
      snscrape builds an X advanced search URL (same as x.com/search?q=...)
      and paginates through results, yielding tweet objects one by one.
      No API key needed — it mimics what a browser does.

    QUERY FORMAT:
      "#GrokAI since:2025-02-16 until:2026-04-17 lang:en"
      This matches X's own advanced search syntax exactly.
    """
    sntwitter = check_snscrape()

    since = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    until = datetime.now().strftime("%Y-%m-%d")
    # lang:en filters to English posts; -filter:retweets excludes retweets
    # so we only get original posts (cleaner for engagement analysis)
    query = f"{tag} lang:en -filter:retweets since:{since} until:{until}"

    print(f"  Query: {query}")
    print(f"  Collecting up to {max_results} posts...")

    posts = []
    try:
        scraper = sntwitter.TwitterSearchScraper(query)
        for i, tweet in enumerate(scraper.get_items()):
            if i >= max_results:
                break

            # Extract all fields we need for analysis
            author = tweet.user

            # Media: check for images/videos
            has_media = bool(tweet.media)
            media_types = ""
            media_urls  = ""
            if tweet.media:
                media_types = ",".join([
                    getattr(m, "type", type(m).__name__) for m in tweet.media
                ])
                media_urls = ",".join([
                    getattr(m, "thumbnailUrl", "") or getattr(m, "previewUrl", "")
                    for m in tweet.media if hasattr(m, "thumbnailUrl") or hasattr(m, "previewUrl")
                ])

            # Hashtags from tweet entities
            hashtags = ""
            if tweet.hashtags:
                hashtags = ",".join([h.lower() for h in tweet.hashtags])

            posts.append({
                "post_id":          str(tweet.id),
                "author_id":        str(author.id) if author else "",
                "author_username":  author.username if author else "",
                "author_followers": getattr(author, "followersCount", 0) or 0,
                "author_created":   str(getattr(author, "created", "")),
                "author_account_age_days": (
                    (datetime.now() - author.created.replace(tzinfo=None)).days
                    if author and getattr(author, "created", None) else 365
                ),
                "text":             tweet.content or "",
                "created_at":       str(tweet.date),
                "lang":             tweet.lang or "en",
                "has_media":        has_media,
                "media_types":      media_types,
                "media_urls":       media_urls,
                "likes":            tweet.likeCount or 0,
                "retweets":         tweet.retweetCount or 0,
                "replies":          tweet.replyCount or 0,
                "views":            tweet.viewCount or 0,
                "bookmarks":        tweet.bookmarkCount or 0,
                "quotes":           tweet.quoteCount or 0,
                "hashtags":         hashtags,
                "content_type":     None,  # set by caller
                "ai_generated":     None,  # set by 02_ai_detection.py
            })

            if (i + 1) % 50 == 0:
                print(f"    Collected {i+1} posts...")

    except Exception as e:
        print(f"  [WARN] snscrape error for '{tag}': {e}")
        print("  This usually means X is rate-limiting. Wait 30 min and retry.")

    print(f"  Collected {len(posts)} posts for {tag}")
    return posts


def scrape_hashtags(hashtags: list, content_type: str,
                    max_per_tag: int = 200) -> pd.DataFrame:
    """
    Scrape all hashtags for one content_type group (treatment or control).
    Saves raw JSON per hashtag, then returns combined cleaned DataFrame.
    """
    all_posts = []

    for tag in hashtags:
        print(f"\n  --- Scraping: {tag} ({content_type}) ---")
        posts = scrape_hashtag(tag, max_results=max_per_tag)

        if posts:
            # Label content_type before saving raw JSON
            for p in posts:
                p["content_type"] = content_type

            # Save raw JSON — so we can re-run cleaning without re-scraping
            safe     = tag.replace("#","").replace(" ","_").lower()
            raw_path = RAW_DIR / f"{content_type}_{safe}_{datetime.now():%Y%m%d}.json"
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(posts, f, ensure_ascii=False, indent=2)
            print(f"  [Saved raw] → {raw_path}")
            all_posts.extend(posts)

    if not all_posts:
        print(f"  [WARNING] No posts collected for {content_type}.")
        return pd.DataFrame()

    df = pd.DataFrame(all_posts)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["engagement_total"] = df["likes"] + df["retweets"] + df["replies"] + df["quotes"]
    df["engagement_rate"]  = df.apply(
        lambda r: r["engagement_total"] / r["views"] if r["views"] > 0 else 0, axis=1
    )

    # Deduplicate posts appearing under multiple hashtags
    before = len(df)
    df = df.drop_duplicates("post_id").reset_index(drop=True)
    if len(df) < before:
        print(f"  [Dedup] Removed {before - len(df)} duplicates.")

    return df


# ── Synthetic data (free, instant, for testing) ───────────────────────────────

def generate_synthetic_data(n_treatment: int = 800, n_control: int = 800,
                             seed: int = 42) -> pd.DataFrame:
    """
    Generate calibrated fake data for pipeline testing.
    Distributions based on:
      - Pew Research 2023 X engagement benchmarks
      - Social Insider 2023 platform averages
      - Negative binomial for count data (Vosoughi et al., Science 2018)

    IMPORTANT: The 2.3x view multiplier for treatment is our HYPOTHESIS
    baked in for testing. Real scraped data will reveal the true effect.
    """
    rng = np.random.default_rng(seed)

    def make_posts(n, content_type, view_multiplier=1.0):
        followers    = rng.negative_binomial(2, 0.004, n).clip(10, 500_000)
        acct_ages    = rng.integers(30, 2000, n)
        views        = (rng.negative_binomial(5, 0.001, n) * view_multiplier).astype(int).clip(0)
        likes        = (views * rng.uniform(0.02, 0.08, n)).astype(int)
        retweets     = (views * rng.uniform(0.005, 0.02, n)).astype(int)
        replies      = (views * rng.uniform(0.002, 0.01, n)).astype(int)
        bookmarks    = (views * rng.uniform(0.003, 0.015, n)).astype(int)
        quotes       = (views * rng.uniform(0.001, 0.005, n)).astype(int)
        eng_total    = likes + retweets + replies + quotes
        eng_rate     = np.where(views > 0, eng_total / views, 0)

        tag_pool = (["grokai","aigenerated","grokimage","syntheticmedia","aiart"]
                    if content_type == "treatment"
                    else ["photography","portrait","digitalart","artistsontwitter"])
        hashtags = [
            ",".join(rng.choice(tag_pool, rng.integers(1,4), replace=False))
            for _ in range(n)
        ]
        created_at = pd.to_datetime("2025-01-01") + pd.to_timedelta(
            rng.integers(0, 120, n), unit="D"
        )
        return pd.DataFrame({
            "post_id":               [f"{content_type}_{i:06d}" for i in range(n)],
            "author_id":             [f"user_{rng.integers(1000,9999)}" for _ in range(n)],
            "author_username":       [f"user_{rng.integers(10000,99999)}" for _ in range(n)],
            "author_followers":      followers,
            "author_account_age_days": acct_ages,
            "author_created":        ["2022-01-01"] * n,
            "text":                  [f"Synthetic {content_type} post #{i}" for i in range(n)],
            "created_at":            created_at,
            "lang":                  ["en"] * n,
            "has_media":             rng.choice([True,False], n, p=[0.85,0.15]),
            "media_types":           ["photo"] * n,
            "media_urls":            [""] * n,
            "likes":                 likes,
            "retweets":              retweets,
            "replies":               replies,
            "views":                 views,
            "bookmarks":             bookmarks,
            "quotes":                quotes,
            "hashtags":              hashtags,
            "content_type":          [content_type] * n,
            "ai_generated":          [content_type == "treatment"] * n,
            "engagement_total":      eng_total,
            "engagement_rate":       eng_rate,
        })

    treatment = make_posts(n_treatment, "treatment", view_multiplier=2.3)
    control   = make_posts(n_control,   "control",   view_multiplier=1.0)
    df = pd.concat([treatment, control]).sample(frac=1, random_state=seed).reset_index(drop=True)
    out = SYNTH_DIR / "synthetic_posts.csv"
    df.to_csv(out, index=False)
    print(f"[Synthetic] {len(df)} posts saved → {out}")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic","treatment","control","all"],
                        default="synthetic")
    parser.add_argument("--max-per-tag", type=int, default=200,
                        help="Max posts per hashtag (default 200)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  01_data_collection.py | mode={args.mode} | {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'='*60}\n")

    if args.mode == "synthetic":
        df = generate_synthetic_data()
        print("\nEngagement by group:")
        print(df.groupby("content_type")[["views","likes","engagement_rate"]].mean().round(3))

    if args.mode in ("treatment", "all"):
        print("\n=== Scraping TREATMENT posts (snscrape, free) ===")
        df_t = scrape_hashtags(TREATMENT_HASHTAGS, "treatment", args.max_per_tag)
        if not df_t.empty:
            out = PROC_DIR / "treatment_posts.csv"
            df_t.to_csv(out, index=False)
            print(f"\n[Saved] {len(df_t)} treatment posts → {out}")

    if args.mode in ("control", "all"):
        print("\n=== Scraping CONTROL posts (snscrape, free) ===")
        df_c = scrape_hashtags(CONTROL_HASHTAGS, "control", args.max_per_tag)
        if not df_c.empty:
            out = PROC_DIR / "control_posts.csv"
            df_c.to_csv(out, index=False)
            print(f"\n[Saved] {len(df_c)} control posts → {out}")

    print(f"\n{'='*60}")
    print("  01_data_collection.py complete.")
    print(f"{'='*60}\n")
