"""
01_data_collection.py
=====================
Collects posts from X using Scweet — a free Python library that calls
X's internal GraphQL API (the same one x.com uses in your browser).

WHY SCWEET:
  - snscrape: broken on Python 3.13 (removed import API)
  - twint: deprecated, non-functional since X's 2024 backend changes
  - Apify: pay-per-result (cost $4.40 for near-empty results)
  - Scweet: verified working March 2026, free, no developer API key

WHAT YOU NEED:
  Just your X browser auth_token cookie — no API application needed.

HOW TO GET YOUR auth_token (2 minutes):
  1. Log into x.com in Chrome or Firefox
  2. Press F12 → Application tab → Cookies → https://x.com
  3. Find the cookie named "auth_token" → copy its Value
  4. Set it as an environment variable (keeps it off GitHub):
       export X_AUTH_TOKEN="your_token_here"          # Mac/Linux
       $env:X_AUTH_TOKEN="your_token_here"            # Windows PowerShell

INSTALL:
  pip3 install Scweet pandas numpy

USAGE:
  python3 01_data_collection.py --mode synthetic   # free, instant, no token
  python3 01_data_collection.py --mode treatment   # needs X_AUTH_TOKEN
  python3 01_data_collection.py --mode control     # needs X_AUTH_TOKEN
  python3 01_data_collection.py --mode all         # both groups
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

# ── Auth token (read from environment variable — never hardcode) ───────────────
# Set before running: export X_AUTH_TOKEN="your_token_here"
# Get it from: x.com → F12 → Application → Cookies → auth_token
X_AUTH_TOKEN = os.getenv("X_AUTH_TOKEN", "")

# ── Seed hashtags ─────────────────────────────────────────────────────────────
TREATMENT_HASHTAGS = [
    "#GrokAI", "#AIGenerated", "#GrokImage",
    "#AIArt", "#SyntheticMedia", "#DeepfakeAI",
]

CONTROL_HASHTAGS = [
    "#photography", "#portrait", "#nsfw",
    "#artistsontwitter", "#digitalart",
]


# ── Token check ───────────────────────────────────────────────────────────────

def check_token():
    if not X_AUTH_TOKEN:
        print("\n" + "="*60)
        print("ERROR: X_AUTH_TOKEN environment variable is not set.")
        print("="*60)
        print("\nHow to get it (2 minutes, no API application needed):")
        print("  1. Log into x.com in Chrome or Firefox")
        print("  2. Press F12 → Application → Cookies → https://x.com")
        print("  3. Find 'auth_token' → copy its Value")
        print("  4. In your terminal:")
        print('     export X_AUTH_TOKEN="paste_your_token_here"')
        print("\nThis is your personal browser token — treat it like a password.")
        print("Never paste it into this Python file.\n")
        sys.exit(1)


# ── Scweet scraping functions ─────────────────────────────────────────────────

def check_scweet():
    """Check Scweet is installed; give clear install message if not."""
    try:
        from Scweet.scweet import Scweet
        return Scweet
    except ImportError:
        print("\n[ERROR] Scweet is not installed.")
        print("  Fix: pip3 install Scweet")
        sys.exit(1)


def scrape_hashtag(tag: str, max_results: int, days_back: int = 60) -> list:
    """
    WHAT THIS SCRAPES:
      Public X posts containing the given hashtag, sorted by most recent,
      in English, from the past `days_back` days.

    HOW SCWEET WORKS:
      Scweet calls X's internal GraphQL search endpoint — the exact same
      API your browser uses when you search on x.com. It authenticates
      using your auth_token cookie so X treats it as your logged-in browser.
      No headless browser, no Selenium — just direct API calls.

    QUERY FORMAT:
      Scweet accepts the hashtag/keyword and date range as separate params.
      Internally it builds: "#GrokAI lang:en since:2025-02-17 until:2026-04-17"
    """
    check_token()
    ScweetClass = check_scweet()

    since = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    until = datetime.now().strftime("%Y-%m-%d")

    print(f"  Query: {tag} | since={since} until={until} | max={max_results}")

    posts = []
    try:
        # Initialise Scweet with your auth_token
        # save=False means results come back as Python objects, not written to file
        # (we handle our own file saving below for consistency)
        s = ScweetClass(auth_token=X_AUTH_TOKEN)

        # search() returns a list of tweet dicts
        # display_type="Latest" = chronological, not popularity-ranked
        results = s.search(
            word=tag,
            since=since,
            until=until,
            limit=max_results,
            display_type="Latest",
            lang="en",
            save=False,
        )

        if not results:
            print(f"  [INFO] No results for {tag} — X may be rate-limiting.")
            print("         Wait 30 min and retry, or try a broader date range.")
            return []

        print(f"  Retrieved {len(results)} posts for {tag}")

        for item in results:
            # Scweet returns dicts with these fields (field names may vary
            # by Scweet version — we try multiple fallbacks for each)
            author_name  = (item.get("UserScreenName") or
                            item.get("username") or
                            item.get("author_username") or "")
            author_id    = (item.get("UserID") or
                            item.get("user_id") or "")
            followers    = int(item.get("UserFollowerCount") or
                               item.get("followers_count") or 0)
            text         = (item.get("Tweet") or
                            item.get("text") or
                            item.get("content") or "")
            created_at   = (item.get("Timestamp") or
                            item.get("created_at") or
                            item.get("date") or "")
            likes        = int(item.get("Likes") or item.get("like_count") or 0)
            retweets     = int(item.get("Retweets") or item.get("retweet_count") or 0)
            replies      = int(item.get("Comments") or item.get("reply_count") or 0)
            views        = int(item.get("Views") or item.get("view_count") or 0)
            bookmarks    = int(item.get("Bookmarks") or item.get("bookmark_count") or 0)
            quotes       = int(item.get("Quotes") or item.get("quote_count") or 0)
            post_id      = (item.get("TweetID") or item.get("tweet_id") or
                            item.get("id") or "")
            hashtags_raw = item.get("Hashtags") or item.get("hashtags") or ""
            if isinstance(hashtags_raw, list):
                hashtags = ",".join([h.lower().lstrip("#") for h in hashtags_raw])
            else:
                hashtags = str(hashtags_raw).lower().lstrip("#")

            has_media = bool(item.get("MediaURLs") or item.get("media_urls") or
                             item.get("has_media"))
            media_urls = ""
            raw_media  = item.get("MediaURLs") or item.get("media_urls") or []
            if isinstance(raw_media, list):
                media_urls = ",".join(raw_media)
            elif isinstance(raw_media, str):
                media_urls = raw_media

            posts.append({
                "post_id":               str(post_id),
                "author_id":             str(author_id),
                "author_username":       str(author_name),
                "author_followers":      followers,
                "author_created":        "",  # not provided by Scweet
                "author_account_age_days": 365,  # default; Scweet doesn't return this
                "text":                  str(text),
                "created_at":            str(created_at),
                "lang":                  "en",
                "has_media":             has_media,
                "media_types":           "photo" if has_media else "",
                "media_urls":            media_urls,
                "likes":                 likes,
                "retweets":              retweets,
                "replies":               replies,
                "views":                 views,
                "bookmarks":             bookmarks,
                "quotes":                quotes,
                "hashtags":              hashtags,
                "content_type":          None,   # set by caller
                "ai_generated":          None,   # set by 02_ai_detection.py
            })

    except Exception as e:
        print(f"  [ERROR] Scweet failed for '{tag}': {e}")
        print("  Common causes:")
        print("    - auth_token expired → get a fresh one from your browser")
        print("    - X rate limiting → wait 30 min and retry")
        print("    - Scweet version mismatch → pip3 install --upgrade Scweet")

    return posts


def scrape_hashtags(hashtags: list, content_type: str,
                    max_per_tag: int = 200) -> pd.DataFrame:
    """
    Scrape all hashtags for one group (treatment or control).
    Saves raw JSON per hashtag before cleaning, so re-cleaning
    never requires re-scraping.
    """
    all_posts = []

    for tag in hashtags:
        print(f"\n  --- Scraping: {tag} ({content_type}) ---")
        posts = scrape_hashtag(tag, max_results=max_per_tag)

        if posts:
            for p in posts:
                p["content_type"] = content_type

            # Save raw JSON immediately
            safe     = tag.replace("#","").replace(" ","_").lower()
            raw_path = RAW_DIR / f"{content_type}_{safe}_{datetime.now():%Y%m%d}.json"
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(posts, f, ensure_ascii=False, indent=2)
            print(f"  [Saved raw JSON] → {raw_path}")
            all_posts.extend(posts)

    if not all_posts:
        print(f"\n  [WARNING] No posts collected for {content_type}.")
        return pd.DataFrame()

    df = pd.DataFrame(all_posts)
    df["created_at"]       = pd.to_datetime(df["created_at"], errors="coerce")
    df["engagement_total"] = df["likes"] + df["retweets"] + df["replies"] + df["quotes"]
    df["engagement_rate"]  = df.apply(
        lambda r: r["engagement_total"] / r["views"] if r["views"] > 0 else 0, axis=1
    )

    before = len(df)
    df = df.drop_duplicates("post_id").reset_index(drop=True)
    if len(df) < before:
        print(f"  [Dedup] Removed {before - len(df)} duplicates.")

    return df


# ── Synthetic data ─────────────────────────────────────────────────────────────

def generate_synthetic_data(n_treatment=800, n_control=800, seed=42):
    """
    Free, instant, no token needed. Use for pipeline testing before
    committing to a live scrape.

    Statistical basis:
      - Negative binomial for view counts (Vosoughi et al., Science 2018)
      - Engagement ratios from Pew Research 2023 / Social Insider 2023
      - 2.3x view multiplier for treatment = hypothesis baked in for testing
        (real multiplier revealed by live data)
    """
    rng = np.random.default_rng(seed)

    def make_posts(n, content_type, view_multiplier=1.0):
        followers = rng.negative_binomial(2, 0.004, n).clip(10, 500_000)
        acct_ages = rng.integers(30, 2000, n)
        views     = (rng.negative_binomial(5, 0.001, n) * view_multiplier).astype(int).clip(0)
        likes     = (views * rng.uniform(0.02, 0.08, n)).astype(int)
        retweets  = (views * rng.uniform(0.005, 0.02, n)).astype(int)
        replies   = (views * rng.uniform(0.002, 0.01, n)).astype(int)
        bookmarks = (views * rng.uniform(0.003, 0.015, n)).astype(int)
        quotes    = (views * rng.uniform(0.001, 0.005, n)).astype(int)
        eng_total = likes + retweets + replies + quotes
        eng_rate  = np.where(views > 0, eng_total / views, 0)
        tag_pool  = (["grokai","aigenerated","grokimage","syntheticmedia","aiart"]
                     if content_type == "treatment"
                     else ["photography","portrait","digitalart","artistsontwitter"])
        hashtags  = [",".join(rng.choice(tag_pool, rng.integers(1,4), replace=False))
                     for _ in range(n)]
        created_at = (pd.to_datetime("2025-01-01") +
                      pd.to_timedelta(rng.integers(0, 120, n), unit="D"))
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

    df = pd.concat([
        make_posts(n_treatment, "treatment", view_multiplier=2.3),
        make_posts(n_control,   "control",   view_multiplier=1.0),
    ]).sample(frac=1, random_state=seed).reset_index(drop=True)

    out = SYNTH_DIR / "synthetic_posts.csv"
    df.to_csv(out, index=False)
    print(f"[Synthetic] {len(df)} posts → {out}")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        choices=["synthetic","treatment","control","all"],
                        default="synthetic")
    parser.add_argument("--max-per-tag", type=int, default=200)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  01_data_collection.py | mode={args.mode} | {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'='*60}\n")

    if args.mode == "synthetic":
        df = generate_synthetic_data()
        print("\nEngagement by group:")
        print(df.groupby("content_type")[["views","likes","engagement_rate"]].mean().round(3))

    if args.mode in ("treatment", "all"):
        print("=== Scraping TREATMENT (AI hashtags via Scweet) ===")
        df_t = scrape_hashtags(TREATMENT_HASHTAGS, "treatment", args.max_per_tag)
        if not df_t.empty:
            out = PROC_DIR / "treatment_posts.csv"
            df_t.to_csv(out, index=False)
            print(f"\n[Saved] {len(df_t)} treatment posts → {out}")

    if args.mode in ("control", "all"):
        print("=== Scraping CONTROL (human hashtags via Scweet) ===")
        df_c = scrape_hashtags(CONTROL_HASHTAGS, "control", args.max_per_tag)
        if not df_c.empty:
            out = PROC_DIR / "control_posts.csv"
            df_c.to_csv(out, index=False)
            print(f"\n[Saved] {len(df_c)} control posts → {out}")

    print(f"\n{'='*60}")
    print("  01_data_collection.py complete.")
    print(f"{'='*60}\n")
