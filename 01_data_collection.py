"""
01_data_collection.py - AI-NCII Research Project
=================================================
PURPOSE:
  Collects posts from X (Twitter) using Apify cloud scraping,
  saves raw JSON and cleaned CSVs for downstream analysis.

WHAT THIS SCRIPT DOES (plain English):
  1. Reads your Apify API key from an ENVIRONMENT VARIABLE (never hardcoded).
  2. Sends search queries to Apify, which runs a cloud browser to scrape
     public X posts matching our hashtags — same as searching X manually
     but automated and exported to JSON.
  3. Normalises raw JSON into a clean DataFrame (consistent column names,
     engagement counts, author info, etc.).
  4. Saves two labelled datasets:
       Treatment: posts with AI-generation markers (#GrokAI, #AIGenerated...)
       Control:   posts from similar human communities (#photography, #portrait...)
     These two groups are compared in 04_engagement_analysis.py.
  5. --mode synthetic: skips Apify entirely, generates statistically
     calibrated fake data so the pipeline can be tested for free.

HOW TO SET YOUR API KEY SAFELY (never paste it into this file):
  Mac/Linux terminal:
    export APIFY_TOKEN="apify_api_xxxxxxxxxxxxxxxxxxxx"

  Windows PowerShell:
    $env:APIFY_TOKEN="apify_api_xxxxxxxxxxxxxxxxxxxx"

  The key lives only in your shell session and is never written to disk,
  so it cannot accidentally end up on GitHub.

  To persist across sessions (Mac/Linux):
    echo 'export APIFY_TOKEN="apify_api_xxxxxxxxxxxxxxxxxxxx"' >> ~/.zshrc
    source ~/.zshrc

USAGE:
  python3 01_data_collection.py --mode synthetic   # free, no API key needed
  python3 01_data_collection.py --mode treatment   # needs APIFY_TOKEN env var
  python3 01_data_collection.py --mode control     # needs APIFY_TOKEN env var
  python3 01_data_collection.py --mode all         # treatment + control
"""

import os
import sys
import json
import time
import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# ── Folder setup ─────────────────────────────────────────────────────────────
# Three separate directories keep raw API data, cleaned CSVs, and synthetic
# data apart — so re-running cleaning never requires re-scraping.
RAW_DIR   = Path("data/raw")        # untouched JSON from Apify
PROC_DIR  = Path("data/processed")  # cleaned CSVs used by analysis scripts
SYNTH_DIR = Path("data/synthetic")  # synthetic test data
for d in [RAW_DIR, PROC_DIR, SYNTH_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Apify config ──────────────────────────────────────────────────────────────
# SECURITY: Token read from environment variable, NEVER hardcoded.
# os.getenv returns "" (empty string) if variable is not set;
# we catch this in check_token() before any API call is made.
APIFY_TOKEN = os.getenv("APIFY_TOKEN", "")
BASE_URL    = "https://api.apify.com/v2"

# Actor ID = Apify's identifier for the Twitter/X Scraper v2 cloud job.
# This actor scrapes public X search results without needing an official
# X API key (which now costs $100+/month for basic access).
ACTOR_ID = "61RPP7dywgiy0JPD0"

# ── Seed hashtags ─────────────────────────────────────────────────────────────
# TREATMENT: tags strongly associated with AI-generated content on X.
# Identified via manual reconnaissance of communities documented in
# investigative reporting on Grok NCII proliferation (early 2025).
TREATMENT_HASHTAGS = [
    "#GrokAI",         # direct Grok attribution
    "#AIGenerated",    # generic AI generation label
    "#GrokImage",      # Grok image-specific tag
    "#AIArt",          # AI art community (high overlap with NCII)
    "#SyntheticMedia", # broader synthetic media tag
    "#DeepfakeAI",     # deepfake-adjacent community
]

# CONTROL: human-generated content communities using similar hashtag-driven
# discoverability strategies. Matched on platform affordance (visual,
# hashtag-heavy) so the ONLY meaningful difference is AI-generation status.
# This is what allows PSM to isolate AI generation's effect on reach.
CONTROL_HASHTAGS = [
    "#photography",       # large legitimate photography community
    "#portrait",          # closest visual match to AI portrait content
    "#nsfw",              # adult content human-generated baseline
    "#artistsontwitter",  # human artist community
    "#digitalart",        # human digital art (visually similar to AI art)
]

# Seed accounts for snowball sampling — fill in after manual recon.
# Format: plain usernames without @. Example: ["user1", "user2"]
SEED_ACCOUNTS = []


def check_token():
    """
    Exit with a helpful message if APIFY_TOKEN is not set.
    Called before any live API request — avoids cryptic HTTP 401 errors.
    """
    if not APIFY_TOKEN:
        print("\n" + "="*60)
        print("ERROR: APIFY_TOKEN environment variable is not set.")
        print("="*60)
        print("\nFix: run this in your terminal before the pipeline:")
        print('  export APIFY_TOKEN="apify_api_xxxxxxxxxxxxxxxxxxxx"')
        print("\nGet your free token (no credit card needed):")
        print("  https://console.apify.com/account/integrations\n")
        sys.exit(1)


def run_apify_actor(query: str, max_items: int = 200) -> list:
    """
    HOW APIFY SCRAPING WORKS:
      1. We POST a job request (with our search query) to the Apify REST API.
      2. Apify launches a cloud browser, goes to x.com/search?q=<query>,
         scrolls through results page by page, saves each post as JSON.
      3. We poll every 10 seconds until the job finishes (status=SUCCEEDED).
      4. We download the resulting dataset and return it as a list of dicts.

    This is identical to searching X manually and copy-pasting results,
    but automated, structured, and runs in the cloud — no local browser needed.

    sort="Latest" gets chronological results (not "Top"/viral posts) so we
    get a representative random sample rather than only outliers.
    """
    check_token()

    input_data = {
        "searchTerms": [query],
        "maxItems": max_items,
        "sort": "Latest",        # chronological, not popularity-ranked
        "tweetLanguage": "en",
        # Limit to past 60 days — keeps data current and costs low
        "start": (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d"),
    }

    # Start the Apify actor run
    run_url = f"{BASE_URL}/acts/{ACTOR_ID}/runs?token={APIFY_TOKEN}"
    try:
        resp = requests.post(run_url, json=input_data, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [ERROR] Could not start Apify run for '{query}': {e}")
        return []

    run_id = resp.json()["data"]["id"]
    print(f"  [Apify] Run started: {run_id} | query='{query}' | max={max_items}")

    # Poll status every 10 seconds (max 5 minutes = 30 polls)
    # Apify runs asynchronously — typical completion: 1-3 minutes
    status_url  = f"{BASE_URL}/actor-runs/{run_id}?token={APIFY_TOKEN}"
    status_resp = None
    for attempt in range(30):
        time.sleep(10)
        try:
            status_resp = requests.get(status_url, timeout=15)
            status = status_resp.json()["data"]["status"]
            print(f"  [Apify] Status: {status} (poll {attempt+1}/30)")
        except requests.RequestException:
            print("  [Apify] Poll failed, retrying...")
            continue
        if status == "SUCCEEDED":
            break
        elif status in ("FAILED", "ABORTED", "TIMED-OUT"):
            print(f"  [ERROR] Run ended with status: {status}")
            return []
    else:
        print(f"  [TIMEOUT] Run {run_id} did not finish within 5 minutes.")
        return []

    # Download the dataset produced by the completed run
    dataset_id = status_resp.json()["data"]["defaultDatasetId"]
    items_url  = (f"{BASE_URL}/datasets/{dataset_id}/items"
                  f"?token={APIFY_TOKEN}&format=json&clean=true")
    try:
        items = requests.get(items_url, timeout=60).json()
        print(f"  [Apify] Downloaded {len(items)} posts for '{query}'")
        return items
    except Exception as e:
        print(f"  [ERROR] Could not download dataset: {e}")
        return []


def scrape_hashtags(hashtags: list, mode: str, max_per_tag: int = 200) -> pd.DataFrame:
    """
    WHAT IS BEING SCRAPED HERE:
      Public X posts containing each hashtag, sorted by most recent.
      One Apify run per hashtag — keeps runs small (free tier: 10 runs/day)
      and lets us resume from any point if a run fails.

    Raw JSON is saved immediately before cleaning so we can always
    re-run cleaning without re-scraping (scraping costs; cleaning is free).
    """
    all_posts = []
    for tag in hashtags:
        print(f"\n  Scraping {mode} hashtag: {tag}")
        raw_items = run_apify_actor(tag, max_items=max_per_tag)
        if raw_items:
            safe     = tag.replace("#","").replace(" ","_").lower()
            raw_path = RAW_DIR / f"{mode}_{safe}_{datetime.now():%Y%m%d}.json"
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(raw_items, f, ensure_ascii=False, indent=2)
            print(f"  [Saved raw JSON] → {raw_path}")
            all_posts.extend(raw_items)
        time.sleep(5)   # pause between runs to avoid rate limits

    return parse_posts(all_posts, mode) if all_posts else pd.DataFrame()


def scrape_accounts(accounts: list, mode: str, max_per_account: int = 100) -> pd.DataFrame:
    """
    WHAT IS BEING SCRAPED HERE (snowball sampling):
      Full timelines of identified seed accounts using X's "from:<user>"
      search operator. This captures ALL hashtags an account uses — not just
      those that appeared in our hashtag searches — giving richer edge data
      for the bipartite network in 03_network_analysis.py.
    """
    all_posts = []
    for account in accounts:
        print(f"\n  Scraping account timeline: @{account}")
        raw_items = run_apify_actor(f"from:{account}", max_items=max_per_account)
        all_posts.extend(raw_items)
        time.sleep(5)
    return parse_posts(all_posts, mode) if all_posts else pd.DataFrame()


def parse_posts(raw_posts: list, content_type: str) -> pd.DataFrame:
    """
    WHAT THIS DOES:
      Flattens Apify's inconsistently-structured JSON into a clean DataFrame.
      Field names vary between Apify actor versions, so we try multiple
      alternatives for each field (e.g. "likeCount" vs "favorite_count").

    WHY EACH FIELD MATTERS FOR OUR ANALYSIS:

      author_followers  — PSM CONFOUNDER: larger accounts get more views
                          naturally. Must control for this before comparing
                          AI vs human engagement.

      views             — PRIMARY OUTCOME: algorithmic reach — how many
                          accounts X surfaced this post to. If AI posts get
                          more views WITHOUT more engagement (likes/RT),
                          that is direct evidence of algorithmic amplification
                          rather than user preference.

      engagement_rate   — TERTIARY OUTCOME: engagement / views. If this is
                          NOT higher for AI posts but views ARE, the algorithm
                          is surfacing AI content to users who don't seek it.

      hashtags          — NETWORK EDGES: used to build the bipartite graph
                          (account <-> hashtag) in 03_network_analysis.py.

      media_urls        — AI DETECTION INPUT: image URLs passed to
                          02_ai_detection.py for CLIP + EXIF watermark check.

      ai_generated      — LABEL: None here, filled by 02_ai_detection.py.
    """
    records = []
    for post in raw_posts:
        try:
            author       = post.get("author", {})
            followers    = int(author.get("followers", author.get("followersCount", 0)) or 0)

            media_list   = post.get("media") or post.get("extendedEntities", {}).get("media", [])
            media_types  = ",".join(m.get("type","") for m in media_list)
            media_urls   = ",".join(
                m.get("url", m.get("media_url_https",""))
                for m in media_list if m.get("url") or m.get("media_url_https")
            )

            likes    = int(post.get("likeCount",    post.get("favorite_count",  0)) or 0)
            retweets = int(post.get("retweetCount", post.get("retweet_count",   0)) or 0)
            replies  = int(post.get("replyCount",   post.get("reply_count",     0)) or 0)
            views    = int(post.get("viewCount",    post.get("views",           0)) or 0)
            bookmarks= int(post.get("bookmarkCount", 0) or 0)
            quotes   = int(post.get("quoteCount",   0) or 0)

            hash_list = post.get("entities", {}).get("hashtags", [])
            hashtags  = ",".join(
                h.get("text", h.get("tag","")).lower().lstrip("#")
                for h in hash_list if h.get("text") or h.get("tag")
            )

            records.append({
                "post_id":          str(post.get("id", post.get("id_str",""))),
                "author_id":        str(author.get("id","")),
                "author_username":  str(author.get("userName", author.get("username",""))),
                "author_followers": followers,
                "author_created":   str(author.get("created", author.get("createdAt",""))),
                "text":             str(post.get("text", post.get("full_text",""))),
                "created_at":       str(post.get("createdAt", post.get("created_at",""))),
                "lang":             str(post.get("lang","en")),
                "has_media":        len(media_list) > 0,
                "media_types":      media_types,
                "media_urls":       media_urls,
                "likes":            likes,
                "retweets":         retweets,
                "replies":          replies,
                "views":            views,
                "bookmarks":        bookmarks,
                "quotes":           quotes,
                "hashtags":         hashtags,
                "content_type":     content_type,
                "ai_generated":     None,   # filled by 02_ai_detection.py
            })
        except Exception as e:
            print(f"  [WARN] Skipped malformed post: {e}")

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["created_at"]       = pd.to_datetime(df["created_at"], errors="coerce")
    df["engagement_total"] = df["likes"] + df["retweets"] + df["replies"] + df["quotes"]
    df["engagement_rate"]  = df.apply(
        lambda r: r["engagement_total"] / r["views"] if r["views"] > 0 else 0, axis=1
    )

    # Deduplicate posts that appeared under multiple hashtag searches
    before = len(df)
    df = df.drop_duplicates("post_id").reset_index(drop=True)
    if len(df) < before:
        print(f"  [Dedup] Removed {before-len(df)} duplicates.")
    return df


def generate_synthetic_data(n_treatment: int = 800, n_control: int = 800,
                             seed: int = 42) -> pd.DataFrame:
    """
    WHY SYNTHETIC DATA (not dummy data):
      Statistically calibrated to match published X engagement benchmarks:
        - Views: negative binomial (standard overdispersed count model —
          Vosoughi et al., Science 2018)
        - Engagement ratios: likes 2-8% of views, retweets 0.5-2%
          (Pew Research 2023, Social Insider 2023)
        - Follower counts: negative binomial, empirical mean ~500

    CRITICAL: The 2.3x view multiplier for treatment is our HYPOTHESIS
    baked in for pipeline testing. The real effect size comes from Apify data.
    State this clearly when presenting results.
    """
    import numpy as np
    rng = np.random.default_rng(seed)   # fixed seed = reproducible results

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
    print(f"[Synthetic] Generated {len(df)} posts → {out}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["treatment","control","synthetic","all"],
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
        print("\n=== Scraping TREATMENT posts (AI hashtags via Apify) ===")
        df_t = scrape_hashtags(TREATMENT_HASHTAGS, "treatment", args.max_per_tag)
        if SEED_ACCOUNTS:
            df_acct = scrape_accounts(SEED_ACCOUNTS, "treatment")
            df_t = pd.concat([df_t, df_acct]).drop_duplicates("post_id")
        if not df_t.empty:
            out = PROC_DIR / "treatment_posts.csv"
            df_t.to_csv(out, index=False)
            print(f"[Saved] {len(df_t)} treatment posts → {out}")

    if args.mode in ("control", "all"):
        print("\n=== Scraping CONTROL posts (human hashtags via Apify) ===")
        df_c = scrape_hashtags(CONTROL_HASHTAGS, "control", args.max_per_tag)
        if not df_c.empty:
            out = PROC_DIR / "control_posts.csv"
            df_c.to_csv(out, index=False)
            print(f"[Saved] {len(df_c)} control posts → {out}")

    print(f"\n{'='*60}")
    print("  01_data_collection.py complete.")
    print(f"{'='*60}\n")
