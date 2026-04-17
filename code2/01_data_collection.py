import pandas as pd
from apify_client import ApifyClient
import time

# Initialize the ApifyClient with your API token from console.apify.com
APIFY_TOKEN = ""
client = ApifyClient(APIFY_TOKEN)

# Seed hashtags (update these based on your qualitative research)
ai_seed_tags = ["#groknsfw", "#aiart", "#grok"]
control_seed_tags = ["#nsfw", "#leaked", "#traditional_nsfw"]

def scrape_twitter_tags(tags, is_ai_flag):
    print(f"Starting scrape for tags: {tags}")
    
    # CRITICAL BUDGET CAP: maxItems set to 500 per group (1000 total)
    # This ensures your Apify cost stays under ~$1.50
    run_input = {
        "searchTerms": tags,
        "maxItems": 500, 
        "tweetLanguage": "en"
    }

    # Call the Apify pay-per-result actor
    run = client.actor("apidojo/tweet-scraper").call(run_input=run_input)
    dataset_items = client.dataset(run["defaultDatasetId"]).iterate_items()

    data = []
    for item in dataset_items:
        # Extract metadata without downloading illicit image URLs
        data.append({
            "tweet_id": item.get("id"),
            "author_id": item.get("author", {}).get("userName"),
            "author_followers": item.get("author", {}).get("followers", 0),
            "author_following": item.get("author", {}).get("following", 0),
            "account_created_at": item.get("author", {}).get("createdAt"),
            "text": item.get("text", ""),
            "views": item.get("viewCount", 0),
            "likes": item.get("likeCount", 0),
            "retweets": item.get("retweetCount", 0),
            "tweet_created_at": item.get("createdAt"),
            "is_ai": is_ai_flag # 1 for Treatment, 0 for Control
        })
    return pd.DataFrame(data)

if __name__ == "__main__":
    print("Initializing Phase 1: Data Scrape...")
    
    # Scrape treatment group (AI-generated)
    df_ai = scrape_twitter_tags(ai_seed_tags, is_ai_flag=1)
    
    # Scrape control group (Human-generated)
    df_control = scrape_twitter_tags(control_seed_tags, is_ai_flag=0)
    
    # Combine and save
    df_master = pd.concat([df_ai, df_control], ignore_index=True)
    
    # Drop exact duplicates just in case hashtags overlapped
    df_master = df_master.drop_duplicates(subset=['tweet_id'])
    df_master.to_csv("x_ncii_master_dataset.csv", index=False)
    
    print(f"Success! Data collection complete. Total unique records saved: {len(df_master)}")