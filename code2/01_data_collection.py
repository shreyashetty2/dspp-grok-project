import pandas as pd
import json

def process_apify_json(filepath, is_ai_flag):
    print(f"Processing {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            dataset_items = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Could not find '{filepath}'. Make sure it is in the same folder as this script!")
        return pd.DataFrame()

    data = []
    for item in dataset_items:
        # Extract metadata just like the API would have done
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
    print("Initializing Phase 1: Local Data Formatting...")
    
    # Process the JSON files you downloaded from the Apify Web Console
    df_ai = process_apify_json('ai_results.json', 1)
    df_control = process_apify_json('control_results.json', 0)
    
    if not df_ai.empty and not df_control.empty:
        # Combine the two datasets
        df_master = pd.concat([df_ai, df_control], ignore_index=True)
        
        # Drop duplicates
        df_master = df_master.drop_duplicates(subset=['tweet_id'])
        
        # Save the master CSV for Scripts 02 and 03
        df_master.to_csv("x_ncii_master_dataset.csv", index=False)
        print(f"Success! Master dataset generated. Total unique records saved: {len(df_master)}")
        print("You can now safely run 02_network_analysis.py!")