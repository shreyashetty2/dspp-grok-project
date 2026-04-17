# Setup Guide — AI-NCII Research Pipeline

## Step 1: Get Your Apify API Key (Free)

1. Go to **https://apify.com** and click **"Sign up for free"**
   - No credit card required
   - You get $5 in free monthly credits (resets every month)

2. Once logged in, go to:
   **https://console.apify.com/account/integrations**

3. Under **"Personal API tokens"**, click **"+ Add token"**
   - Give it a name like "NCII Research"
   - Copy the token — it looks like: `apify_api_aBcDeFgHiJ...`

4. **Keep this token secret** — treat it like a password.
   Never paste it into any Python file or commit it to GitHub.

---

## Step 2: Set the Token in Your Terminal (Every Session)

Open your terminal and run:

```bash
export APIFY_TOKEN="apify_api_xxxxxxxxxxxxxxxxxxxx"
```

Replace the value with your actual token from Step 1.

**Verify it is set:**
```bash
echo $APIFY_TOKEN
# Should print your token, not an empty line
```

---

## Step 3: Make It Permanent (Optional)

If you don't want to re-export every time you open a terminal:

**Mac (zsh):**
```bash
echo 'export APIFY_TOKEN="apify_api_xxxxxxxxxxxxxxxxxxxx"' >> ~/.zshrc
source ~/.zshrc
```

**Mac (bash, older systems):**
```bash
echo 'export APIFY_TOKEN="apify_api_xxxxxxxxxxxxxxxxxxxx"' >> ~/.bash_profile
source ~/.bash_profile
```

**Windows PowerShell (permanent):**
```powershell
[Environment]::SetEnvironmentVariable("APIFY_TOKEN", "apify_api_xxx...", "User")
```

---

## Step 4: Add a .gitignore to Protect Secrets

Run this once in your project folder:

```bash
cd /path/to/your/DSPP/folder
cat .gitignore   # check if it exists already
```

Make sure `.gitignore` contains these lines (add if missing):
```
.env
*.env
data/raw/
data/processed/
outputs/
__pycache__/
*.pyc
.DS_Store
```

**Why data/raw/ and data/processed/ are gitignored:**
These folders contain scraped post data. Committing scraped X data
raises ethical and legal concerns, and the files are large anyway.
Only commit the code, not the data.

---

## Step 5: Run the Pipeline

**Test with synthetic data first (free, no API key needed):**
```bash
cd /path/to/DSPP
python3 05_run_pipeline.py --mode synthetic | tee "outputs/pipeline_$(date +%Y%m%d_%H%M%S).log"
```

**Run with real Apify data (requires token from Step 2):**
```bash
export APIFY_TOKEN="apify_api_xxxxxxxxxxxxxxxxxxxx"   # if not already set
python3 05_run_pipeline.py --mode live | tee "outputs/pipeline_$(date +%Y%m%d_%H%M%S).log"
```

**Your log command is correct** — `tee` writes output to both terminal
and a timestamped log file simultaneously. Perfect for a project like this.

---

## Step 6: Monitor Your Apify Usage

Free tier: $5/month, resets on billing date.
Check your remaining credits at any time:
**https://console.apify.com/billing**

For this project (2,000 posts, treatment + control):
- Estimated cost: ~$0.85 total
- Well within the $5 free monthly credit

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `APIFY_TOKEN not set` | Run `export APIFY_TOKEN="..."` in your terminal |
| `python: command not found` | Use `python3` instead (fixed in 05_run_pipeline.py) |
| `ModuleNotFoundError` | Run `pip3 install requests pandas numpy networkx scikit-learn matplotlib scipy` |
| Apify run FAILED | Check https://console.apify.com/actors/runs for error details |
| Empty results | X may be rate-limiting Apify; wait 1 hour and retry |
