import pandas as pd
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors
from scipy.stats import ttest_ind

# Load data
df = pd.read_csv("x_ncii_master_dataset.csv")

# 1. Feature Engineering
print("Preparing data for Propensity Score Matching...")
df['account_created_at'] = pd.to_datetime(df['account_created_at'], errors='coerce')
df['tweet_created_at'] = pd.to_datetime(df['tweet_created_at'], errors='coerce')

# Calculate account age in days to control for older vs newer accounts
df['account_age_days'] = (df['tweet_created_at'] - df['account_created_at']).dt.days

# Drop incomplete rows
df = df.dropna(subset=['author_followers', 'account_age_days', 'views'])

# Target Metric: Views divided by Followers (Engagement Ratio)
df['engagement_ratio'] = df['views'] / (df['author_followers'] + 1) 

# 2. Propensity Score Matching (PSM)
# Covariates: factors influencing engagement regardless of content
X = df[['author_followers', 'account_age_days']]
X = sm.add_constant(X)
y = df['is_ai'] 

logit_model = sm.Logit(y, X).fit(disp=0)
df['propensity_score'] = logit_model.predict(X)

# 3. Nearest Neighbor Matching
treatment = df[df['is_ai'] == 1].reset_index(drop=True)
control = df[df['is_ai'] == 0].reset_index(drop=True)

nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
nn.fit(control[['propensity_score']])
distances, indices = nn.kneighbors(treatment[['propensity_score']])

matched_control = control.iloc[indices.flatten()].reset_index(drop=True)

# 4. Statistical Testing
t_stat, p_val = ttest_ind(treatment['engagement_ratio'], matched_control['engagement_ratio'])

print("\n--- ALGORITHMIC AMPLIFICATION RESULTS ---")
print(f"Matched N = {len(treatment)} pairs")
print(f"Mean Engagement Ratio (AI):    {treatment['engagement_ratio'].mean():.4f}")
print(f"Mean Engagement Ratio (Human): {matched_control['engagement_ratio'].mean():.4f}")
print(f"T-Statistic: {t_stat:.4f}")
print(f"P-Value:     {p_val:.4g}")

if p_val < 0.05:
    print("\n[+] STATISTICALLY SIGNIFICANT DIFFERENCE DETECTED.")
    if treatment['engagement_ratio'].mean() > matched_control['engagement_ratio'].mean():
        print("[!] X's algorithm amplifies AI content more than matched human content.")
else:
    print("\n[-] NO STATISTICALLY SIGNIFICANT DIFFERENCE DETECTED.")
    print("[-] AI content scales due to volume, not algorithmic favoritism.")