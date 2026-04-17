import pandas as pd
import networkx as nx
import community.community_louvain as community_louvain 
import re

# Load the dataset
df = pd.read_csv("x_ncii_master_dataset.csv")

# Extract hashtags from text
df['text'] = df['text'].fillna("")
df['hashtags'] = df['text'].apply(lambda x: re.findall(r"#(\w+)", str(x).lower()))

# 1. Initialize Bipartite Graph
B = nx.Graph()

print("Building Network Graphs...")
for index, row in df.iterrows():
    user = row['author_id']
    if pd.isna(user): continue
    
    B.add_node(user, bipartite=0) # Node class 0: Users
    
    for tag in row['hashtags']:
        B.add_node(tag, bipartite=1) # Node class 1: Hashtags
        B.add_edge(user, tag)

# 2. Project to User-to-User Graph (connected if they use same tags)
user_nodes = {n for n, d in B.nodes(data=True) if d["bipartite"] == 0}
user_graph = nx.bipartite.projected_graph(B, user_nodes)

# 3. Identify Super-Spreaders (Degree Centrality)
centrality = nx.degree_centrality(user_graph)
top_spreaders = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]

print("\n--- TOP 10 'SUPER-SPREADERS' ---")
for spreader, score in top_spreaders:
    print(f"User: @{spreader} | Centrality Score: {score:.4f}")

# 4. Community Detection (Louvain Method)
partition = community_louvain.best_partition(user_graph)
community_counts = pd.Series(partition.values()).value_counts()

print(f"\n--- COMMUNITY DETECTION ---")
print(f"Discovered {len(community_counts)} distinct distributor communities.")
print("Top 5 Largest Communities by Member Count:")
print(community_counts.head(5))