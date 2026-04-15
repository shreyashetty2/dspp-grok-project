"""
03_network_analysis.py
======================
Constructs and analyzes the bipartite network of:
  - Actors (accounts) ←→ Hashtags
  
Applies community detection (Louvain method) to identify whether the
AI-NCII ecosystem is centralized (super-spreaders) or decentralized.

Outputs:
  - data/processed/network_stats.csv
  - outputs/figures/network_bipartite.png
  - outputs/figures/degree_distribution.png

Usage:
  python 03_network_analysis.py --input data/processed/classified_posts.csv
  python 03_network_analysis.py --input data/synthetic/synthetic_posts.csv
"""

import argparse
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import Counter

PROC_DIR   = Path("data/processed")
FIG_DIR    = Path("outputs/figures")
TABLE_DIR  = Path("outputs/tables")
for d in [PROC_DIR, FIG_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

np.random.seed(42)


# ── Network Construction ──────────────────────────────────────────────────────
def build_bipartite_network(df: pd.DataFrame) -> nx.Graph:
    """
    Build a bipartite graph: Account nodes ←→ Hashtag nodes.
    Edge weight = number of posts linking that account to that hashtag.
    """
    B = nx.Graph()

    for _, row in df.iterrows():
        author = str(row.get("author_username", row.get("author_id", "unknown")))
        hashtags_raw = str(row.get("hashtags", ""))
        if not hashtags_raw or hashtags_raw == "nan":
            continue

        hashtags = [h.strip().lower().lstrip("#") for h in hashtags_raw.split(",") if h.strip()]

        # Add account node
        if not B.has_node(f"acct_{author}"):
            B.add_node(f"acct_{author}",
                       bipartite=0,
                       node_type="account",
                       followers=int(row.get("author_followers", 0)),
                       content_type=str(row.get("content_type", "unknown")))

        # Add hashtag nodes and edges
        for tag in hashtags:
            tag_node = f"tag_{tag}"
            if not B.has_node(tag_node):
                B.add_node(tag_node, bipartite=1, node_type="hashtag")
            if B.has_edge(f"acct_{author}", tag_node):
                B[f"acct_{author}"][tag_node]["weight"] += 1
            else:
                B.add_edge(f"acct_{author}", tag_node, weight=1)

    return B


def build_account_projection(B: nx.Graph) -> nx.Graph:
    """
    Project bipartite network onto account layer only.
    Two accounts are connected if they share at least one hashtag.
    Edge weight = number of shared hashtags.
    """
    account_nodes = [n for n, d in B.nodes(data=True) if d.get("bipartite") == 0]
    return nx.bipartite.weighted_projected_graph(B, account_nodes)


# ── Community Detection ───────────────────────────────────────────────────────
def detect_communities(G: nx.Graph) -> dict:
    """
    Apply Louvain community detection.
    Falls back to greedy modularity if python-louvain not installed.
    Returns dict: node → community_id
    """
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G, random_state=42)
        method = "Louvain"
    except ImportError:
        print("[INFO] python-louvain not installed. Using greedy modularity.")
        communities = nx.algorithms.community.greedy_modularity_communities(G)
        partition = {}
        for i, comm in enumerate(communities):
            for node in comm:
                partition[node] = i
        method = "Greedy Modularity"

    n_communities = len(set(partition.values()))
    print(f"[{method}] Detected {n_communities} communities.")
    return partition


# ── Network Statistics ────────────────────────────────────────────────────────
def compute_network_stats(B: nx.Graph, G_proj: nx.Graph, partition: dict) -> pd.DataFrame:
    """Compute centrality and community metrics for account nodes."""
    account_nodes = [n for n, d in B.nodes(data=True) if d.get("bipartite") == 0]

    # Degree in bipartite graph (= number of unique hashtags used)
    degree_dict = dict(B.degree(account_nodes))

    # Betweenness centrality in projection (= "bridge" accounts)
    if len(G_proj) > 1:
        betweenness = nx.betweenness_centrality(G_proj, normalized=True)
    else:
        betweenness = {n: 0.0 for n in account_nodes}

    records = []
    for node in account_nodes:
        node_data = B.nodes[node]
        records.append({
            "node":           node,
            "username":       node.replace("acct_", ""),
            "degree":         degree_dict.get(node, 0),
            "betweenness":    round(betweenness.get(node, 0.0), 6),
            "community":      partition.get(node, -1),
            "followers":      node_data.get("followers", 0),
            "content_type":   node_data.get("content_type", "unknown"),
        })

    stats_df = pd.DataFrame(records).sort_values("betweenness", ascending=False)
    return stats_df


def compute_hashtag_stats(B: nx.Graph) -> pd.DataFrame:
    """Top hashtags by degree (number of accounts using them)."""
    tag_nodes = [(n, d) for n, d in B.nodes(data=True) if d.get("bipartite") == 1]
    records = []
    for node, data in tag_nodes:
        records.append({
            "hashtag":         node.replace("tag_", "#"),
            "accounts_using":  B.degree(node),
            "total_uses":      sum(B[node][nb].get("weight", 1) for nb in B.neighbors(node)),
        })
    return pd.DataFrame(records).sort_values("total_uses", ascending=False)


# ── Centralization Test ───────────────────────────────────────────────────────
def test_centralization(G_proj: nx.Graph, stats_df: pd.DataFrame) -> dict:
    """
    Test whether the network is super-spreader-driven (centralized)
    vs. peer-to-peer (decentralized).

    Metric: Gini coefficient of degree distribution.
    Gini > 0.6 = highly centralized (super-spreaders dominate)
    Gini < 0.4 = decentralized
    """
    degrees = np.array([d for _, d in G_proj.degree()])
    if len(degrees) == 0:
        return {}

    # Gini coefficient
    degrees_sorted = np.sort(degrees)
    n = len(degrees_sorted)
    cumulative = np.cumsum(degrees_sorted)
    gini = (2 * np.sum((np.arange(1, n+1)) * degrees_sorted) / (n * cumulative[-1])) - (n+1)/n

    # Top-10% share
    top10_threshold = np.percentile(degrees, 90)
    top10_share = degrees[degrees >= top10_threshold].sum() / degrees.sum()

    # Network density
    density = nx.density(G_proj)

    result = {
        "gini_coefficient": round(gini, 4),
        "top10_degree_share": round(top10_share, 4),
        "network_density": round(density, 6),
        "n_nodes": len(G_proj),
        "n_edges": G_proj.number_of_edges(),
        "n_communities": len(set(stats_df["community"])),
        "structure": "Centralized (super-spreaders)" if gini > 0.5 else "Decentralized (peer-to-peer)"
    }
    return result


# ── Visualizations ────────────────────────────────────────────────────────────
def plot_network(B: nx.Graph, partition: dict, output_path: Path):
    """Plot bipartite network with community color coding."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    account_nodes = [n for n, d in B.nodes(data=True) if d.get("bipartite") == 0]
    hashtag_nodes = [n for n, d in B.nodes(data=True) if d.get("bipartite") == 1]

    # Layout
    pos = {}
    n_acc = len(account_nodes)
    n_tag = len(hashtag_nodes)
    for i, n in enumerate(account_nodes):
        pos[n] = (0, i / max(n_acc, 1))
    for i, n in enumerate(hashtag_nodes):
        pos[n] = (1, i / max(n_tag, 1))

    # Community colors for account nodes
    communities = set(partition.get(n, 0) for n in account_nodes)
    cmap = plt.cm.get_cmap("tab10", max(len(communities), 1))
    node_colors = [cmap(partition.get(n, 0) % 10) for n in account_nodes]

    # Draw
    nx.draw_networkx_nodes(B, pos, nodelist=account_nodes,
                           node_color=node_colors, node_size=40, alpha=0.85, ax=ax)
    nx.draw_networkx_nodes(B, pos, nodelist=hashtag_nodes,
                           node_color="#f0c040", node_size=60, alpha=0.85, ax=ax)
    nx.draw_networkx_edges(B, pos, alpha=0.15, edge_color="#ffffff", ax=ax)

    # Labels for top hashtags only
    top_tags = sorted(hashtag_nodes, key=lambda n: B.degree(n), reverse=True)[:15]
    labels = {n: n.replace("tag_", "#") for n in top_tags}
    nx.draw_networkx_labels(B, pos, labels=labels, font_size=7,
                            font_color="#f0c040", ax=ax)

    patches = [mpatches.Patch(color="#f0c040", label="Hashtag"),
               mpatches.Patch(color=cmap(0), label="Account (community)")]
    ax.legend(handles=patches, loc="upper right", facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    ax.set_title("Bipartite Network: Accounts ↔ Hashtags\n(AI-NCII Ecosystem on X)",
                 color="white", fontsize=13, pad=15)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"[Plot] Network graph → {output_path}")


def plot_degree_distribution(stats_df: pd.DataFrame, output_path: Path):
    """Log-log degree distribution plot to test for power-law (scale-free network)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0d1117")

    for ax in axes:
        ax.set_facecolor("#111827")

    # Degree distribution
    degrees = stats_df["degree"].values
    degree_counts = Counter(degrees)
    x = sorted(degree_counts.keys())
    y = [degree_counts[d] for d in x]

    axes[0].scatter(x, y, color="#60a5fa", alpha=0.7, s=25)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Degree (log)", color="white")
    axes[0].set_ylabel("Count (log)", color="white")
    axes[0].set_title("Degree Distribution (log-log)\nPower-law ≈ scale-free network", color="white")
    axes[0].tick_params(colors="white")
    for spine in axes[0].spines.values():
        spine.set_edgecolor("#374151")

    # Community sizes
    comm_sizes = stats_df["community"].value_counts().head(10)
    bars = axes[1].barh(range(len(comm_sizes)), comm_sizes.values, color="#34d399", alpha=0.85)
    axes[1].set_yticks(range(len(comm_sizes)))
    axes[1].set_yticklabels([f"Community {i}" for i in comm_sizes.index], color="white", fontsize=9)
    axes[1].set_xlabel("Number of accounts", color="white")
    axes[1].set_title("Community Size Distribution\n(Louvain method)", color="white")
    axes[1].tick_params(colors="white")
    for spine in axes[1].spines.values():
        spine.set_edgecolor("#374151")

    plt.suptitle("Network Structure Analysis — AI-NCII Ecosystem on X",
                 color="white", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"[Plot] Degree distribution → {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/synthetic/synthetic_posts.csv")
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} posts.")

    print("\n=== Building bipartite network ===")
    B = build_bipartite_network(df)
    print(f"Bipartite graph: {B.number_of_nodes()} nodes, {B.number_of_edges()} edges")

    print("\n=== Building account projection ===")
    G_proj = build_account_projection(B)
    print(f"Account projection: {G_proj.number_of_nodes()} nodes, {G_proj.number_of_edges()} edges")

    print("\n=== Detecting communities ===")
    partition = detect_communities(G_proj)

    print("\n=== Computing statistics ===")
    stats_df = compute_network_stats(B, G_proj, partition)
    hashtag_df = compute_hashtag_stats(B)
    centralization = test_centralization(G_proj, stats_df)

    # Save tables
    stats_df.to_csv(TABLE_DIR / "account_network_stats.csv", index=False)
    hashtag_df.to_csv(TABLE_DIR / "hashtag_stats.csv", index=False)
    pd.DataFrame([centralization]).to_csv(TABLE_DIR / "centralization_test.csv", index=False)

    print("\n=== Centralization Results ===")
    for k, v in centralization.items():
        print(f"  {k}: {v}")

    print("\n=== Top 10 accounts by betweenness centrality ===")
    print(stats_df.head(10)[["username", "degree", "betweenness", "community", "followers"]].to_string(index=False))

    print("\n=== Top 15 hashtags ===")
    print(hashtag_df.head(15).to_string(index=False))

    print("\n=== Plotting ===")
    plot_network(B, partition, FIG_DIR / "network_bipartite.png")
    plot_degree_distribution(stats_df, FIG_DIR / "degree_distribution.png")

    print("\nNetwork analysis complete.")
