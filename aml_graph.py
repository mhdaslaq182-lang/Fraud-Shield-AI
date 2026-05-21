# =============================================
#   AML Graph Analysis Module
#   Anti-Money Laundering Network Detection
#   Sri Lanka Banking Fraud Detection 🇱🇰
# =============================================

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import datetime

os.makedirs("plots", exist_ok=True)
os.makedirs("data",  exist_ok=True)

# Sri Lankan Banks
SL_BANKS = [
    "BOC", "Peoples", "Commercial", "Sampath",
    "HNB", "NSB", "Seylan", "NTB", "DFCC", "PanAsia"
]

# Sri Lankan Cities
SL_CITIES = [
    "Colombo", "Kandy", "Galle", "Jaffna",
    "Negombo", "Matara", "Kurunegala", "Trincomalee",
    "Anuradhapura", "Badulla"
]


def generate_transaction_network(n_accounts=30, n_transactions=80, seed=42):
    """
    Generate a realistic Sri Lankan banking transaction network.
    Includes normal transfers + suspicious money laundering patterns.
    """
    np.random.seed(seed)

    # Generate account IDs
    accounts = [f"LK{bank}{i:04d}"
                for i in range(n_accounts)
                for bank in ["BOC","SAM","HNB","COM"][:1]]
    accounts = [f"LK-{np.random.choice(SL_BANKS)}-{i:04d}"
                for i in range(n_accounts)]

    transactions = []

    # ── Normal transactions ──
    for _ in range(int(n_transactions * 0.7)):
        sender   = np.random.choice(accounts)
        receiver = np.random.choice([a for a in accounts if a != sender])
        transactions.append({
            "sender":    sender,
            "receiver":  receiver,
            "amount":    round(np.random.exponential(15000), 2),
            "city":      np.random.choice(SL_CITIES),
            "timestamp": datetime.now().isoformat(),
            "is_suspicious": 0
        })

    # ── Suspicious: Circular transfers (A→B→C→A) ──
    for _ in range(5):
        a, b, c = np.random.choice(accounts, 3, replace=False)
        amt = round(np.random.choice([99900, 499900, 999900]), 2)
        for sender, receiver in [(a,b), (b,c), (c,a)]:
            transactions.append({
                "sender":    sender,
                "receiver":  receiver,
                "amount":    amt,
                "city":      np.random.choice(SL_CITIES),
                "timestamp": datetime.now().isoformat(),
                "is_suspicious": 1
            })

    # ── Suspicious: Smurfing (one account → many) ──
    smurf_account = np.random.choice(accounts)
    targets       = np.random.choice(
        [a for a in accounts if a != smurf_account], 6, replace=False
    )
    for target in targets:
        transactions.append({
            "sender":    smurf_account,
            "receiver":  target,
            "amount":    round(np.random.uniform(45000, 55000), 2),
            "city":      "Colombo",
            "timestamp": datetime.now().isoformat(),
            "is_suspicious": 1
        })

    df = pd.DataFrame(transactions)
    df.to_csv("data/transaction_network.csv", index=False)
    print(f"  ✅ Network generated: {len(df)} transactions, {n_accounts} accounts")
    return df, accounts, smurf_account


def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    """Build directed transaction graph."""
    G = nx.DiGraph()
    for _, row in df.iterrows():
        if G.has_edge(row["sender"], row["receiver"]):
            G[row["sender"]][row["receiver"]]["weight"]      += row["amount"]
            G[row["sender"]][row["receiver"]]["count"]       += 1
            G[row["sender"]][row["receiver"]]["suspicious"]  += row["is_suspicious"]
        else:
            G.add_edge(
                row["sender"], row["receiver"],
                weight=row["amount"],
                count=1,
                suspicious=row["is_suspicious"]
            )
    return G


def detect_circular_transfers(G: nx.DiGraph) -> list:
    """Detect circular money flows — classic AML pattern."""
    circles = []
    try:
        cycles = list(nx.simple_cycles(G))
        for cycle in cycles:
            if len(cycle) >= 2:
                total = sum(
                    G[cycle[i]][cycle[(i+1) % len(cycle)]]["weight"]
                    for i in range(len(cycle))
                )
                circles.append({
                    "accounts": cycle,
                    "length":   len(cycle),
                    "total_lkr": total
                })
    except:
        pass
    return circles


def detect_smurfing(G: nx.DiGraph, threshold: int = 4) -> list:
    """
    Detect smurfing — one account sending to many others.
    Common in Sri Lankan hawala networks.
    """
    smurfs = []
    for node in G.nodes():
        out_degree = G.out_degree(node)
        if out_degree >= threshold:
            total_sent = sum(
                G[node][nb]["weight"]
                for nb in G.successors(node)
            )
            smurfs.append({
                "account":    node,
                "sent_to":    out_degree,
                "total_lkr":  total_sent,
                "risk":       "HIGH" if out_degree >= 6 else "MEDIUM"
            })
    return sorted(smurfs, key=lambda x: x["sent_to"], reverse=True)


def detect_high_risk_accounts(G: nx.DiGraph, threshold: int = 5) -> list:
    """Find accounts with unusually high connections."""
    high_risk = []
    for node in G.nodes():
        degree     = G.degree(node)
        in_degree  = G.in_degree(node)
        out_degree = G.out_degree(node)
        if degree >= threshold:
            high_risk.append({
                "account":    node,
                "total_connections": degree,
                "incoming":   in_degree,
                "outgoing":   out_degree,
            })
    return sorted(high_risk, key=lambda x: x["total_connections"], reverse=True)


def visualize_network(G: nx.DiGraph, title: str = "Transaction Network"):
    """Visualize the transaction network."""
    plt.figure(figsize=(16, 10))

    # Layout
    pos = nx.spring_layout(G, seed=42, k=2)

    # Node sizes by degree
    node_sizes = [300 + G.degree(n) * 200 for n in G.nodes()]

    # Node colors by risk
    node_colors = []
    for node in G.nodes():
        if G.degree(node) >= 6:
            node_colors.append("#E63946")   # red = high risk
        elif G.degree(node) >= 4:
            node_colors.append("#F4A261")   # orange = medium
        else:
            node_colors.append("#2A9D8F")   # green = normal

    # Edge colors by suspicious flag
    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        if G[u][v].get("suspicious", 0) > 0:
            edge_colors.append("#E63946")
            edge_widths.append(2.5)
        else:
            edge_colors.append("#ADB5BD")
            edge_widths.append(0.8)

    # Draw
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                           node_color=node_colors, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors,
                           width=edge_widths, arrows=True,
                           arrowsize=15, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=6, font_color="white")

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend = [
        Patch(color="#E63946", label="High Risk Account"),
        Patch(color="#F4A261", label="Medium Risk Account"),
        Patch(color="#2A9D8F", label="Normal Account"),
        Line2D([0],[0], color="#E63946", lw=2, label="Suspicious Transfer"),
        Line2D([0],[0], color="#ADB5BD", lw=1, label="Normal Transfer"),
    ]
    plt.legend(handles=legend, loc="upper left", fontsize=9)
    plt.title(f"🇱🇰 {title}\nSri Lanka Banking Transaction Network",
              fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("plots/aml_network.png", dpi=150, bbox_inches="tight")
    print("  ✅ Network graph saved: plots/aml_network.png")
    plt.close()


def run_aml_analysis():
    """Run complete AML analysis."""
    print("\n" + "="*55)
    print("  🔍 AML GRAPH ANALYSIS")
    print("  Sri Lanka Banking Fraud Detection 🇱🇰")
    print("="*55)

    # Generate network
    print("\n[1/4] Generating transaction network...")
    df, accounts, smurf = generate_transaction_network(
        n_accounts=30, n_transactions=80
    )

    # Build graph
    print("\n[2/4] Building transaction graph...")
    G = build_graph(df)
    print(f"  Accounts (nodes) : {G.number_of_nodes()}")
    print(f"  Transactions     : {G.number_of_edges()}")

    # Detect patterns
    print("\n[3/4] Detecting suspicious patterns...")

    # Circular transfers
    circles = detect_circular_transfers(G)
    print(f"\n  🔄 CIRCULAR TRANSFERS DETECTED: {len(circles)}")
    for c in circles[:5]:
        print(f"     Chain : {' → '.join(c['accounts'])} → {c['accounts'][0]}")
        print(f"     Amount: Rs. {c['total_lkr']:,.2f}")

    # Smurfing
    smurfs = detect_smurfing(G, threshold=4)
    print(f"\n  🐟 SMURFING ACCOUNTS DETECTED: {len(smurfs)}")
    for s in smurfs[:5]:
        print(f"     Account : {s['account']}")
        print(f"     Sent to : {s['sent_to']} accounts")
        print(f"     Total   : Rs. {s['total_lkr']:,.2f}")
        print(f"     Risk    : {s['risk']}")

    # High risk accounts
    high_risk = detect_high_risk_accounts(G, threshold=5)
    print(f"\n  ⚠️  HIGH RISK ACCOUNTS: {len(high_risk)}")
    for h in high_risk[:5]:
        print(f"     Account    : {h['account']}")
        print(f"     Connections: {h['total_connections']}")
        print(f"     In/Out     : {h['incoming']}/{h['outgoing']}")

    # Visualize
    print("\n[4/4] Generating network visualization...")
    visualize_network(G, title="AML Detection")

    print("\n" + "="*55)
    print("  ✅ AML ANALYSIS COMPLETE!")
    print(f"  🔄 Circular transfers : {len(circles)}")
    print(f"  🐟 Smurfing accounts  : {len(smurfs)}")
    print(f"  ⚠️  High risk accounts : {len(high_risk)}")
    print(f"  📊 Graph saved        : plots/aml_network.png")
    print("="*55)

    return G, circles, smurfs, high_risk


if __name__ == "__main__":
    run_aml_analysis()
