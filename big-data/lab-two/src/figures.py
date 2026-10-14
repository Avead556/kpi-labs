import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

LAB = Path(__file__).resolve().parents[1]
RESULTS = json.loads((LAB / "results" / "pagerank.json").read_text(encoding="utf-8"))
OUT = LAB / "docs" / "screenshots"
COLORS = ["#1f6fb2", "#d1495b", "#2e933c", "#edae49", "#7b4ea3", "#00798c"]

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def toy():
    data = next(d for d in RESULTS["datasets"] if d["key"] == "toy")
    graph = nx.DiGraph(data["edgeList"])
    pos = {"A": (0, 1), "B": (1, 1.6), "C": (1.6, 0.6), "D": (0.5, -0.2), "E": (1.8, -0.4)}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    sizes = [9000 * data["ranks"][n] + 600 for n in graph.nodes]
    nx.draw_networkx_nodes(graph, pos, node_size=sizes, node_color="#cfe3f5",
                           edgecolors="#1f6fb2", ax=ax)
    nx.draw_networkx_edges(graph, pos, node_size=sizes, arrowsize=18, width=1.5,
                           edge_color="#555", connectionstyle="arc3,rad=0.12", ax=ax)
    for node, (x, y) in pos.items():
        classic = data["classicRanks"][node]
        ax.text(x, y + 0.02, node, ha="center", va="center", fontsize=14, weight="bold")
        ax.text(x, y - 0.2 - 0.75 * data['ranks'][node], f"PR = {data['ranks'][node]:.3f}\nкласична: "
                + (f"{classic:.3f}" if classic is not None else "зникла"),
                ha="center", va="top", fontsize=9)
    ax.set_xlim(-0.5, 2.3)
    ax.set_ylim(-1.1, 2.0)
    ax.axis("off")
    fig.savefig(OUT / "fig-toy.png")
    plt.close(fig)


def convergence():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for color, data in zip(COLORS, RESULTS["datasets"]):
        deltas = [h["delta"] for h in data["history"]]
        ax.semilogy(range(1, len(deltas) + 1), deltas, color=color, lw=1.8,
                    label=f"{data['file'].split('.')[0]} ({data['iterations']} іт.)")
    ax.axhline(RESULTS["tol"], color="#888", ls="--", lw=1)
    ax.text(1, RESULTS["tol"] * 1.4, "поріг 1e-6", color="#666", fontsize=9)
    ax.set_xlabel("ітерація")
    ax.set_ylabel("L1-зміна вектора рейтингів")
    ax.legend(fontsize=9)
    fig.savefig(OUT / "fig-convergence.png")
    plt.close(fig)


def damping():
    rows = RESULTS["dampingRuns"]
    d = [r["damping"] for r in rows]
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.4, 3.6))
    left.plot(d, [r["iterations"] for r in rows], "o-", color=COLORS[0])
    left.set_xlabel("коефіцієнт згасання d")
    left.set_ylabel("ітерацій до збіжності")
    right.plot(d, [100 * r["top10Share"] for r in rows], "o-", color=COLORS[1])
    right.set_xlabel("коефіцієнт згасання d")
    right.set_ylabel("частка рейтингу в топ-10, %")
    fig.tight_layout()
    fig.savefig(OUT / "fig-damping.png")
    plt.close(fig)


def partitions():
    rows = RESULTS["partitionRuns"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot([r["partitions"] for r in rows], [r["seconds"] for r in rows], "o-", color=COLORS[2])
    ax.set_xscale("log", base=2)
    ax.set_xticks([r["partitions"] for r in rows], [str(r["partitions"]) for r in rows])
    ax.set_xlabel("кількість розділів RDD")
    ax.set_ylabel("час 10 ітерацій, с")
    fig.savefig(OUT / "fig-partitions.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    toy()
    convergence()
    damping()
    partitions()
    print(f"saved {', '.join(sorted(p.name for p in OUT.glob('fig-*.png')))}")


if __name__ == "__main__":
    main()
