import argparse
import json
import platform
import time
from pathlib import Path

import networkx as nx
from scipy.stats import kendalltau

from pagerank import build_links, classic_pagerank, pagerank, read_edges
from session import spark_session

LAB = Path(__file__).resolve().parents[1]
DATA = LAB / "data"
RESULTS = LAB / "results"

DATASETS = [
    ("toy", "toy.txt", "іграшковий граф із 5 сторінок, E без вихідних посилань"),
    ("email", "email-Eu-core.txt.gz", "листування в європейському дослідницькому інституті"),
    ("wiki", "wiki-Vote.txt.gz", "голосування за адміністраторів Вікіпедії"),
    ("gnutella", "p2p-Gnutella31.txt.gz", "однорангова мережа Gnutella"),
    ("epinions", "soc-Epinions1.txt.gz", "мережа довіри Epinions.com"),
    ("notredame", "web-NotreDame.txt.gz", "веб-граф домену nd.edu"),
]
PARTITIONS = 12
DAMPING = 0.85
TOL = 1e-6
DAMPINGS = [0.5, 0.7, 0.85, 0.9, 0.95]
PARTITION_COUNTS = [1, 2, 4, 8, 12, 24]

LOG = []


def log(message=""):
    print(message, flush=True)
    LOG.append(message)


def top(ranks, k=10):
    return [page for page, _ in sorted(ranks.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def standard_pagerank(edges, damping):
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    started = time.perf_counter()
    ranks = nx.pagerank(graph, alpha=damping, tol=1e-12, max_iter=1000)
    return ranks, time.perf_counter() - started


def compare(ours, reference):
    pages = list(reference)
    l1 = sum(abs(ours.get(p, 0.0) - reference[p]) for p in pages)
    largest = max(abs(ours.get(p, 0.0) - reference[p]) for p in pages)
    head = top(reference, 100)
    tau = kendalltau([reference[p] for p in head], [ours.get(p, 0.0) for p in head]).statistic
    overlap = len(set(top(ours)) & set(top(reference)))
    return {"l1": l1, "max": largest, "top10Overlap": overlap, "topK": min(10, len(pages)),
            "kendallTop100": tau}


def run_dataset(sc, key, filename, description):
    path = DATA / filename
    started = time.perf_counter()
    edges = read_edges(sc, path, PARTITIONS).cache()
    links = build_links(edges, PARTITIONS)
    n = links.count()
    m = edges.count()
    dangling = links.filter(lambda kv: not kv[1]).count()
    load = time.perf_counter() - started

    ranks, history = pagerank(links, DAMPING, TOL)
    ours = dict(ranks.collect())
    seconds = history[-1]["seconds"]

    reference, nx_seconds = standard_pagerank(edges.collect(), DAMPING)
    classic = dict(classic_pagerank(links).collect())
    classic_total = sum(classic.values())
    classic_norm = {p: r / classic_total for p, r in classic.items()}

    result = {
        "key": key, "file": filename, "description": description,
        "nodes": n, "edges": m, "dangling": dangling, "loadSeconds": load,
        "iterations": len(history), "seconds": seconds, "history": history,
        "standardSeconds": nx_seconds, "sum": sum(ours.values()),
        "vsStandard": compare(ours, reference),
        "classic": {"pages": len(classic), "sum": classic_total,
                    "vsStandard": compare(classic_norm, reference)},
        "top10": [(p, ours[p], reference[p]) for p in top(ours)],
        "classicTop10": top(classic_norm),
    }
    if n <= 10:
        result["ranks"] = {p: ours[p] for p in sorted(ours)}
        result["classicRanks"] = {p: classic.get(p) for p in sorted(ours)}
        result["edgeList"] = sorted(edges.collect())

    v = result["vsStandard"]
    c = result["classic"]
    log(f"\n== {key}: {filename} - {description} ==")
    log(f"nodes {n}, edges {m}, without out-links {dangling}; loaded in {load:.1f} s")
    log(f"Spark MapReduce: {len(history)} iterations, {seconds:.1f} s "
        f"({seconds / len(history):.2f} s/iteration), sum of ranks {result['sum']:.6f}")
    log(f"networkx.pagerank: {nx_seconds:.2f} s")
    log(f"vs networkx: L1 {v['l1']:.2e}, max diff {v['max']:.2e}, "
        f"top-10 overlap {v['top10Overlap']}/{v['topK']}, Kendall tau top-100 {v['kendallTop100']:.4f}")
    log(f"classic Spark example (10 it.): {c['pages']} of {n} pages kept, sum {c['sum']:.1f}, "
        f"top-10 overlap {c['vsStandard']['top10Overlap']}/{v['topK']}, L1 after normalising "
        f"{c['vsStandard']['l1']:.3f}")
    log("top-10: " + ", ".join(f"{p} {r:.5f}" for p, r, _ in result["top10"]))
    edges.unpersist()
    links.unpersist()
    return result


def damping_experiment(sc):
    edges = read_edges(sc, DATA / "wiki-Vote.txt.gz", PARTITIONS)
    links = build_links(edges, PARTITIONS)
    runs = {}
    for d in DAMPINGS:
        ranks, history = pagerank(links, d, TOL, max_iter=200)
        runs[d] = (dict(ranks.collect()), history)
    reference = runs[DAMPING][0]
    rows = []
    log("\n== Damping factor, wiki-Vote ==")
    for d, (ranks, history) in runs.items():
        values = sorted(ranks.values(), reverse=True)
        row = {"damping": d, "iterations": len(history), "seconds": history[-1]["seconds"],
               "top1": values[0], "top10Share": sum(values[:10]),
               "minRank": values[-1],
               "top10OverlapWith085": len(set(top(ranks)) & set(top(reference))),
               "history": history}
        rows.append(row)
        log(f"d = {d:.2f}: {row['iterations']:3} iterations, {row['seconds']:5.1f} s, "
            f"max rank {row['top1']:.5f}, top-10 hold {100 * row['top10Share']:.2f} %, "
            f"min rank {row['minRank']:.2e}, top-10 shared with d = 0.85: "
            f"{row['top10OverlapWith085']}/10")
    links.unpersist()
    return rows


def partition_experiment(sc):
    rows = []
    log("\n== Partitions, soc-Epinions1, 10 iterations ==")
    for p in PARTITION_COUNTS:
        edges = read_edges(sc, DATA / "soc-Epinions1.txt.gz", p)
        links = build_links(edges, p)
        links.count()
        _, history = pagerank(links, DAMPING, tol=0.0, max_iter=10)
        seconds = history[-1]["seconds"]
        rows.append({"partitions": p, "seconds": seconds})
        log(f"{p:3} partitions: {seconds:5.1f} s, {seconds / 10:.2f} s/iteration")
        links.unpersist()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("only", nargs="*")
    parser.add_argument("--hold", type=int, default=0)
    args = parser.parse_args()
    only = args.only
    spark = spark_session("pagerank", memory="4g")
    sc = spark.sparkContext
    log(f"Spark {sc.version}, {sc.master}, {sc.defaultParallelism} cores, "
        f"Python {platform.python_version()}, networkx {nx.__version__}")
    results = {"damping": DAMPING, "tol": TOL, "partitions": PARTITIONS, "datasets": []}
    for key, filename, description in DATASETS:
        if not only or key in only:
            results["datasets"].append(run_dataset(sc, key, filename, description))
    if not only:
        results["dampingRuns"] = damping_experiment(sc)
        results["partitionRuns"] = partition_experiment(sc)
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "pagerank.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
        (RESULTS / "experiments_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    if args.hold:
        print(f"\nSpark UI: {sc.uiWebUrl}")
        time.sleep(args.hold)
    spark.stop()


if __name__ == "__main__":
    main()
