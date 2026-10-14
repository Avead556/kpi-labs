import time
from operator import add

DANGLING = "__dangling__"


def read_edges(sc, path, partitions):
    lines = sc.textFile(str(path), minPartitions=partitions)
    return (lines.filter(lambda line: line.strip() and not line.startswith("#"))
            .map(lambda line: tuple(line.split()[:2]))
            .distinct(partitions))


def build_links(edges, partitions):
    nodes = edges.flatMap(lambda edge: edge).distinct(partitions)
    out = edges.groupByKey(partitions)
    return (nodes.map(lambda node: (node, None))
            .leftOuterJoin(out, partitions)
            .mapValues(lambda pair: list(pair[1]) if pair[1] is not None else [])
            .partitionBy(partitions)
            .cache())


def contributions(record):
    page, (links, rank) = record
    yield page, 0.0
    if links:
        share = rank / len(links)
        for target in links:
            yield target, share
    else:
        yield DANGLING, rank


def pagerank(links, damping=0.85, tol=1e-6, max_iter=100, checkpoint_every=10):
    partitions = links.getNumPartitions()
    n = links.count()
    ranks = links.mapValues(lambda _: 1.0 / n).cache()
    history = []
    started = time.perf_counter()
    for iteration in range(1, max_iter + 1):
        sums = links.join(ranks).flatMap(contributions).reduceByKey(add, partitions).cache()
        dangling = sums.filter(lambda kv: kv[0] == DANGLING).values().sum()
        base = (1 - damping) / n + damping * dangling / n
        new = (sums.filter(lambda kv: kv[0] != DANGLING)
               .mapValues(lambda s: base + damping * s)
               .cache())
        if iteration % checkpoint_every == 0:
            new.localCheckpoint()
        delta = new.join(ranks).map(lambda kv: abs(kv[1][0] - kv[1][1])).sum()
        ranks.unpersist()
        sums.unpersist()
        ranks = new
        history.append({"iteration": iteration, "delta": delta, "dangling": dangling,
                        "seconds": time.perf_counter() - started})
        if delta < tol:
            break
    return ranks, history


def classic_pagerank(links, iterations=10):
    ranks = links.mapValues(lambda _: 1.0)
    for _ in range(iterations):
        sums = links.join(ranks).flatMap(
            lambda r: [(t, r[1][1] / len(r[1][0])) for t in r[1][0]])
        ranks = sums.reduceByKey(add).mapValues(lambda s: 0.15 + 0.85 * s)
    return ranks
