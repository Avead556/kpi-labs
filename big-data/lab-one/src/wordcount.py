import argparse
import re
import shutil
import time
from operator import add
from pathlib import Path

from session import spark_session

LAB = Path(__file__).resolve().parents[1]
SMALL = LAB / "data" / "wordcount"
CORPUS = LAB / "data" / "shakespeare"
RESULTS = LAB / "results"

WORD = re.compile(r"[a-z]+(?:['’][a-z]+)*")


def words(line):
    return WORD.findall(line.lower())


def mapper(line):
    return [(word, 1) for word in words(line)]


def reducer(a, b):
    return a + b


def show_small(sc):
    print("== Mapper output ==")
    for path in sorted(SMALL.glob("*.txt")):
        pairs = sc.textFile(str(path)).flatMap(mapper).collect()
        print(f"{path.name}: " + " ".join(f"<{w}, {n}>" for w, n in pairs))

    pairs = sc.textFile(str(SMALL / "*.txt")).flatMap(mapper)
    grouped = pairs.groupByKey().mapValues(list).collectAsMap()
    print('\n== Reducer input for "it" ==')
    print(f'key="it"; values=iterator{grouped["it"]}')

    counts = pairs.reduceByKey(reducer).sortBy(lambda kv: (-kv[1], kv[0])).collect()
    print("\n== Word count, A.txt + B.txt ==")
    for word, n in counts:
        print(f"<{word}, {n}>")


def count_corpus(sc):
    files = sorted(CORPUS.glob("*.txt"))
    started = time.perf_counter()
    lines = sc.textFile(",".join(str(f) for f in files))
    counts = lines.flatMap(mapper).reduceByKey(reducer).cache()

    out = RESULTS / "wordcount"
    shutil.rmtree(out, ignore_errors=True)
    counts.sortBy(lambda kv: (-kv[1], kv[0])).map(lambda kv: f"{kv[0]}\t{kv[1]}") \
        .saveAsTextFile(str(out))
    total = counts.values().sum()
    distinct = counts.count()
    top = counts.takeOrdered(15, key=lambda kv: (-kv[1], kv[0]))
    elapsed = time.perf_counter() - started

    print(f"\n== Shakespeare: {len(files)} files, {lines.getNumPartitions()} partitions ==")
    print(f"words: {total}, distinct: {distinct}, {elapsed:.1f} s")
    print("top 15:", ", ".join(f"{w} {n}" for w, n in top))
    parts = sorted(p.name for p in out.glob("part-*"))
    print(f"saved to results/wordcount: {', '.join(parts)}")

    print("\n== Words per document ==")
    for path in files:
        n = sc.textFile(str(path)).flatMap(mapper).map(lambda kv: kv[1]).reduce(add)
        print(f"{path.name:16} {n:7}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hold", type=int, default=0)
    args = parser.parse_args()
    spark = spark_session("wordcount")
    sc = spark.sparkContext
    print(f"Spark {sc.version}, master {sc.master}, {sc.defaultParallelism} cores")
    show_small(sc)
    count_corpus(sc)
    if args.hold:
        print(f"\nSpark UI: {sc.uiWebUrl}")
        time.sleep(args.hold)
    spark.stop()


if __name__ == "__main__":
    main()
