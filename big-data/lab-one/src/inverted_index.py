import re
import shutil
import sys
import time
from pathlib import Path

from session import spark_session

LAB = Path(__file__).resolve().parents[1]
DATA = LAB / "data"
RESULTS = LAB / "results"
INPUT_FORMAT = "org.apache.hadoop.mapreduce.lib.input.TextInputFormat"
KEY_CLASS = "org.apache.hadoop.io.LongWritable"
VALUE_CLASS = "org.apache.hadoop.io.Text"

WORD = re.compile(r"[a-z]+(?:['’][a-z]+)*")
LOOKUPS = ["cipher", "words", "file"]


def read_lines(sc, path):
    name = path.name
    lines = sc.newAPIHadoopFile(str(path), INPUT_FORMAT, KEY_CLASS, VALUE_CLASS)
    return lines.map(lambda kv: (name, kv[0], kv[1]))


def position_mapper(record):
    name, offset, line = record
    return [(word, f"{name}@{offset}") for word in set(WORD.findall(line.lower()))]


def document_mapper(record):
    name, _, line = record
    return [(word, name) for word in set(WORD.findall(line.lower()))]


def location_key(location):
    name, offset = location.rsplit("@", 1)
    return name, int(offset)


def position_reducer(locations):
    return ",".join(sorted(set(locations), key=location_key))


def document_reducer(names):
    return ",".join(sorted(set(names)))


def save(rdd, name):
    out = RESULTS / name
    shutil.rmtree(out, ignore_errors=True)
    rdd.sortByKey().map(lambda kv: f"{kv[0]}\t{kv[1]}").saveAsTextFile(str(out))
    return out


def main():
    lookups = sys.argv[1:] or LOOKUPS
    spark = spark_session("inverted-index")
    sc = spark.sparkContext
    files = sorted(DATA.glob("*/*.txt"))
    started = time.perf_counter()
    records = sc.union([read_lines(sc, f) for f in files]).cache()

    by_document = records.flatMap(document_mapper).groupByKey().mapValues(document_reducer).cache()
    by_position = records.flatMap(position_mapper).groupByKey().mapValues(position_reducer).cache()
    save(by_document, "inverted_index_documents")
    save(by_position, "inverted_index_positions")
    words = by_position.count()
    elapsed = time.perf_counter() - started

    print(f"Spark {sc.version}: {len(files)} files, {records.count()} lines, "
          f"{words} distinct words, {elapsed:.1f} s")
    print("saved to results/inverted_index_documents and results/inverted_index_positions")

    lines = {(n, o): t for n, o, t in records.collect()}
    documents = dict(by_document.filter(lambda kv: kv[0] in lookups).collect())
    positions = dict(by_position.filter(lambda kv: kv[0] in lookups).collect())
    for word in lookups:
        print(f"\n== {word} ==")
        print(f"documents: {word}\t{documents.get(word, '')}")
        located = [loc for loc in positions.get(word, "").split(",") if loc]
        shown = ",".join(located[:6]) + (f",... ({len(located)} in total)" if len(located) > 6 else "")
        print(f"positions: {word}\t{shown}")
        for location in located[:6]:
            name, offset = location_key(location)
            print(f"  {name}, offset {offset}: {lines[(name, offset)].strip()}")
    spark.stop()


if __name__ == "__main__":
    main()
