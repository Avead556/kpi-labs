from pathlib import Path

from rdflib import Graph
from rdflib.compare import isomorphic

LAB = Path(__file__).resolve().parents[1]
SOURCE = LAB / "results" / "devices.owl"
OUT = LAB / "results" / "export"

FORMATS = [
    ("RDF/XML", "xml", "owl"),
    ("Turtle", "turtle", "ttl"),
    ("N-Triples", "nt", "nt"),
    ("N3", "n3", "n3"),
    ("JSON-LD", "json-ld", "jsonld"),
]


def main():
    graph = Graph().parse(SOURCE, format="xml")
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{SOURCE.name}: {len(graph)} triples")
    print(f"{'format':10} {'file':16} {'bytes':>7} {'lines':>6}  read back")
    for name, rdflib_format, ext in FORMATS:
        path = OUT / f"devices.{ext}"
        graph.serialize(destination=path, format=rdflib_format, encoding="utf-8")
        back = Graph().parse(path, format=rdflib_format)
        text = path.read_text(encoding="utf-8")
        same = "same graph" if isomorphic(graph, back) else "DIFFERENT"
        print(f"{name:10} {path.name:16} {path.stat().st_size:7} {text.count(chr(10)):6}  "
              f"{len(back)} triples, {same}")


if __name__ == "__main__":
    main()
