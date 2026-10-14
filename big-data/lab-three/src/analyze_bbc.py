from collections import Counter
from pathlib import Path

from rdflib import BNode, Graph, URIRef
from rdflib.collection import Collection
from rdflib.namespace import OWL, RDF, RDFS

LAB = Path(__file__).resolve().parents[1]
SOURCE = LAB / "data" / "food.ttl"


def names(graph, node):
    if isinstance(node, BNode):
        members = graph.value(node, OWL.unionOf)
        if members is not None:
            return [names(graph, m)[0] for m in Collection(graph, members)]
        return ["(anonymous)"]
    return [graph.qname(node)]


def describe(graph, prop, predicate):
    found = []
    for node in graph.objects(prop, predicate):
        found += names(graph, node)
    return " | ".join(found) or "-"


def text(graph, node, predicate):
    value = graph.value(node, predicate)
    return " ".join(str(value).split()) if value else ""


def main():
    graph = Graph().parse(SOURCE, format="turtle")
    graph.bind("fo-authors", "http://purl.org/ontology/fo/#")
    graph.bind("cc-license", "http://creativecommons.org/licenses/by/4.0#")
    print(f"{SOURCE.name}: {len(graph)} triples, ontology "
          f"{next(graph.subjects(RDF.type, OWL.Ontology))}")
    print(f"title: {text(graph, URIRef('http://purl.org/ontology/fo/'), URIRef('http://purl.org/dc/terms/title'))}")

    classes = sorted(graph.subjects(RDF.type, OWL.Class), key=graph.qname)
    print(f"\n== {len(classes)} classes ==")
    for cls in classes:
        parents = [p for parent in graph.objects(cls, RDFS.subClassOf) for p in names(graph, parent)]
        print(f"{graph.qname(cls):22} {'< ' + ', '.join(parents) if parents else '':28} "
              f"{text(graph, cls, RDFS.label)}")

    objects = sorted(graph.subjects(RDF.type, OWL.ObjectProperty), key=graph.qname)
    print(f"\n== {len(objects)} object properties (domain -> range) ==")
    for prop in objects:
        print(f"{graph.qname(prop):22} {describe(graph, prop, RDFS.domain):38} -> "
              f"{describe(graph, prop, RDFS.range)}")

    data = sorted(graph.subjects(RDF.type, OWL.DatatypeProperty), key=graph.qname)
    print(f"\n== {len(data)} datatype properties ==")
    for prop in data:
        print(f"{graph.qname(prop):22} {describe(graph, prop, RDFS.domain):38} -> "
              f"{describe(graph, prop, RDFS.range)}")

    prefixes = Counter()
    for triple in graph:
        for node in triple:
            if isinstance(node, URIRef):
                prefix = graph.qname(node).split(":")[0] if ":" in graph.qname(node) else node
                prefixes[prefix] += 1
    print("\n== vocabularies used (terms per prefix) ==")
    print(", ".join(f"{p} {n}" for p, n in prefixes.most_common()))


if __name__ == "__main__":
    main()
