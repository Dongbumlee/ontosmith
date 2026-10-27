#!/usr/bin/env python3
"""Extract the FIBO inventories the contracts import consumes.

Parses the RDF/XML (.rdf) files of a pinned FIBO checkout with the Python
standard library only (xml.etree.ElementTree) and writes flat JSON
inventories of every declared owl:Class and property.

Input (relative to this script — clone FIBO beside the script):
    fibo/FND/Agreements/Agreements.rdf
    fibo/FND/Agreements/Contracts.rdf
    fibo/FND/AgentsAndPeople/People.rdf
Output (next to this script):
    fibo-agreements-inventory.json   (the scoped module)
    fibo-people-inventory.json       (the reference file Person
                                      is drawn from)

Each entry:
    iri         full IRI (XML entities are expanded by the parser)
    kind        "class" | "object_property" | "datatype_property"
    label       rdfs:label text, verbatim, or null
    definition  skos:definition if present, else rdfs:comment, else null
    parents     IRIs from rdfs:subClassOf / rdfs:subPropertyOf elements
                that carry an rdf:resource attribute, as written (expanded
                to full IRIs by the parser). Blank-node parents -- OWL
                restrictions and union/intersection class expressions --
                have no IRI and are therefore not listed.
    domain      rdfs:domain IRI for properties, or null
    range       rdfs:range IRI for properties, or null
    source_file repo-relative path of the declaring file

Notes:
  * Only top-level declarations (direct children of rdf:RDF carrying an
    rdf:about) are counted; nested anonymous classes are not declarations.
  * No rdf:Property declarations occur in these two files; if one ever
    appears it is emitted with kind "property" rather than being silently
    mis-typed as an object/datatype property.
  * If a property ever declares multiple domain/range resources, the value
    is emitted as a list; the pinned FIBO files declare at most one each,
    so in practice domain/range are a single IRI string or null.
"""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
}

RDF_ABOUT = f"{{{NS['rdf']}}}about"
RDF_RESOURCE = f"{{{NS['rdf']}}}resource"

# Extraction jobs: (repo-relative source files, output name).
# The Agreements module is the scoped inventory; People is a
# reference file (Contracts.rdf imports it) drawn on for the
# Person class only.
JOBS = [
    (["FND/Agreements/Agreements.rdf",
      "FND/Agreements/Contracts.rdf"],
     "fibo-agreements-inventory.json"),
    (["FND/AgentsAndPeople/People.rdf"],
     "fibo-people-inventory.json"),
]

TAG_TO_KIND = {
    f"{{{NS['owl']}}}Class": "class",
    f"{{{NS['owl']}}}ObjectProperty": "object_property",
    f"{{{NS['owl']}}}DatatypeProperty": "datatype_property",
    f"{{{NS['rdf']}}}Property": "property",  # not present in the pinned files
}


def text_of(element, tag):
    """Verbatim text of the first direct child with `tag`, or None."""
    child = element.find(tag)
    if child is None or child.text is None:
        return None
    text = child.text.strip()
    return text if text else None


def resource_list(element, tag):
    """rdf:resource values of direct children with `tag` (IRIs only)."""
    values = []
    for child in element.findall(tag):
        resource = child.get(RDF_RESOURCE)
        if resource:
            values.append(resource)
    return values


def single_or_none(values):
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return values  # see module docstring; does not occur in the pinned files


def extract_file(path, source_file):
    tree = ET.parse(path)
    root = tree.getroot()
    entries = []
    for element in root:
        kind = TAG_TO_KIND.get(element.tag)
        if kind is None:
            continue
        iri = element.get(RDF_ABOUT)
        if not iri:
            continue  # anonymous / nested class expression, not a declaration
        if kind == "class":
            parents = resource_list(element, f"{{{NS['rdfs']}}}subClassOf")
        else:
            parents = resource_list(element, f"{{{NS['rdfs']}}}subPropertyOf")
        definition = text_of(element, f"{{{NS['skos']}}}definition")
        if definition is None:
            definition = text_of(element, f"{{{NS['rdfs']}}}comment")
        is_property = kind != "class"
        entries.append(
            {
                "iri": iri,
                "kind": kind,
                "label": text_of(element, f"{{{NS['rdfs']}}}label"),
                "definition": definition,
                "parents": parents,
                "domain": single_or_none(
                    resource_list(element, f"{{{NS['rdfs']}}}domain")
                )
                if is_property
                else None,
                "range": single_or_none(
                    resource_list(element, f"{{{NS['rdfs']}}}range")
                )
                if is_property
                else None,
                "source_file": source_file,
            }
        )
    return entries


def main():
    work_dir = Path(__file__).resolve().parent
    fibo_dir = work_dir / "fibo"
    for source_files, out_name in JOBS:
        inventory = []
        for source_file in source_files:
            path = fibo_dir / source_file
            if not path.is_file():
                print(f"error: FIBO file not found: {path}",
                      file=sys.stderr)
                return 1
            inventory.extend(extract_file(path, source_file))
        out_path = work_dir / out_name
        out_path.write_text(
            json.dumps(inventory, indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {out_path} ({len(inventory)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
