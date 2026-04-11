import argparse
import itertools
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args():
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Create SapBERT training data from Cell Ontology names and linked corpus mentions."
    )
    parser.add_argument(
        "--ontology-json",
        type=Path,
        default=base_dir / "cell_ontology_v2025-12-17.jsonl",
        help="Path to the Cell Ontology JSONL file.",
    )
    parser.add_argument(
        "--reference-xml",
        type=Path,
        nargs="*",
        default=[
            base_dir / "dataset" / "celllink" / "train.xml",
        ],
        help="One or more BioC XML files with gold CL identifiers in annotation infons.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "sapbert_data",
        help="Directory where the SapBERT training pairs file will be written.",
    )
    return parser.parse_args()


def normalize_term(value):
    return " ".join(str(value).strip().split())


def dedupe_terms(name, synonyms):
    seen = set()
    terms = []
    for raw_term in [name, *synonyms]:
        term = normalize_term(raw_term)
        if not term:
            continue
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        terms.append(term)
    return terms


def load_ontology_concepts(path):
    concepts = {}
    with path.open(encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            concept_id = normalize_term(entry.get("norm_concept_id", ""))
            name = normalize_term(entry.get("norm_preferred_label", ""))
            synonyms = entry.get("synonyms", []) or []
            if not concept_id or not name:
                continue
            merged_terms = concepts.get(concept_id, [])
            concepts[concept_id] = dedupe_terms("", [*merged_terms, name, *synonyms])
    return sorted(concepts.items())


def extract_cl_ids(identifier_text, allowed_relations=None):
    if not identifier_text or "CL:" not in identifier_text:
        return []
    if allowed_relations:
        pattern = r"\(skos:([^)]+)\)(CL:\d+)"
        return [
            concept_id
            for relation, concept_id in re.findall(pattern, identifier_text)
            if relation in allowed_relations
        ]
    return re.findall(r"CL:\d+", identifier_text)


def iter_corpus_mentions(xml_paths, allowed_relations=None):
    concept_to_mentions = {}
    for xml_path in xml_paths:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for annotation in root.findall(".//annotation"):
            text_node = annotation.find("text")
            mention_text = normalize_term(text_node.text if text_node is not None else "")
            if not mention_text:
                continue

            infons = {
                infon.attrib.get("key"): normalize_term(infon.text or "")
                for infon in annotation.findall("infon")
            }
            identifier = infons.get("identifier", "")
            for concept_id in extract_cl_ids(identifier, allowed_relations=allowed_relations):
                concept_to_mentions.setdefault(concept_id, []).append(mention_text)
    return {
        concept_id: dedupe_terms("", mentions)
        for concept_id, mentions in concept_to_mentions.items()
    }


def merge_concept_terms(ontology_concepts, concept_mentions):
    merged = {}
    for concept_id, terms in ontology_concepts:
        merged[concept_id] = list(terms)

    for concept_id, mentions in concept_mentions.items():
        merged.setdefault(concept_id, [])
        merged[concept_id].extend(mentions)

    merged_concepts = []
    for concept_id in sorted(merged):
        canonical_terms = dedupe_terms("", merged[concept_id])
        if canonical_terms:
            merged_concepts.append((concept_id, canonical_terms))
    return merged_concepts


def write_name_dictionary(concepts, output_path):
    row_count = 0
    with output_path.open("w", encoding="utf-8") as fp:
        for concept_id, terms in concepts:
            for term in terms:
                fp.write(f"{term}\t{concept_id}\n")
                row_count += 1
    return row_count


def write_training_pairs(concepts, output_path):
    pair_count = 0
    concept_count = 0
    with output_path.open("w", encoding="utf-8") as fp:
        for concept_id, terms in concepts:
            if len(terms) < 2:
                continue
            concept_count += 1
            for left, right in itertools.combinations(terms, 2):
                fp.write(f"{concept_id}||{left}||{right}\n")
                pair_count += 1
    return concept_count, pair_count


def main():
    args = parse_args()
    ontology_concepts = load_ontology_concepts(args.ontology_json)
    reference_xmls = [path for path in args.reference_xml if path.is_file()]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    concept_mentions = iter_corpus_mentions(reference_xmls, allowed_relations={"exact"})
    combined_concepts = merge_concept_terms(ontology_concepts, concept_mentions)
    dictionary_path = args.output_dir / "cell_ontology_plus_mentions.tsv"
    pairs_path = args.output_dir / "sapbert_training_pairs.txt"
    dictionary_rows = write_name_dictionary(combined_concepts, dictionary_path)
    paired_concepts, pair_rows = write_training_pairs(combined_concepts, pairs_path)

    print(f"Ontology: {args.ontology_json}")
    print(f"Reference XML files used: {len(reference_xmls)}")
    for xml_path in reference_xmls:
        print(f"  - {xml_path}")
    print(f"Ontology concepts with names: {len(ontology_concepts)}")
    print(f"Dictionary rows written: {dictionary_rows}")
    print(f"Dictionary file: {dictionary_path}")
    print(f"Concepts with at least 2 terms: {paired_concepts}")
    print(f"Pair rows written: {pair_rows}")
    print(f"Pairs file: {pairs_path}")


if __name__ == "__main__":
    main()
