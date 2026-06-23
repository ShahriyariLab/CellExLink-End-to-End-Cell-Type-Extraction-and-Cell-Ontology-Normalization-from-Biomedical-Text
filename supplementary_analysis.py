#!/usr/bin/env python3
"""Generate corpus vs Cell Ontology overlap.

The figure shows case-sensitive unique entity strings per corpus as stacked bars:
- matches a CL preferred label or synonym after conservative normalization
- no CL lexical match

For CellLink, annotations with type ``cell_vague`` are excluded.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - depends on user environment
    raise SystemExit(
        "matplotlib is required to create the figure. Install it with: "
        "pip install matplotlib"
    ) from exc


SPACE_RE = re.compile(r"\s+")
DASH_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2212_-]+")
BOUNDARY_PUNCT_RE = re.compile(r"^[\s\.,;:()\[\]{}]+|[\s\.,;:()\[\]{}]+$")
CL_ID_RE = re.compile(r"(?<![A-Za-z0-9])CL\s*[:_]\s*(\d+)", re.IGNORECASE)
CELLLINK_VAGUE_TYPE = "cell_vague"

# Default split selection follows Table 1 in the manuscript:
# CellLink train + validation; all other corpora train + test.
DEFAULT_CORPORA: Mapping[str, Mapping[str, Sequence[str]]] = {
    "CellLink": {
        "aliases": ("celllink",),
        "splits": ("train", "validation", "valid", "dev"),
    },
    "BioID": {
        "aliases": ("bioid", "bio-id"),
        "splits": ("train", "test"),
    },
    "CRAFT": {
        "aliases": ("craft",),
        "splits": ("train", "test"),
    },
    "AnatEM": {
        "aliases": ("anatem", "anat-em"),
        "splits": ("train", "test"),
    },
    "JNLPBA": {
        "aliases": ("jnlpba", "jnlp-ba"),
        "splits": ("train", "test"),
    },
}


@dataclass
class Ontology:
    relaxed_aliases: set[str]


@dataclass
class CorpusFigureData:
    name: str
    unique_surface_forms: set[str] = field(default_factory=set)
    matched_surface_forms: set[str] = field(default_factory=set)


def compact_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return SPACE_RE.sub(" ", text).strip()


def strict_key(value: object) -> str:
    return compact_text(value).casefold()


def relaxed_key(value: object) -> str:
    text = strict_key(value)
    text = text.translate(
        str.maketrans(
            {
                "’": "'",
                "‘": "'",
                "`": "'",
                "“": '"',
                "”": '"',
            }
        )
    )
    text = DASH_RE.sub(" ", text)
    text = BOUNDARY_PUNCT_RE.sub("", text)
    text = re.sub(r"\bcells\b", "cell", text)
    text = SPACE_RE.sub(" ", text).strip()
    if not text:
        return text

    tokens = text.split(" ")
    final = tokens[-1]
    if final.endswith("ies") and len(final) > 4 and final not in {"species", "series"}:
        final = final[:-3] + "y"
    elif final.endswith(("sses", "shes", "ches", "xes", "zes")) and len(final) > 4:
        final = final[:-2]
    elif (
        final.endswith("s")
        and len(final) > 3
        and not final.endswith(("ss", "us", "is", "ous"))
    ):
        final = final[:-1]
    tokens[-1] = final
    return " ".join(tokens)


def _flatten_synonyms(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        output: list[str] = []
        for nested in value.values():
            output.extend(_flatten_synonyms(nested))
        return output
    if isinstance(value, Sequence):
        output: list[str] = []
        for nested in value:
            output.extend(_flatten_synonyms(nested))
        return output
    return [str(value)]


def iter_json_records(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        first_nonempty = ""
        position = handle.tell()
        for line in handle:
            if line.strip():
                first_nonempty = line.lstrip()[:1]
                break
        handle.seek(position)

        if first_nonempty == "[":
            payload = json.load(handle)
            if not isinstance(payload, list):
                raise ValueError(f"Expected a JSON array in {path}")
            for item in payload:
                if isinstance(item, dict):
                    yield item
            return

        line_failed = False
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError:
                line_failed = True
                break
            if isinstance(item, dict):
                yield item

    if not line_failed:
        return

    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        item, index = decoder.raw_decode(text, index)
        if isinstance(item, dict):
            yield item


def first_present(record: Mapping[str, object], keys: Sequence[str]) -> object:
    for key in keys:
        value = record.get(key)
        if value not in (None, "", []):
            return value
    return ""


def load_ontology(path: Path) -> Ontology:
    relaxed_aliases: set[str] = set()

    for record in iter_json_records(path):
        obsolete = bool(record.get("obsolete") or record.get("is_obsolete") or record.get("deprecated"))
        if obsolete:
            continue

        raw_id = compact_text(
            first_present(
                record,
                ("norm_concept_id", "concept_id", "id", "obo_id", "identifier"),
            )
        )
        if not CL_ID_RE.findall(raw_id):
            continue

        label = compact_text(
            first_present(
                record,
                (
                    "norm_preferred_label",
                    "preferred_label",
                    "label",
                    "name",
                    "prefLabel",
                ),
            )
        )
        synonyms = _flatten_synonyms(
            first_present(
                record,
                ("synonyms", "synonym", "aliases", "alias", "alt_labels"),
            )
        )

        for term in [label, *synonyms]:
            key = relaxed_key(term)
            if key:
                relaxed_aliases.add(key)

    if not relaxed_aliases:
        raise ValueError(
            f"No CL concepts were found in {path}. Expected fields such as "
            "norm_concept_id and norm_preferred_label."
        )
    return Ontology(relaxed_aliases=relaxed_aliases)


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def annotation_content(annotation: ET.Element) -> tuple[str, str]:
    mention = ""
    annotation_type = ""
    for child in annotation:
        name = local_name(child.tag)
        if name == "text":
            mention = compact_text(child.text or "")
        elif name == "infon":
            key = compact_text(child.attrib.get("key", "")).casefold()
            if key in {"type", "semantic_type", "entity_type"} and not annotation_type:
                annotation_type = compact_text(child.text or "")
    return mention, annotation_type


def parse_bioc_files(name: str, files: Sequence[Path], ontology: Ontology) -> CorpusFigureData:
    corpus = CorpusFigureData(name=name)

    for xml_path in files:
        try:
            iterator = ET.iterparse(xml_path, events=("end",))
            for _, element in iterator:
                tag = local_name(element.tag)
                if tag == "annotation":
                    mention, annotation_type = annotation_content(element)
                    if name == "CellLink" and annotation_type == CELLLINK_VAGUE_TYPE:
                        element.clear()
                        continue
                    if mention:
                        corpus.unique_surface_forms.add(mention)
                        if relaxed_key(mention) in ontology.relaxed_aliases:
                            corpus.matched_surface_forms.add(mention)
                    element.clear()
                elif tag in {"document", "passage"}:
                    element.clear()
        except ET.ParseError as exc:
            raise ValueError(f"Could not parse BioC XML file {xml_path}: {exc}") from exc

    return corpus


def searchable_path_key(path: Path) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(path).casefold())


def discover_default_files(dataset_root: Path) -> dict[str, list[Path]]:
    xml_files = sorted(dataset_root.rglob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML files were found under {dataset_root}")

    selected: dict[str, list[Path]] = {}
    for corpus_name, config in DEFAULT_CORPORA.items():
        aliases = [searchable_path_key(Path(alias)) for alias in config["aliases"]]
        split_keys = [searchable_path_key(Path(split)) for split in config["splits"]]

        corpus_candidates = [
            path
            for path in xml_files
            if any(alias in searchable_path_key(path.relative_to(dataset_root)) for alias in aliases)
        ]
        split_candidates = [
            path
            for path in corpus_candidates
            if any(split in searchable_path_key(path.relative_to(dataset_root)) for split in split_keys)
        ]

        if corpus_name == "CellLink":
            split_candidates = [
                path
                for path in split_candidates
                if "test" not in searchable_path_key(path.relative_to(dataset_root))
            ]

        if split_candidates:
            selected[corpus_name] = split_candidates
        elif corpus_candidates:
            selected[corpus_name] = corpus_candidates
        else:
            selected[corpus_name] = []
    return selected


def parse_file_overrides(specifications: Sequence[str]) -> dict[str, list[Path]]:
    overrides: dict[str, list[Path]] = {}
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(
                f"Invalid --files value {specification!r}; expected DATASET=path1.xml,path2.xml"
            )
        name, raw_paths = specification.split("=", 1)
        paths = [Path(item).expanduser() for item in raw_paths.split(",") if item.strip()]
        if not compact_text(name) or not paths:
            raise ValueError(f"Invalid --files value {specification!r}")
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing files for {name}: " + ", ".join(str(path) for path in missing)
            )
        overrides[compact_text(name)] = paths
    return overrides


def percent(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


def make_figure(corpora: Sequence[CorpusFigureData], output_dir: Path) -> None:
    datasets = [corpus.name for corpus in corpora]
    matched = [len(corpus.matched_surface_forms) for corpus in corpora]
    totals = [len(corpus.unique_surface_forms) for corpus in corpora]
    unmatched = [total - overlap for total, overlap in zip(totals, matched)]
    label_fontsize = 12
    tick_fontsize = 12
    title_fontsize = 14
    annotation_fontsize = 10
    legend_fontsize = 11

    figure, axis = plt.subplots(figsize=(10, 5.6))
    positions = list(range(len(datasets)))
    axis.barh(positions, matched, label="Matches a CL preferred label or synonym")
    axis.barh(positions, unmatched, left=matched, label="No CL lexical match")
    axis.set_yticks(positions, labels=datasets, fontsize=tick_fontsize)
    axis.invert_yaxis()
    axis.set_xlabel("Unique annotated entity strings", fontsize=label_fontsize)
    axis.set_title(
        "Corpus entity coverage by Cell Ontology (CL)",
        fontsize=title_fontsize,
    )
    axis.tick_params(axis="x", labelsize=tick_fontsize)

    largest_total = max(totals, default=1)
    for y, total, overlap in zip(positions, totals, matched):
        axis.text(
            total + largest_total * 0.012,
            y,
            f"n={total:,}; {percent(overlap, total):.1f}% matched",
            va="center",
            fontsize=annotation_fontsize,
        )

    axis.set_xlim(0, largest_total * 1.35)
    axis.legend(loc="lower right", frameon=False, fontsize=legend_fontsize)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    figure.savefig(
        output_dir / "S1_Fig_corpus_CL_overlap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create S1_Fig_corpus_CL_overlap.png from BioC corpora and a CL dictionary."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset"),
        help="Root directory containing the corpus directories (default: dataset).",
    )
    parser.add_argument(
        "--ontology",
        type=Path,
        default=Path("normalization/cell_ontology_v2025-12-17.jsonl"),
        help="Cell Ontology JSON/JSONL alias inventory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/corpus_cl_overlap"),
        help="Output directory.",
    )
    parser.add_argument(
        "--files",
        action="append",
        default=[],
        metavar="DATASET=FILE1,FILE2",
        help="Override automatic discovery for one dataset.",
    )
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    ontology_path = args.ontology.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not ontology_path.is_file():
        raise FileNotFoundError(f"Ontology file not found: {ontology_path}")
    if not dataset_root.is_dir() and not args.files:
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

    ontology = load_ontology(ontology_path)
    corpus_files = discover_default_files(dataset_root) if dataset_root.is_dir() else {}
    corpus_files.update(parse_file_overrides(args.files))

    ordered_names = list(DEFAULT_CORPORA)
    ordered_names.extend(name for name in corpus_files if name not in ordered_names)

    corpora: list[CorpusFigureData] = []
    for corpus_name in ordered_names:
        files = corpus_files.get(corpus_name, [])
        if files:
            corpora.append(parse_bioc_files(corpus_name, files, ontology))

    if not corpora:
        raise RuntimeError(
            "No corpora were analyzed. Check --dataset-root or provide explicit --files arguments."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    make_figure(corpora, output_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)



"""
python3 CL_analysis.py \
  --ontology cell_ontology_v2025-12-17.jsonl \
  --output-dir analysis/corpus_cl_overlap
"""
