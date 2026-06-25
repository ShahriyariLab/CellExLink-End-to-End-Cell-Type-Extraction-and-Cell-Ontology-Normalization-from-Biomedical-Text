"""Concatenate BioC XML training files into one BioC XML file.

This script is optional. It does not change the normal training flow.

Input:
- One BioC XML file or a directory of corpus folders.
- An output BioC XML path.

Output:
- One BioC XML file made by concatenating the selected input documents.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator, Sequence

import bioc


def iter_input_files(paths: Sequence[str | Path]) -> Iterator[Path]:
    """Yield BioC XML files to concatenate.

    Input:
    - File paths or directory paths.

    Output:
    - Paths to BioC XML files. For directories, `train.xml` is selected.
    """
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            yield path
            continue

        if path.is_dir():
            direct_split = path / "train.xml"
            if direct_split.is_file():
                yield direct_split
                continue

            child_splits = [
                child / "train.xml"
                for child in sorted(path.iterdir())
                if child.is_dir() and (child / "train.xml").is_file()
            ]
            if child_splits:
                yield from child_splits
                continue

        raise FileNotFoundError(f"Could not find BioC training XML under: {path}")


def parse_args() -> argparse.Namespace:
    """Read command-line arguments for XML concatenation.

    Input:
    - Values passed on the command line.

    Output:
    - An `argparse.Namespace` with input paths and output path.
    """
    parser = argparse.ArgumentParser(
        description="Concatenate BioC XML training files into one BioC XML file."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input BioC XML files or directories. Directory inputs use `train.xml`.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination BioC XML file.",
    )
    return parser.parse_args()


def concatenate_bioc_xml(input_paths: Sequence[str | Path], output_path: str | Path) -> int:
    """Concatenate multiple BioC XML files into one output BioC XML file.

    Input:
    - Input BioC XML file paths or dataset directories.
    - One output BioC XML path.

    Output:
    - The number of documents written to the output file.
    """
    resolved_inputs = list(iter_input_files(input_paths))
    if not resolved_inputs:
        raise ValueError("No input BioC XML files were found.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged_collection = bioc.BioCCollection()
    merged_collection.source = "CellExLink"
    merged_collection.date = ""
    merged_collection.key = "train"

    document_count = 0
    for input_path in resolved_inputs:
        with input_path.open("r", encoding="utf-8") as handle:
            collection = bioc.load(handle)

        if document_count == 0:
            merged_collection.source = collection.source
            merged_collection.date = collection.date
            merged_collection.key = collection.key
            merged_collection.infons = dict(collection.infons)
            merged_collection.encoding = collection.encoding
            merged_collection.version = collection.version
            merged_collection.standalone = collection.standalone

        for document in collection.documents:
            merged_collection.add_document(document)
            document_count += 1

    with output_path.open("w", encoding="utf-8") as handle:
        bioc.dump(merged_collection, handle)

    return document_count


def main() -> int:
    """Run BioC XML concatenation from the command line.

    Input:
    - Command-line arguments.

    Output:
    - Returns `0` after writing the concatenated BioC XML file.
    """
    args = parse_args()
    document_count = concatenate_bioc_xml(args.inputs, args.output)
    print(f"Wrote {document_count} documents to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
