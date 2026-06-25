"""Run the full CellExLink pipeline on BioC XML files.

Input:
- One BioC XML file or a directory of BioC XML files.
- Paths to the NER model, NEN model, ontology JSONL, and abbreviation TSV.

Output:
- One normalized BioC XML file per input file with mention spans and Cell Ontology IDs added.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def pick_default_path(*candidates: str) -> str:
    """Return the first existing path from the given options.

    Input:
    - `candidates`: possible file or directory paths.

    Output:
    - A string path. If none exist yet, the first candidate is returned.
    """
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return candidates[0]


def parse_args() -> argparse.Namespace:
    """Read command-line arguments for the end-to-end pipeline.

    Input:
    - Values passed on the command line.

    Output:
    - An `argparse.Namespace` containing the user settings.
    """
    parser = argparse.ArgumentParser(
        description="Run CellExLink end-to-end on one BioC XML file or every XML file in a directory."
    )
    parser.add_argument(
        "input_path",
        help="Path to a BioC XML file or a directory containing BioC XML files.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Directory where per-file outputs will be written.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Explicit output BioC XML path for a single input file.",
    )
    parser.add_argument(
        "--ner-model-path",
        default=pick_default_path(
            "models/CellExLink-bioformer16L",
            "recognition/models/CellExLink-bioformer16L",
        ),
        help="NER model directory or Hub path.",
    )
    parser.add_argument(
        "--nen-model-path",
        default=pick_default_path(
            "models/CellExLink-Sapbert",
            "normalization/models/CellExLink-Sapbert",
        ),
        help="NEN model directory or Hub path.",
    )
    parser.add_argument(
        "--ontology-path",
        default="normalization/cell_ontology_v2025-12-17.jsonl",
        help="Cell Ontology JSONL file.",
    )
    parser.add_argument(
        "--abbreviations-path",
        default="normalization/abbreviations.tsv",
        help="Abbreviations TSV file.",
    )
    return parser.parse_args()


def derive_base_name(path: Path) -> str:
    """Build a clean file stem used for output names.

    Input:
    - `path`: input XML file path.

    Output:
    - A base filename without `.bioc.xml` or `.xml`.
    """
    name = path.name
    if name.endswith(".bioc.xml"):
        return name[: -len(".bioc.xml")]
    if name.endswith(".xml"):
        return name[: -len(".xml")]
    return path.stem


def collect_input_files(input_path: Path) -> list[Path]:
    """Expand a file or directory into the XML files that should be processed.

    Input:
    - `input_path`: one XML file or a directory containing XML files.

    Output:
    - A list of XML file paths sorted in processing order.
    """
    if input_path.is_file():
        if input_path.suffix.lower() != ".xml":
            raise ValueError(f"Expected an XML file, got: {input_path}")
        return [input_path]

    if input_path.is_dir():
        xml_files = sorted(path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() == ".xml")
        if not xml_files:
            raise ValueError(f"No XML files found in directory: {input_path}")
        return xml_files

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def run_command(command: list[str], stage_name: str, input_file: Path) -> None:
    """Run one pipeline stage and raise an error if it fails.

    Input:
    - `command`: command to execute.
    - `stage_name`: human-readable stage label.
    - `input_file`: file currently being processed.

    Output:
    - No return value. Raises `RuntimeError` on failure.
    """
    status = subprocess.run(command, check=False).returncode
    if status != 0:
        raise RuntimeError(f"{stage_name} failed for {input_file} with exit code {status}")


def run_pipeline_for_file(
    input_file: Path,
    output_root: Path | None,
    output_path: Path | None,
    ner_model_path: str,
    nen_model_path: str,
    ontology_path: str,
    abbreviations_path: str,
) -> Path:
    """Run NER first, then normalization, for one BioC XML file.

    Input:
    - One input XML file plus model/resource paths.
    - Either `output_root` for batch mode or `output_path` for a single file.

    Output:
    - The final normalized BioC XML path that was written.
    """
    base_name = derive_base_name(input_file)
    if output_path is not None:
        final_output_xml = output_path
    else:
        if output_root is None:
            raise ValueError("Either output_root or output_path must be provided.")
        final_output_xml = output_root / f"{base_name}.normalized.xml"
    final_output_xml.parent.mkdir(parents=True, exist_ok=True)

    if not Path(ner_model_path).exists():
        raise FileNotFoundError(
            f"NER model not found at {ner_model_path}. Run `python download_models.py` first "
            "or pass --ner-model-path."
        )
    if not Path(nen_model_path).exists():
        raise FileNotFoundError(
            f"NEN model not found at {nen_model_path}. Run `python download_models.py` first "
            "or pass --nen-model-path."
        )

    # Keep intermediate NER files in a temporary folder so only the final XML is saved.
    with tempfile.TemporaryDirectory(prefix=f"cellexlink_{base_name}_") as tmp_dir:
        ner_output_dir = Path(tmp_dir) / "ner"
        ner_output_dir.mkdir(parents=True, exist_ok=True)
        ner_output_xml = ner_output_dir / f"{base_name}.ner.xml"

        run_command(
            [
                sys.executable,
                "recognition/src/predict_NER.py",
                "--model-path",
                ner_model_path,
                "--input-xml",
                str(input_file),
                "--output-dir",
                str(ner_output_dir),
                "--output-xml",
                str(ner_output_xml),
            ],
            "CellExLink NER",
            input_file,
        )

        run_command(
            [
                sys.executable,
                "normalization/normalize.py",
                ontology_path,
                abbreviations_path,
                str(ner_output_xml),
                str(final_output_xml),
                "--model-path",
                nen_model_path,
            ],
            "CellExLink NEN",
            input_file,
        )

    return final_output_xml


def main() -> None:
    """Entry point for batch or single-file CellExLink prediction.

    Input:
    - Command-line arguments.

    Output:
    - Writes normalized XML files to disk and prints progress messages.
    """
    args = parse_args()
    input_path = Path(args.input_path)
    input_files = collect_input_files(input_path)
    output_path = Path(args.output_path) if args.output_path is not None else None

    if output_path is not None:
        if len(input_files) != 1:
            raise ValueError("--output-path can only be used when input_path is a single XML file.")
        output_root = None
    else:
        output_root = Path(args.output_root)
        output_root.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(input_files)} XML file(s) to process.")
    for input_file in input_files:
        print(f"Processing: {input_file}")
        final_output_xml = run_pipeline_for_file(
            input_file=input_file,
            output_root=output_root,
            output_path=output_path,
            ner_model_path=args.ner_model_path,
            nen_model_path=args.nen_model_path,
            ontology_path=args.ontology_path,
            abbreviations_path=args.abbreviations_path,
        )
        print(f"Finished: {final_output_xml}")


if __name__ == "__main__":
    main()

#python prediction_script.py path/to/your/bioc_xml_folder

#python prediction_script.py path/to/your/bioc_xml_folder --output-root batch_outputs

#python prediction_script.py examples/input_bioc.xml --output-path outputs/input_bioc.normalized.xml
