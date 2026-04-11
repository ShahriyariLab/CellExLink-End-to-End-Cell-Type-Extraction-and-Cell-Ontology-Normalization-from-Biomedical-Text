from pathlib import Path

from .normalize import main as run_normalization


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CELL_TYPES = BASE_DIR / "cell_ontology_v2025-12-17.jsonl"
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "CellExLink-Sapbert"
DEFAULT_ABBREVIATIONS = BASE_DIR / "abbreviations.tsv"


def resolve_model_reference(model_reference):
    model_reference_str = str(model_reference)
    model_reference_path = Path(model_reference_str)
    if model_reference_path.exists():
        resolved_path = model_reference_path.resolve()
        return str(resolved_path), resolved_path.name
    return model_reference_str, model_reference_path.name


def normalize_bioc(
    input_xml,
    output_xml,
    cell_types=DEFAULT_CELL_TYPES,
    model_path=DEFAULT_MODEL_PATH,
    abbreviations=DEFAULT_ABBREVIATIONS,
    disable_abbreviations=False,
    abbr_verbose=False,
):
    input_xml = Path(input_xml)
    output_xml = Path(output_xml)
    cell_types = Path(cell_types)
    abbreviations = Path(abbreviations) if abbreviations is not None else None

    if not input_xml.is_file():
        raise FileNotFoundError(f"Missing input XML: {input_xml}")
    if not cell_types.is_file():
        raise FileNotFoundError(f"Missing cell ontology JSONL: {cell_types}")

    model_name_or_path, model_name = resolve_model_reference(model_path)
    abbr_paths = []
    if not disable_abbreviations and abbreviations is not None:
        if not abbreviations.is_file():
            raise FileNotFoundError(f"Missing abbreviations TSV: {abbreviations}")
        abbr_paths = [str(abbreviations)]

    output_xml.parent.mkdir(parents=True, exist_ok=True)

    run_normalization(
        term_filename=str(cell_types),
        abbr_paths=abbr_paths,
        input_paths=str(input_xml),
        output_paths=str(output_xml),
        model_names={model_name: model_name_or_path},
        abbr_verbose=abbr_verbose,
    )
