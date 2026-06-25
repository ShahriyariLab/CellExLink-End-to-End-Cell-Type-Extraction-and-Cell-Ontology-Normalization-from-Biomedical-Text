"""Download the default CellExLink model checkpoints.

Input:
- Optional output directory and a `--force` flag from the command line.

Output:
- Local model folders and a `models.json` manifest inside the chosen output directory.
"""

import argparse
import json
from pathlib import Path


MODEL_SPECS = {
    "CellExLink-bioformer16L": {
        "repo_id": "almire/CellExLink-bioformer16L",
        "stage": "NER",
    },
    "CellExLink-Sapbert": {
        "repo_id": "almire/CellExLink-Sapbert",
        "stage": "NEN",
    },
}


def parse_args() -> argparse.Namespace:
    """Read command-line options for model download.

    Input:
    - Values passed on the command line.

    Output:
    - An `argparse.Namespace` with the output folder and force flag.
    """
    parser = argparse.ArgumentParser(
        description="Download CellExLink models into the repository root models/ folder."
    )
    parser.add_argument(
        "--output-root",
        default="models",
        help="Directory where model folders and models.json will be written.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if the destination folders already exist.",
    )
    return parser.parse_args()


def get_snapshot_download():
    """Import the Hugging Face download helper only when needed.

    Input:
    - No direct function arguments.

    Output:
    - The `snapshot_download` function from `huggingface_hub`.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed. Run `python -m pip install huggingface_hub` first."
        ) from exc
    return snapshot_download


def write_manifest(output_root: Path) -> Path:
    """Write a small JSON file describing the downloaded models.

    Input:
    - `output_root`: folder where the models are stored.

    Output:
    - The path to the created `models.json` file.
    """
    manifest_path = output_root / "models.json"
    manifest_data = {
        "models": {
            model_name: {
                "repo_id": spec["repo_id"],
                "stage": spec["stage"],
                "local_path": str(output_root / model_name),
            }
            for model_name, spec in MODEL_SPECS.items()
        }
    }
    manifest_path.write_text(json.dumps(manifest_data, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    """Download each default model and record where it was saved.

    Input:
    - Command-line arguments.

    Output:
    - Writes model files and a manifest to disk.
    """
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    snapshot_download = get_snapshot_download()

    print(f"Downloading CellExLink models into: {output_root}")
    for model_name, spec in MODEL_SPECS.items():
        local_dir = output_root / model_name
        if local_dir.exists() and not args.force:
            print(f"Skipping existing model folder: {local_dir}")
            continue

        print(f"Downloading {model_name} from {spec['repo_id']}")
        snapshot_download(
            repo_id=spec["repo_id"],
            local_dir=str(local_dir),
        )

    manifest_path = write_manifest(output_root)
    print(f"Wrote manifest: {manifest_path}")
    print("Finished downloading models.")


if __name__ == "__main__":
    main()
