import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
RUN_NER_PREDICT = PROJECT_ROOT / "recognition" / "src" / "predict_NER.py"
NER_RUNTIME_SUMMARY_FILENAME = "predict_runtime_summary.json"
EL_RUNTIME_SUMMARY_FILENAME = "el_predict_runtime_summary.json"


def resolve_model_reference(model_reference):
    model_reference_str = str(model_reference)
    model_reference_path = Path(model_reference_str)
    if model_reference_path.exists():
        resolved_path = model_reference_path.resolve()
        return str(resolved_path), resolved_path
    return model_reference_str, None


def parse_args():
    parser = argparse.ArgumentParser(description="Measure runtime for NER or EL.")
    parser.add_argument("--task", choices={"ner", "el"}, required=True, help="Which pipeline stage to benchmark.")

    parser.add_argument("--input-xml", type=Path, default=None, help="Input BioC XML file.")
    parser.add_argument(
        "--output-xml",
        type=Path,
        default=None,
        help=(
            "Optional output BioC XML file. For NER this is written after prediction finishes, and requires --input-xml."
        ),
    )

    parser.add_argument("--model-path", default=None, help="Local model directory or Hugging Face model name.")
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Optional tokenizer path if it differs from the model path (NER only).",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Optional tokenizer sequence length override for NER.",
    )
    parser.add_argument(
        "--pad-to-max-length",
        action="store_true",
        help="Pad every batch item to max length during NER prediction.",
    )
    parser.add_argument(
        "--max-predict-samples",
        type=int,
        default=None,
        help="Optional cap on NER prediction examples.",
    )
    parser.add_argument(
        "--ner-warmup-runs",
        type=int,
        default=1,
        help="Number of untimed warmup runs before measuring NER inference.",
    )
    parser.add_argument(
        "--per-device-predict-batch-size",
        type=int,
        default=16,
        help="NER prediction batch size per device.",
    )
    parser.add_argument(
        "--preprocessing-num-workers",
        type=int,
        default=None,
        help="Dataset preprocessing workers for NER.",
    )
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Overwrite cached NER preprocessing artifacts.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory for NER.",
    )
    parser.add_argument(
        "--model-revision",
        default="main",
        help="Model revision to use when loading from the Hugging Face Hub (NER only).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face auth token for NER.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom code from the model repo during NER loading.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable fp16 prediction for NER if supported by the hardware.",
    )

    parser.add_argument(
        "--cell-types",
        type=Path,
        default=PROJECT_ROOT / "normalization" / "cell_ontology_v2025-12-17.jsonl",
        help="Cell ontology JSONL file for EL.",
    )
    parser.add_argument(
        "--abbreviations",
        type=Path,
        default=PROJECT_ROOT / "normalization" / "abbreviations.tsv",
        help="Abbreviation TSV file for EL.",
    )
    parser.add_argument(
        "--disable-abbreviations",
        action="store_true",
        help="Disable abbreviation handling for EL runtime measurement.",
    )
    parser.add_argument(
        "--el-warmup-runs",
        type=int,
        default=1,
        help="Number of EL warmup runs before timing mention normalization.",
    )
    return parser.parse_args()


def print_gpu_info():
    try:
        import torch

        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        print(f"torch.cuda.is_available() = {gpu_available}")
        print(f"GPU count = {gpu_count}")
        if gpu_available:
            for idx in range(gpu_count):
                print(f"GPU {idx}: {torch.cuda.get_device_name(idx)}")
    except Exception:
        print("Could not query GPU information.")


def get_bioc_stats(xml_path: Path):
    parser = ET.XMLParser(encoding="utf-8")
    root = ET.parse(xml_path, parser=parser).getroot()

    num_documents = 0
    num_passages = 0
    total_tokens = 0
    total_characters = 0
    total_mentions = 0

    for document in root.findall(".//document"):
        num_documents += 1
        for passage in document.findall("./passage"):
            num_passages += 1
            text_node = passage.find("./text")
            if text_node is not None and text_node.text:
                total_characters += len(text_node.text)
                total_tokens += len(text_node.text.split())
            total_mentions += len(passage.findall("./annotation"))

    if num_documents == 0:
        raise ValueError(f"No documents found in {xml_path}")

    return {
        "num_documents": num_documents,
        "num_passages": num_passages,
        "total_tokens": total_tokens,
        "total_characters": total_characters,
        "total_mentions": total_mentions,
    }


def run_ner(args):
    if args.model_path is None:
        raise ValueError("NER runtime requires --model-path.")

    model_name_or_path, resolved_model_path = resolve_model_reference(args.model_path)

    input_xml = args.input_xml.resolve() if args.input_xml is not None else None
    output_xml = args.output_xml.resolve() if args.output_xml is not None else None

    if input_xml is None:
        raise ValueError("NER runtime requires --input-xml.")
    if input_xml is not None and not input_xml.is_file():
        raise FileNotFoundError(f"Missing input XML: {input_xml}")
    if output_xml is not None and input_xml is None:
        raise ValueError("Writing NER output XML requires --input-xml.")
    if args.ner_warmup_runs < 0:
        raise ValueError("--ner-warmup-runs must be >= 0.")

    if output_xml is not None:
        output_xml.parent.mkdir(parents=True, exist_ok=True)

    output_dir = (
        output_xml.parent
        if output_xml is not None
        else (BASE_DIR / "tmp_ner_runtime")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model path: {resolved_model_path or model_name_or_path}")
    print(f"Prediction output dir: {output_dir}")
    if input_xml is not None:
        print(f"Input XML: {input_xml}")
    if output_xml is not None:
        print(f"Prediction XML output: {output_xml}")
    print_gpu_info()

    stats = get_bioc_stats(input_xml)
    print(f"Number of documents/abstracts: {stats['num_documents']}")
    print(f"Number of passages: {stats['num_passages']}")
    print(f"Total characters: {stats['total_characters']}")
    print(f"Total number of whitespace tokens (from XML): {stats['total_tokens']}")

    model_path_arg = str(resolved_model_path or model_name_or_path)

    cmd = [
        sys.executable,
        "-u",
        str(RUN_NER_PREDICT),
        "--model-path",
        model_path_arg,
        "--output-dir",
        str(output_dir),
        "--warmup-runs",
        str(args.ner_warmup_runs),
        "--per-device-predict-batch-size",
        str(args.per_device_predict_batch_size),
        "--model-revision",
        str(args.model_revision),
    ]

    if output_xml is not None:
        cmd.extend(["--input-xml", str(input_xml), "--output-xml", str(output_xml)])
    else:
        cmd.extend(["--input-xml", str(input_xml)])

    if args.tokenizer_path is not None:
        cmd.extend(["--tokenizer-path", str(args.tokenizer_path)])
    if args.max_seq_length is not None:
        cmd.extend(["--max-seq-length", str(args.max_seq_length)])
    if args.pad_to_max_length:
        cmd.append("--pad-to-max-length")
    if args.max_predict_samples is not None:
        cmd.extend(["--max-predict-samples", str(args.max_predict_samples)])
    if args.preprocessing_num_workers is not None:
        cmd.extend(["--preprocessing-num-workers", str(args.preprocessing_num_workers)])
    if args.overwrite_cache:
        cmd.append("--overwrite-cache")
    if args.cache_dir is not None:
        cmd.extend(["--cache-dir", str(args.cache_dir)])
    if args.token is not None:
        cmd.extend(["--token", str(args.token)])
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if args.fp16:
        cmd.append("--fp16")

    print("Launching NER predict-only benchmark...")
    print("Command:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=BASE_DIR, check=False)
    runtime_summary_path = output_dir / NER_RUNTIME_SUMMARY_FILENAME

    print("\n=== NER Runtime Summary ===")
    if not runtime_summary_path.is_file():
        print(f"Missing predict-only runtime summary: {runtime_summary_path}")
        return result.returncode

    runtime_summary = json.loads(runtime_summary_path.read_text(encoding="utf-8"))
    elapsed = float(runtime_summary["elapsed_seconds"])
    print(f"NER predict-only wall time: {elapsed:.6f} seconds")

    if stats["num_passages"] > 0 and elapsed > 0:
        ms_per_passage = (elapsed * 1000.0) / stats["num_passages"]
        print(f"Latency: {ms_per_passage:.4f} ms/passage")

    if stats["num_documents"] > 0 and elapsed > 0:
        ms_per_abstract = (elapsed * 1000.0) / stats["num_documents"]
        print(f"Latency: {ms_per_abstract:.4f} ms/abstract")

    if stats["total_characters"] > 0 and elapsed > 0:
        chars_per_second = stats["total_characters"] / elapsed
        print(f"Throughput: {chars_per_second:.4f} characters/second")

    if stats["total_tokens"] > 0 and elapsed > 0:
        ms_per_1k_tokens = (elapsed * 1000.0) / (stats["total_tokens"] / 1000.0)
        print(f"Latency: {ms_per_1k_tokens:.4f} ms/1k tokens")

    return result.returncode


def run_el(args):
    if args.model_path is None or args.input_xml is None:
        raise ValueError("EL runtime requires --model-path and --input-xml.")

    model_name_or_path, resolved_model_path = resolve_model_reference(args.model_path)
    input_xml = args.input_xml.resolve()
    output_xml = (
        args.output_xml.resolve()
        if args.output_xml is not None
        else (BASE_DIR / "tmp_el_runtime" / "el_predictions.xml").resolve()
    )
    cell_types = args.cell_types.resolve()
    abbreviations = args.abbreviations.resolve()

    if not input_xml.is_file():
        raise FileNotFoundError(f"Missing input XML: {input_xml}")
    if not cell_types.is_file():
        raise FileNotFoundError(f"Missing cell ontology JSONL: {cell_types}")
    if not args.disable_abbreviations and not abbreviations.is_file():
        raise FileNotFoundError(f"Missing abbreviations TSV: {abbreviations}")

    output_xml.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input XML: {input_xml}")
    print(f"Output XML: {output_xml}")
    print(f"Model path: {resolved_model_path or model_name_or_path}")
    print(f"Cell ontology: {cell_types}")
    print_gpu_info()

    stats = get_bioc_stats(input_xml)
    print(f"Number of documents/abstracts: {stats['num_documents']}")
    print(f"Total number of mentions to link: {stats['total_mentions']}")
    print(f"Total number of whitespace tokens (from XML): {stats['total_tokens']}")

    print(f"EL warmup runs: {args.el_warmup_runs}")

    abbr_arg = str(abbreviations) if not args.disable_abbreviations else ""
    model_path_arg = str(resolved_model_path or model_name_or_path)

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "normalization.normalize",
        str(cell_types),
        abbr_arg,
        str(input_xml),
        str(output_xml),
        "--model-path",
        model_path_arg,
        "--el-warmup-runs",
        str(args.el_warmup_runs),
    ]
    if args.disable_abbreviations:
        cmd.append("--disable-abbreviations")

    print("Launching normalization benchmark...")
    print("Command:", " ".join(cmd))

    result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
    runtime_summary_path = output_xml.parent / EL_RUNTIME_SUMMARY_FILENAME

    print("\n=== EL Runtime Summary ===")
    if not runtime_summary_path.is_file():
        print(f"Missing predict-only runtime summary: {runtime_summary_path}")
        return result.returncode

    runtime_summaries = json.loads(runtime_summary_path.read_text(encoding="utf-8"))
    if len(runtime_summaries) != 1:
        print(f"Expected exactly one EL runtime summary, found {len(runtime_summaries)}")
        return result.returncode

    runtime_summary = next(iter(runtime_summaries.values()))
    elapsed = float(runtime_summary["elapsed_seconds"])
    total_mentions_processed = int(runtime_summary.get("total_mentions_processed", stats["total_mentions"]))
    print(f"EL predict-only wall time: {elapsed:.6f} seconds")

    if total_mentions_processed > 0 and elapsed > 0:
        mentions_per_second = total_mentions_processed / elapsed
        print(f"Throughput: {mentions_per_second:.4f} mentions/second")

    if total_mentions_processed > 0 and elapsed > 0:
        ms_per_mention = (elapsed * 1000.0) / total_mentions_processed
        print(f"Latency: {ms_per_mention:.4f} ms/mention")

    return result.returncode


def main():
    args = parse_args()
    if args.task == "ner":
        raise SystemExit(run_ner(args))
    raise SystemExit(run_el(args))


if __name__ == "__main__":
    main()
