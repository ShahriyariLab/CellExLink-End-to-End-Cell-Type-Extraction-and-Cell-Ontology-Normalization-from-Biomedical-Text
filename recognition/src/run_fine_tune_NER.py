#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
RUN_NER = PROJECT_DIR / "NER.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch offset-based NER fine-tuning from raw BioC/text passages.")

    train_group = parser.add_mutually_exclusive_group(required=False)
    train_group.add_argument("--train-file", type=Path, help="Training JSON/JSONL/CSV file.")
    train_group.add_argument(
        "--train-xml",
        type=Path,
        default=Path("dataset/train_all.xml"),
        help="Training BioC XML file.",
    )

    validation_group = parser.add_mutually_exclusive_group(required=False)
    validation_group.add_argument("--validation-file", type=Path, help="Optional validation JSON/JSONL/CSV file.")
    validation_group.add_argument("--validation-xml", type=Path, help="Optional validation BioC XML file.")

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("recognition/models/pubmed_all"),
        help="Directory where the trained model will be saved.",
    )
    parser.add_argument(
        "--model-name-or-path",
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        help="Base Hugging Face model to fine-tune.",
    )
    parser.add_argument("--tokenizer-name", default=None, help="Optional tokenizer path/name override.")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--token", default=None, help="Optional Hugging Face auth token.")
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--num-train-epochs", default="15")
    parser.add_argument("--learning-rate", default="3e-05")
    parser.add_argument("--weight-decay", default="0.0")
    parser.add_argument("--save-strategy", default="steps", choices=["no", "steps", "epoch"])
    parser.add_argument("--save-steps", default="1000")
    parser.add_argument("--save-total-limit", default="3")
    parser.add_argument("--logging-steps", default="50")

    parser.add_argument("--per-device-train-batch-size", default="8")
    parser.add_argument("--per-device-eval-batch-size", default=None)
    parser.add_argument("--max-seq-length", default=None)
    parser.add_argument("--doc-stride", default="128")
    parser.add_argument("--preprocessing-num-workers", default=None)
    parser.add_argument("--alignment-mode", default="expand", choices=["strict", "expand", "skip"])
    parser.add_argument(
        "--overlap-policy",
        default="last",
        choices=["last", "error"],
        help="How to resolve overlapping gold spans during training.",
    )
    parser.add_argument("--seed", default="42")

    parser.add_argument("--evaluation-strategy", default="no", choices=["no", "steps", "epoch"])
    parser.add_argument("--eval-steps", default=None)
    parser.add_argument("--load-best-model-at-end", action="store_true")
    parser.add_argument("--metric-for-best-model", default=None)
    parser.add_argument("--greater-is-better", default=None, choices=["true", "false"])

    parser.add_argument("--pad-to-max-length", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()
    if args.train_file is None and args.train_xml is None:
        args.train_xml = Path("dataset/train_all.xml")
    return args


def print_runtime_info(output_dir: Path) -> None:
    print(f"Output dir: {output_dir}")
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        print(f"CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
    try:
        import torch

        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        print(f"torch.cuda.is_available() = {gpu_available}")
        print(f"GPU count = {gpu_count}")
        if gpu_available:
            for idx in range(gpu_count):
                print(f"GPU {idx}: {torch.cuda.get_device_name(idx)}")
    except ImportError:
        print("torch is not installed in this environment; skipping GPU check.")


def append_optional_arg(cmd: list[str], flag: str, value) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def resolve_eval_strategy_flag() -> str:
    try:
        from transformers import TrainingArguments
        field_names = set(getattr(TrainingArguments, "__dataclass_fields__", {}).keys())
        if "eval_strategy" in field_names:
            return "--eval_strategy"
    except Exception:
        pass
    return "--evaluation_strategy"


def format_command_for_logging(cmd: list[str]) -> str:
    redacted: list[str] = []
    i = 0
    while i < len(cmd):
        part = cmd[i]
        redacted.append(part)
        if part == "--token" and i + 1 < len(cmd):
            redacted.append("***REDACTED***")
            i += 2
            continue
        i += 1
    return " ".join(redacted)


def main() -> int:
    args = parse_args()

    if args.train_file is not None and not args.train_file.is_file():
        raise FileNotFoundError(f"Missing training file: {args.train_file}")
    if args.train_xml is not None and not args.train_xml.is_file():
        raise FileNotFoundError(f"Missing training XML: {args.train_xml}")
    if args.validation_file is not None and not args.validation_file.is_file():
        raise FileNotFoundError(f"Missing validation file: {args.validation_file}")
    if args.validation_xml is not None and not args.validation_xml.is_file():
        raise FileNotFoundError(f"Missing validation XML: {args.validation_xml}")

    if args.train_file is not None:
        args.train_file = args.train_file.resolve()
    if args.train_xml is not None:
        args.train_xml = args.train_xml.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.validation_file is not None or args.validation_xml is not None:
        print("Validation input was provided.")
        args.validation_file = None
        args.validation_xml = None
    args.evaluation_strategy = "no"
    args.eval_steps = None
    args.load_best_model_at_end = False
    args.metric_for_best_model = None
    args.greater_is_better = None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print_runtime_info(args.output_dir)

    cmd = [
        sys.executable,
        "-u",
        str(RUN_NER),
        "--model_name_or_path",
        args.model_name_or_path,
        "--output_dir",
        str(args.output_dir),
        "--num_train_epochs",
        str(args.num_train_epochs),
        "--learning_rate",
        str(args.learning_rate),
        "--weight_decay",
        str(args.weight_decay),
        "--save_strategy",
        str(args.save_strategy),
        "--save_steps",
        str(args.save_steps),
        "--save_total_limit",
        str(args.save_total_limit),
        "--logging_steps",
        str(args.logging_steps),
        "--per_device_train_batch_size",
        str(args.per_device_train_batch_size),
        "--doc_stride",
        str(args.doc_stride),
        "--alignment_mode",
        str(args.alignment_mode),
        "--overlap_policy",
        str(args.overlap_policy),
        "--seed",
        str(args.seed),
        "--report_to",
        "none",
    ]

    if args.train_file is not None:
        cmd.extend(["--train_file", str(args.train_file)])
    else:
        cmd.extend(["--train_xml", str(args.train_xml)])

    append_optional_arg(cmd, "--tokenizer_name", args.tokenizer_name)
    append_optional_arg(cmd, "--cache_dir", args.cache_dir)
    append_optional_arg(cmd, "--model_revision", args.model_revision)
    append_optional_arg(cmd, "--token", args.token)
    append_optional_arg(cmd, "--per_device_eval_batch_size", args.per_device_eval_batch_size)
    append_optional_arg(cmd, "--max_seq_length", args.max_seq_length)
    append_optional_arg(cmd, "--preprocessing_num_workers", args.preprocessing_num_workers)
    append_optional_arg(cmd, resolve_eval_strategy_flag(), args.evaluation_strategy)
    append_optional_arg(cmd, "--eval_steps", args.eval_steps)
    append_optional_arg(cmd, "--metric_for_best_model", args.metric_for_best_model)

    if args.greater_is_better is not None:
        cmd.extend(["--greater_is_better", "true" if args.greater_is_better == "true" else "false"])

    if args.load_best_model_at_end:
        cmd.append("--load_best_model_at_end")
    if args.pad_to_max_length:
        cmd.append("--pad_to_max_length")
    if args.fp16:
        cmd.append("--fp16")
    if args.overwrite_output_dir:
        cmd.append("--overwrite_output_dir")
    if args.trust_remote_code:
        cmd.append("--trust_remote_code")
    if args.push_to_hub:
        cmd.append("--push_to_hub")

    print("Launching offset-based fine-tuning...")
    print("Command:", format_command_for_logging(cmd))

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.run(cmd, cwd=PROJECT_DIR, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
