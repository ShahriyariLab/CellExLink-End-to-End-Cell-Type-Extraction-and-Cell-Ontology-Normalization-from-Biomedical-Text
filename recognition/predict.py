
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Union


PathLike = Union[str, Path]

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
PREDICT_NER = SRC_DIR / "predict_NER.py"


def _resolve_model_reference(model_reference: PathLike) -> str:
    model_reference_str = str(model_reference)
    candidate = Path(model_reference_str)
    if candidate.exists():
        return str(candidate.resolve())
    return model_reference_str


def _format_command_for_logging(cmd: list[str]) -> str:
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


def predict_ner(
    *,
    model_path: PathLike,
    input_xml: Optional[PathLike] = None,
    input_file: Optional[PathLike] = None,
    output_dir: PathLike,
    output_xml: Optional[PathLike] = None,
    tokenizer_path: Optional[PathLike] = None,
    max_seq_length: Optional[int] = None,
    doc_stride: int = 128,
    pad_to_max_length: bool = False,
    max_predict_samples: Optional[int] = None,
    warmup_runs: int = 1,
    per_device_predict_batch_size: int = 16,
    preprocessing_num_workers: Optional[int] = None,
    overwrite_cache: bool = False,
    cache_dir: Optional[PathLike] = None,
    model_revision: str = "main",
    token: Optional[str] = None,
    trust_remote_code: bool = False,
    fp16: bool = False,
) -> int:
    """Run NER prediction using the raw-text, offset-based pipeline.
    """
    if (input_xml is None) == (input_file is None):
        raise ValueError("Provide exactly one of `input_xml` or `input_file`.")
    if output_xml is not None and input_xml is None:
        raise ValueError("`output_xml` requires `input_xml`, because BioC export needs the original XML structure.")
    if warmup_runs < 0:
        raise ValueError("`warmup_runs` must be >= 0.")
    if doc_stride < 0:
        raise ValueError("`doc_stride` must be >= 0.")

    model_name_or_path = _resolve_model_reference(model_path)
    output_dir_path = Path(output_dir).resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        str(PREDICT_NER),
        "--model-path",
        model_name_or_path,
        "--output-dir",
        str(output_dir_path),
        "--warmup-runs",
        str(warmup_runs),
        "--per-device-predict-batch-size",
        str(per_device_predict_batch_size),
        "--model-revision",
        str(model_revision),
        "--doc-stride",
        str(doc_stride),
    ]

    if input_xml is not None:
        input_xml_path = Path(input_xml).resolve()
        if not input_xml_path.is_file():
            raise FileNotFoundError(f"Missing input XML: {input_xml_path}")
        cmd.extend(["--input-xml", str(input_xml_path)])
    else:
        input_file_path = Path(input_file).resolve()
        if not input_file_path.is_file():
            raise FileNotFoundError(f"Missing input file: {input_file_path}")
        cmd.extend(["--input-file", str(input_file_path)])

    if output_xml is not None:
        output_xml_path = Path(output_xml).resolve()
        output_xml_path.parent.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--output-xml", str(output_xml_path)])

    if tokenizer_path is not None:
        cmd.extend(["--tokenizer-path", str(tokenizer_path)])
    if max_seq_length is not None:
        cmd.extend(["--max-seq-length", str(max_seq_length)])
    if pad_to_max_length:
        cmd.append("--pad-to-max-length")
    if max_predict_samples is not None:
        cmd.extend(["--max-predict-samples", str(max_predict_samples)])
    if preprocessing_num_workers is not None:
        cmd.extend(["--preprocessing-num-workers", str(preprocessing_num_workers)])
    if overwrite_cache:
        cmd.append("--overwrite-cache")
    if cache_dir is not None:
        cmd.extend(["--cache-dir", str(cache_dir)])
    if token is not None:
        cmd.extend(["--token", str(token)])
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    if fp16:
        cmd.append("--fp16")

    if not PREDICT_NER.is_file():
        raise FileNotFoundError(f"Missing predict script: {PREDICT_NER}")

    print("Launching prediction...")
    print("Command:", _format_command_for_logging(cmd))
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.run(cmd, cwd=SRC_DIR, env=env, check=False).returncode
