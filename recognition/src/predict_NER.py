#!/usr/bin/env python
# coding=utf-8
"""Run offset-based NER prediction from raw passage text and optionally export BioC XML.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from transformers import TrainingArguments
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from ner_common import (
    GENERATED_JSON_FILENAMES,
    DatasetColumns,
    ModelOptions,
    TokenClassificationBatchCollator,
    build_label_schema_from_model,
    build_trainer,
    configure_logging,
    infer_columns,
    load_fast_tokenizer,
    load_model_config,
    load_raw_datasets,
    load_token_classification_model,
    maybe_convert_bioc_xml,
    normalize_auth_arguments,
    preprocess_prediction_dataset,
    reconstruct_prediction_outputs,
    resolve_effective_max_seq_length,
    save_prediction_outputs,
    write_predictions_to_bioc_xml,
)


check_min_version("4.37.0")
require_version("datasets>=1.8.0", "Please install a compatible `datasets` version for this project.")

RUNTIME_SUMMARY_FILENAME = "predict_runtime_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NER prediction and optionally convert predictions back to BioC XML."
    )
    parser.add_argument("--model-path", type=Path, required=True, help="Fine-tuned model directory or Hub id.")
    parser.add_argument("--tokenizer-path", type=str, default=None, help="Optional tokenizer path if different from model path.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-xml", type=Path, help="Input BioC XML file to convert and predict.")
    input_group.add_argument("--input-file", type=Path, help="Input JSON/JSONL/CSV file with raw passage text.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where prediction artifacts will be written.")
    parser.add_argument("--output-xml", type=Path, default=None, help="Optional output BioC XML path.")
    parser.add_argument("--text-column-name", default=None, help="Optional raw text column name override.")
    parser.add_argument("--max-seq-length", type=int, default=None, help="Optional tokenizer max sequence length.")
    parser.add_argument("--doc-stride", type=int, default=128, help="Overlap in tokens between overflowing chunks.")
    parser.add_argument("--pad-to-max-length", action="store_true", help="Pad every batch item to max length.")
    parser.add_argument("--max-predict-samples", type=int, default=None, help="Optional cap on prediction examples.")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Number of untimed warmup predict runs.")
    parser.add_argument("--per-device-predict-batch-size", type=int, default=16, help="Prediction batch size per device.")
    parser.add_argument("--preprocessing-num-workers", type=int, default=None, help="Dataset map workers.")
    parser.add_argument("--overwrite-cache", action="store_true", help="Overwrite cached dataset preprocessing.")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--model-revision", default="main", help="Model revision to use when loading from the Hub.")
    parser.add_argument("--token", default=None, help="Optional Hugging Face auth token.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom code from the model repo.")
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 prediction if supported by the hardware.")
    return parser.parse_args()


def build_model_options(args: argparse.Namespace) -> ModelOptions:
    options = ModelOptions(
        model_name_or_path=str(args.model_path),
        tokenizer_name=args.tokenizer_path,
        cache_dir=args.cache_dir,
        model_revision=args.model_revision,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
    )
    normalize_auth_arguments(options)
    return options


def build_prediction_training_args(args: argparse.Namespace) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(args.output_dir),
        do_predict=True,
        per_device_eval_batch_size=args.per_device_predict_batch_size,
        remove_unused_columns=False,
        report_to=[],
        fp16=args.fp16,
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.output_xml is not None and args.input_xml is None:
        raise ValueError("`--output-xml` requires `--input-xml`, because the original BioC structure is needed.")
    if args.warmup_runs < 0:
        raise ValueError("`--warmup-runs` must be >= 0.")
    if args.doc_stride < 0:
        raise ValueError("`--doc-stride` must be >= 0.")

    training_args = build_prediction_training_args(args)
    configure_logging(training_args)
    model_options = build_model_options(args)

    test_file = str(args.input_file) if args.input_file is not None else None
    if test_file is None:
        test_file = maybe_convert_bioc_xml(
            str(args.input_xml),
            str(args.output_dir / GENERATED_JSON_FILENAMES["test"]),
            include_entities=False,
        )

    raw_datasets = load_raw_datasets({"test": test_file}, cache_dir=model_options.cache_dir)
    inferred_columns = infer_columns(raw_datasets, args.text_column_name, None)
    columns = DatasetColumns(
        text=inferred_columns.text,
        entities=None,
        document_id=inferred_columns.document_id,
        passage_id=inferred_columns.passage_id,
        passage_offset=inferred_columns.passage_offset,
        record_id=inferred_columns.record_id,
    )

    config = load_model_config(model_options, task_name="ner", label_schema=None)
    label_schema = build_label_schema_from_model(config)
    tokenizer = load_fast_tokenizer(model_options, config)
    model = load_token_classification_model(model_options, config)

    effective_max_seq_length = resolve_effective_max_seq_length(tokenizer, config, args.max_seq_length)
    print(f"Using max_seq_length={effective_max_seq_length}")
    padding = "max_length" if args.pad_to_max_length else False

    with training_args.main_process_first(desc="tokenize prediction dataset"):
        raw_predict_subset, predict_dataset = preprocess_prediction_dataset(
            raw_dataset=raw_datasets["test"],
            columns=columns,
            tokenizer=tokenizer,
            max_seq_length=effective_max_seq_length,
            stride=args.doc_stride,
            padding=padding,
            num_proc=args.preprocessing_num_workers,
            overwrite_cache=args.overwrite_cache,
            limit=args.max_predict_samples,
        )

    data_collator = TokenClassificationBatchCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )
    trainer = build_trainer(
        model=model,
        training_args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
        train_dataset=None,
        eval_dataset=None,
    )

    try:
        import torch
    except Exception:  # pragma: no cover
        torch = None

    for warmup_index in range(args.warmup_runs):
        print(f"Warmup run {warmup_index + 1}/{args.warmup_runs}")
        _ = trainer.predict(predict_dataset, metric_key_prefix=f"warmup_{warmup_index + 1}")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()

    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    predict_start_time = time.perf_counter()
    predict_output = trainer.predict(predict_dataset, metric_key_prefix="predict")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - predict_start_time

    trainer.save_metrics("predict", dict(predict_output.metrics))

    prediction_entries = reconstruct_prediction_outputs(
        raw_predict_dataset=raw_predict_subset,
        tokenized_predict_dataset=predict_dataset,
        prediction_logits=np.asarray(predict_output.predictions),
        label_schema=label_schema,
        tokenizer=tokenizer,
        columns=columns,
    )
    save_prediction_outputs(str(args.output_dir), prediction_entries)

    runtime_summary_path = args.output_dir / RUNTIME_SUMMARY_FILENAME
    runtime_summary_path.write_text(
        json.dumps(
            {
                "elapsed_seconds": elapsed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.output_xml is not None:
        write_predictions_to_bioc_xml(args.input_xml, args.output_xml, prediction_entries)
        print(f"Wrote BioC XML predictions to {args.output_xml}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
