"""Run offset-based NER prediction from BioC XML."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from transformers import TrainingArguments
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

try:
    from .ner_common import (
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
except ImportError:  # pragma: no cover
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
        description="Run NER prediction from BioC XML and optionally write predicted BioC XML."
    )
    parser.add_argument("--model-path", type=Path, required=True, help="Fine-tuned model directory or Hub id.")
    parser.add_argument("--tokenizer-path", type=str, default=None, help="Optional tokenizer path if different from model path.")
    parser.add_argument("--input-xml", type=Path, required=True, help="Input BioC XML file to convert and predict.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where prediction artifacts will be written.")
    parser.add_argument("--output-xml", type=Path, default=None, help="Optional output BioC XML path.")
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


def build_model_options(
    *,
    model_path,
    tokenizer_path=None,
    cache_dir=None,
    model_revision="main",
    token=None,
    trust_remote_code=False,
) -> ModelOptions:
    options = ModelOptions(
        model_name_or_path=str(model_path),
        tokenizer_name=tokenizer_path,
        cache_dir=cache_dir,
        model_revision=model_revision,
        token=token,
        trust_remote_code=trust_remote_code,
    )
    normalize_auth_arguments(options)
    return options


def build_prediction_training_args(*, output_dir, per_device_predict_batch_size=16, fp16=False) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        do_predict=True,
        per_device_eval_batch_size=per_device_predict_batch_size,
        remove_unused_columns=False,
        report_to=[],
        fp16=fp16,
    )


def run_prediction(
    *,
    model_path,
    input_xml,
    output_dir,
    output_xml=None,
    tokenizer_path=None,
    max_seq_length=None,
    doc_stride=128,
    pad_to_max_length=False,
    max_predict_samples=None,
    warmup_runs=1,
    per_device_predict_batch_size=16,
    preprocessing_num_workers=None,
    overwrite_cache=False,
    cache_dir=None,
    model_revision="main",
    token=None,
    trust_remote_code=False,
    fp16=False,
) -> int:
    output_dir = Path(output_dir)
    input_xml = Path(input_xml)
    output_xml = Path(output_xml) if output_xml is not None else None

    output_dir.mkdir(parents=True, exist_ok=True)

    if warmup_runs < 0:
        raise ValueError("`--warmup-runs` must be >= 0.")
    if doc_stride < 0:
        raise ValueError("`--doc-stride` must be >= 0.")
    if not input_xml.is_file():
        raise FileNotFoundError(f"Missing input XML: {input_xml}")
    if output_xml is not None:
        output_xml.parent.mkdir(parents=True, exist_ok=True)

    training_args = build_prediction_training_args(
        output_dir=output_dir,
        per_device_predict_batch_size=per_device_predict_batch_size,
        fp16=fp16,
    )
    configure_logging(training_args)
    model_options = build_model_options(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        cache_dir=cache_dir,
        model_revision=model_revision,
        token=token,
        trust_remote_code=trust_remote_code,
    )

    test_file = maybe_convert_bioc_xml(
        str(input_xml),
        str(output_dir / GENERATED_JSON_FILENAMES["test"]),
        include_entities=False,
    )

    raw_datasets = load_raw_datasets({"test": test_file}, cache_dir=model_options.cache_dir)
    inferred_columns = infer_columns(raw_datasets, None, None)
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

    effective_max_seq_length = resolve_effective_max_seq_length(tokenizer, config, max_seq_length)
    print(f"Using max_seq_length={effective_max_seq_length}")
    padding = "max_length" if pad_to_max_length else False

    with training_args.main_process_first(desc="tokenize prediction dataset"):
        raw_predict_subset, predict_dataset = preprocess_prediction_dataset(
            raw_dataset=raw_datasets["test"],
            columns=columns,
            tokenizer=tokenizer,
            max_seq_length=effective_max_seq_length,
            stride=doc_stride,
            padding=padding,
            num_proc=preprocessing_num_workers,
            overwrite_cache=overwrite_cache,
            limit=max_predict_samples,
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

    for warmup_index in range(warmup_runs):
        print(f"Warmup run {warmup_index + 1}/{warmup_runs}")
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
    save_prediction_outputs(str(output_dir), prediction_entries)

    runtime_summary_path = output_dir / RUNTIME_SUMMARY_FILENAME
    runtime_summary_path.write_text(
        json.dumps(
            {
                "elapsed_seconds": elapsed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if output_xml is not None:
        write_predictions_to_bioc_xml(input_xml, output_xml, prediction_entries)
        print(f"Wrote BioC XML predictions to {output_xml}")

    return 0


def main() -> int:
    args = parse_args()
    return run_prediction(
        model_path=args.model_path,
        input_xml=args.input_xml,
        output_dir=args.output_dir,
        output_xml=args.output_xml,
        tokenizer_path=args.tokenizer_path,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        pad_to_max_length=args.pad_to_max_length,
        max_predict_samples=args.max_predict_samples,
        warmup_runs=args.warmup_runs,
        per_device_predict_batch_size=args.per_device_predict_batch_size,
        preprocessing_num_workers=args.preprocessing_num_workers,
        overwrite_cache=args.overwrite_cache,
        cache_dir=args.cache_dir,
        model_revision=args.model_revision,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
        fp16=args.fp16,
    )


if __name__ == "__main__":
    raise SystemExit(main())
