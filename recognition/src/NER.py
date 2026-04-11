#!/usr/bin/env python
# coding=utf-8
"""Train/fine-tune an offset-based token-classification model for biomedical NER.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser, TrainingArguments, set_seed
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from ner_common import (
    GENERATED_JSON_FILENAMES,
    ModelOptions,
    TokenClassificationBatchCollator,
    build_label_schema,
    build_trainer,
    configure_logging,
    has_active_eval_strategy,
    infer_columns,
    infer_last_checkpoint,
    load_fast_tokenizer,
    load_model_config,
    load_raw_datasets,
    load_token_classification_model,
    maybe_align_label_schema_to_model,
    maybe_convert_bioc_xml,
    maybe_log_and_save_metrics,
    normalize_auth_arguments,
    normalize_training_args_for_train_only,
    preprocess_train_dataset,
    resolve_effective_max_seq_length,
)


check_min_version("4.37.0")
require_version("datasets>=1.8.0", "Please install a compatible `datasets` version for this project.")


@dataclass
class TrainDataOptions:
    task_name: str = field(default="ner", metadata={"help": "Task name."})
    train_file: Optional[str] = field(default=None, metadata={"help": "Training JSON/JSONL/CSV file."})
    train_xml: Optional[str] = field(default=None, metadata={"help": "Training BioC XML file to convert internally."})
    validation_file: Optional[str] = field(default=None, metadata={"help": "Optional validation JSON/JSONL/CSV file."})
    validation_xml: Optional[str] = field(default=None, metadata={"help": "Optional validation BioC XML file to convert internally."})
    text_column_name: Optional[str] = field(default=None, metadata={"help": "Name of the raw text column."})
    entities_column_name: Optional[str] = field(default=None, metadata={"help": "Name of the entity annotation column."})
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite cached dataset preprocessing."})
    preprocessing_num_workers: Optional[int] = field(default=None, metadata={"help": "Dataset map workers."})
    max_seq_length: Optional[int] = field(default=None, metadata={"help": "Maximum tokenizer sequence length."})
    doc_stride: int = field(default=128, metadata={"help": "Overlap in tokens between overflowing chunks."})
    pad_to_max_length: bool = field(default=False, metadata={"help": "Pad all sequences to max length."})
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "Optional cap on train examples."})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "Optional cap on validation examples."})
    alignment_mode: str = field(
        default="expand",
        metadata={"help": "How to handle entity spans that do not align perfectly to tokenizer offsets: strict, expand, or skip."},
    )
    overlap_policy: str = field(
        default="last",
        metadata={"help": "How to handle overlapping gold spans for flat BIO tagging: last or error."},
    )

    def __post_init__(self) -> None:
        self.task_name = self.task_name.lower()
        self.alignment_mode = self.alignment_mode.lower()
        self.overlap_policy = self.overlap_policy.lower()
        if self.alignment_mode not in {"strict", "expand", "skip"}:
            raise ValueError("alignment_mode must be one of: strict, expand, skip.")
        if self.overlap_policy not in {"last", "error"}:
            raise ValueError("overlap_policy must be one of: last, error.")
        if self.doc_stride < 0:
            raise ValueError("doc_stride must be >= 0.")

        for field_name in ("train_file", "validation_file"):
            value = getattr(self, field_name)
            if value is None:
                continue
            extension = value.rsplit(".", 1)[-1].lower()
            if extension not in {"csv", "json", "jsonl"}:
                raise ValueError(f"`{field_name}` should be csv, json, or jsonl. Got: {value}")


def parse_args() -> tuple[ModelOptions, TrainDataOptions, TrainingArguments]:
    parser = HfArgumentParser((ModelOptions, TrainDataOptions, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    return parser.parse_args_into_dataclasses()


def prepare_train_files(data_options: TrainDataOptions, training_args: TrainingArguments) -> None:
    if data_options.train_file and data_options.train_xml:
        raise ValueError("Specify only one of --train_file or --train_xml.")
    if data_options.validation_file and data_options.validation_xml:
        raise ValueError("Specify only one of --validation_file or --validation_xml.")
    if not data_options.train_file and not data_options.train_xml:
        raise ValueError("Training requires --train_file or --train_xml.")

    if data_options.train_file is None:
        data_options.train_file = maybe_convert_bioc_xml(
            data_options.train_xml,
            os.path.join(training_args.output_dir, GENERATED_JSON_FILENAMES["train"]),
            include_entities=True,
            overlap_policy=data_options.overlap_policy,
        )

    if data_options.validation_file is None and data_options.validation_xml is not None:
        data_options.validation_file = maybe_convert_bioc_xml(
            data_options.validation_xml,
            os.path.join(training_args.output_dir, GENERATED_JSON_FILENAMES["validation"]),
            include_entities=True,
            overlap_policy=data_options.overlap_policy,
        )


def main() -> None:
    model_options, data_options, training_args = parse_args()
    normalize_auth_arguments(model_options)

    training_args.do_train = True
    last_checkpoint = infer_last_checkpoint(training_args)
    prepare_train_files(data_options, training_args)

    has_validation_data = data_options.validation_file is not None
    use_validation_eval = has_validation_data and (
        training_args.do_eval or has_active_eval_strategy(training_args) or training_args.load_best_model_at_end
    )
    normalize_training_args_for_train_only(training_args, has_validation_data=has_validation_data)
    configure_logging(training_args)

    if has_validation_data and not use_validation_eval:
        print("Validation data was provided.")

    set_seed(training_args.seed)

    data_files = {"train": data_options.train_file}
    if has_validation_data:
        data_files["validation"] = data_options.validation_file
    raw_datasets = load_raw_datasets(data_files, cache_dir=model_options.cache_dir)

    columns = infer_columns(raw_datasets, data_options.text_column_name, data_options.entities_column_name)
    if columns.entities is None:
        raise ValueError("Could not infer the entity column for training.")

    label_schema = build_label_schema(
        raw_datasets,
        columns.entities,
        source_splits=("train",),
        overlap_policy=data_options.overlap_policy,
    )
    config = load_model_config(model_options, data_options.task_name, label_schema)
    tokenizer = load_fast_tokenizer(model_options, config)
    model = load_token_classification_model(model_options, config)
    label_schema = maybe_align_label_schema_to_model(label_schema, model.config)

    model.config.label2id = {label: index for index, label in enumerate(label_schema.label_list)}
    model.config.id2label = {index: label for index, label in enumerate(label_schema.label_list)}

    effective_max_seq_length = resolve_effective_max_seq_length(tokenizer, config, data_options.max_seq_length)
    print(f"Using max_seq_length={effective_max_seq_length}")
    padding = "max_length" if data_options.pad_to_max_length else False

    with training_args.main_process_first(desc="tokenize train dataset"):
        raw_train_subset, train_dataset = preprocess_train_dataset(
            raw_dataset=raw_datasets["train"],
            columns=columns,
            tokenizer=tokenizer,
            label_schema=label_schema,
            max_seq_length=effective_max_seq_length,
            stride=data_options.doc_stride,
            padding=padding,
            alignment_mode=data_options.alignment_mode,
            overlap_policy=data_options.overlap_policy,
            num_proc=data_options.preprocessing_num_workers,
            overwrite_cache=data_options.overwrite_cache,
            limit=data_options.max_train_samples,
            training_args=training_args,
        )

    raw_eval_subset = None
    eval_dataset = None
    if use_validation_eval:
        if "validation" not in raw_datasets:
            raise ValueError("Evaluation was requested, but no validation dataset was provided.")
        with training_args.main_process_first(desc="tokenize validation dataset"):
            raw_eval_subset, eval_dataset = preprocess_train_dataset(
                raw_dataset=raw_datasets["validation"],
                columns=columns,
                tokenizer=tokenizer,
                label_schema=label_schema,
                max_seq_length=effective_max_seq_length,
                stride=data_options.doc_stride,
                padding=padding,
                alignment_mode=data_options.alignment_mode,
                overlap_policy=data_options.overlap_policy,
                num_proc=data_options.preprocessing_num_workers,
                overwrite_cache=data_options.overwrite_cache,
                limit=data_options.max_eval_samples,
                training_args=training_args,
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
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    checkpoint = training_args.resume_from_checkpoint or last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

    train_metrics = dict(train_result.metrics)
    train_metrics["train_samples"] = len(raw_train_subset)
    maybe_log_and_save_metrics(trainer, "train", train_metrics)
    trainer.save_state()

    if eval_dataset is not None and raw_eval_subset is not None:
        eval_metrics = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix="eval")
        eval_metrics["eval_samples"] = len(raw_eval_subset)
        maybe_log_and_save_metrics(trainer, "eval", eval_metrics)

    card_kwargs = {"finetuned_from": model_options.model_name_or_path, "tasks": "token-classification"}
    if data_options.train_file is not None:
        card_kwargs["dataset"] = os.path.basename(data_options.train_file)
    if training_args.push_to_hub:
        trainer.push_to_hub(**card_kwargs)
    else:
        trainer.create_model_card(**card_kwargs)


if __name__ == "__main__":
    main()
