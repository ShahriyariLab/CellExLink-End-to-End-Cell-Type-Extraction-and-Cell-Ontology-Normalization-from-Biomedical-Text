
from __future__ import annotations

import inspect
import json
import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import datasets
from datasets import Dataset, DatasetDict, load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

LOGGER = logging.getLogger(__name__)

IGNORE_INDEX = -100
REASONABLE_MAX_LENGTH_FALLBACK = 512
INTERNAL_EXAMPLE_INDEX_COLUMN = "__example_index__"

GENERATED_JSON_FILENAMES = {
    "train": "train.hf.jsonl",
    "validation": "validation.hf.jsonl",
    "test": "test.hf.jsonl",
}

TOKENIZER_TYPES_REQUIRING_PREFIX_SPACE = {"bloom", "deberta", "gpt2", "roberta"}


@dataclass
class ModelOptions:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."}
    )
    config_name: Optional[str] = field(default=None, metadata={"help": "Optional config name/path."})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "Optional tokenizer name/path."})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Directory used to cache Hugging Face assets."})
    model_revision: str = field(default="main", metadata={"help": "Model revision to use."})
    token: Optional[str] = field(default=None, metadata={"help": "HF auth token for remote files."})
    use_auth_token: Optional[bool] = field(default=None, metadata={"help": "Deprecated. Use `token` instead."})
    trust_remote_code: bool = field(default=False, metadata={"help": "Allow custom code from model repo."})
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Allow classifier head size mismatches when loading a checkpoint."},
    )


@dataclass
class DatasetColumns:
    text: str
    entities: Optional[str]
    document_id: Optional[str]
    passage_id: Optional[str]
    passage_offset: Optional[str]
    record_id: Optional[str]


@dataclass
class LabelSchema:
    label_list: List[str]
    label_to_id: Dict[str, int]
    id_to_label: Dict[int, str]

    def encode(self, label: str) -> int:
        try:
            return self.label_to_id[str(label)]
        except KeyError as exc:
            raise KeyError(f"Unknown label {label!r}. Known labels: {self.label_list}") from exc


@dataclass(frozen=True)
class Entity:
    start: int
    end: int
    label: str

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"Invalid entity span: {self.start}..{self.end}")


class TokenClassificationBatchCollator(DataCollatorForTokenClassification):
    """Drop metadata fields before padding/model dispatch."""

    ignored_feature_keys = {
        "sample_index",
        "offset_mapping",
        "special_tokens_mask",
        "text",
        "document_id",
        "passage_id",
        "passage_offset",
        "id",
    }

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        sanitized = [
            {key: value for key, value in feature.items() if key not in self.ignored_feature_keys}
            for feature in features
        ]
        return super().__call__(sanitized)


def normalize_auth_arguments(model_options: ModelOptions) -> None:
    if model_options.use_auth_token is not None:
        warnings.warn("`use_auth_token` is deprecated. Use `token` instead.", FutureWarning)
        if model_options.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only `token`.")
        model_options.token = model_options.use_auth_token


def configure_logging(training_args: TrainingArguments) -> None:
    import sys

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    LOGGER.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    LOGGER.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, fp16: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        training_args.parallel_mode.value == "distributed",
        training_args.fp16,
    )
    LOGGER.info("Training arguments: %s", training_args)


def maybe_convert_bioc_xml(
    xml_path: Optional[str],
    output_path: str,
    *,
    include_entities: bool,
    overlap_policy: str = "last",
) -> Optional[str]:
    if xml_path is None:
        return None

    from convert_bioc_to_json import convert_bioc_to_json

    LOGGER.info(
        "Converting BioC XML to JSONL (include_entities=%s, overlap_policy=%s): %s -> %s",
        include_entities,
        overlap_policy,
        xml_path,
        output_path,
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    convert_bioc_to_json(
        [xml_path],
        output_path,
        include_entities=include_entities,
        overlap_policy=overlap_policy,
    )
    return output_path


def load_raw_datasets(data_files: Dict[str, str], cache_dir: Optional[str]) -> DatasetDict:
    if not data_files:
        raise ValueError("No dataset files were provided.")
    first_path = next(iter(data_files.values()))
    extension = first_path.rsplit(".", 1)[-1].lower()
    if extension == "jsonl":
        extension = "json"
    return load_dataset(extension, data_files=data_files, cache_dir=cache_dir)


def choose_reference_split(raw_datasets: DatasetDict) -> str:
    for split_name in ("train", "validation", "test"):
        if split_name in raw_datasets:
            return split_name
    raise ValueError("No dataset splits were loaded.")


def infer_columns(
    raw_datasets: DatasetDict,
    text_column_name: Optional[str],
    entities_column_name: Optional[str],
) -> DatasetColumns:
    reference_split = choose_reference_split(raw_datasets)
    column_names = raw_datasets[reference_split].column_names

    if text_column_name is not None:
        text_column = text_column_name
    elif "text" in column_names:
        text_column = "text"
    else:
        text_column = column_names[0]

    if entities_column_name is not None:
        entities_column = entities_column_name
    elif "entities" in column_names:
        entities_column = "entities"
    else:
        entities_column = None

    document_id_column = "document_id" if "document_id" in column_names else None
    passage_id_column = "passage_id" if "passage_id" in column_names else None
    passage_offset_column = "passage_offset" if "passage_offset" in column_names else None
    record_id_column = "id" if "id" in column_names else None

    return DatasetColumns(
        text=text_column,
        entities=entities_column,
        document_id=document_id_column,
        passage_id=passage_id_column,
        passage_offset=passage_offset_column,
        record_id=record_id_column,
    )


def entities_overlap(left: Entity, right: Entity) -> bool:
    return max(left.start, right.start) < min(left.end, right.end)


def validate_no_overlapping_entities(entities: Sequence[Entity]) -> None:
    for index, left in enumerate(entities):
        for right in entities[index + 1:]:
            if entities_overlap(left, right):
                raise ValueError(
                    "Overlapping entities are not representable in flat BIO tagging when overlap_policy='error': "
                    f"{left.label!r} at {left.start}-{left.end} overlaps {right.label!r} at {right.start}-{right.end}"
                )


def flatten_entities_for_flat_ner(
    entities: Sequence[Entity],
    *,
    overlap_policy: str = "last",
) -> List[Entity]:

    if overlap_policy not in {"last", "error"}:
        raise ValueError("overlap_policy must be one of: last, error.")
    entities_list = list(entities)
    if overlap_policy == "error":
        validate_no_overlapping_entities(entities_list)
    return entities_list


def canonicalize_entities(
    raw_value: Any,
    *,
    overlap_policy: str = "last",
) -> List[Entity]:
    if raw_value is None:
        return []

    value = raw_value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        value = json.loads(stripped)

    if not isinstance(value, list):
        raise ValueError(f"Expected `entities` to be a list. Got: {type(value).__name__}")

    entities: List[Entity] = []
    for item in value:
        if isinstance(item, Entity):
            entities.append(item)
            continue
        if not isinstance(item, dict):
            raise ValueError(f"Expected entity record to be a dict. Got: {item!r}")
        if "start" not in item or "end" not in item or "label" not in item:
            raise ValueError(f"Entity record is missing required keys: {item!r}")
        entities.append(
            Entity(
                start=int(item["start"]),
                end=int(item["end"]),
                label=str(item["label"]),
            )
        )

    return flatten_entities_for_flat_ner(entities, overlap_policy=overlap_policy)


def label_sort_key(label: str) -> Tuple[int, str]:
    if label == "O":
        return (0, label)
    if label.startswith("B-"):
        return (1, label[2:] + "\t0")
    if label.startswith("I-"):
        return (1, label[2:] + "\t1")
    return (2, label)


def build_label_schema(
    raw_datasets: DatasetDict,
    entities_column: str,
    source_splits: Sequence[str],
    *,
    overlap_policy: str = "last",
) -> LabelSchema:
    entity_types = set()
    for split_name in source_splits:
        if split_name not in raw_datasets:
            continue
        split_dataset = raw_datasets[split_name]
        if entities_column not in split_dataset.column_names:
            continue
        for raw_entities in split_dataset[entities_column]:
            for entity in canonicalize_entities(raw_entities, overlap_policy=overlap_policy):
                entity_types.add(entity.label)

    if not entity_types:
        raise ValueError(f"Could not infer labels from entity column `{entities_column}`.")

    label_list = ["O"]
    for entity_type in sorted(entity_types):
        label_list.append(f"B-{entity_type}")
        label_list.append(f"I-{entity_type}")

    label_list = sorted(label_list, key=label_sort_key)
    label_to_id = {label: index for index, label in enumerate(label_list)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    return LabelSchema(label_list=label_list, label_to_id=label_to_id, id_to_label=id_to_label)


def canonicalize_id2label_mapping(id2label: Dict[Any, str]) -> Dict[int, str]:
    return {int(index): str(label) for index, label in id2label.items()}


def build_label_schema_from_model(config: transformers.PretrainedConfig) -> LabelSchema:
    if not getattr(config, "id2label", None):
        raise ValueError("Prediction requires label mappings in the saved model config.")
    canonical = canonicalize_id2label_mapping(config.id2label)
    label_list = [canonical[index] for index in range(len(canonical))]
    label_to_id = {label: index for index, label in enumerate(label_list)}
    return LabelSchema(
        label_list=label_list,
        label_to_id=label_to_id,
        id_to_label={index: label for label, index in label_to_id.items()},
    )


def maybe_align_label_schema_to_model(label_schema: LabelSchema, config: transformers.PretrainedConfig) -> LabelSchema:
    existing_label2id = {str(label): int(index) for label, index in getattr(config, "label2id", {}).items()}
    default_label2id = transformers.PretrainedConfig(num_labels=len(label_schema.label_list)).label2id
    if not existing_label2id or existing_label2id == default_label2id:
        return label_schema

    if sorted(existing_label2id.keys()) != sorted(label_schema.label_list):
        LOGGER.warning(
            "Model label names do not match training label names. Continuing with training label order. "
            "Model labels: %s; training labels: %s",
            sorted(existing_label2id.keys()),
            sorted(label_schema.label_list),
        )
        return label_schema

    canonical = canonicalize_id2label_mapping(getattr(config, "id2label", {}))
    aligned_label_list = [canonical[index] for index in range(len(canonical))]
    LOGGER.info("Using label order from loaded model config: %s", aligned_label_list)
    return LabelSchema(
        label_list=aligned_label_list,
        label_to_id={label: index for index, label in enumerate(aligned_label_list)},
        id_to_label={index: label for index, label in enumerate(aligned_label_list)},
    )


def load_model_config(
    model_options: ModelOptions,
    task_name: str,
    label_schema: Optional[LabelSchema],
) -> transformers.PretrainedConfig:
    config_kwargs: Dict[str, Any] = {
        "finetuning_task": task_name,
        "cache_dir": model_options.cache_dir,
        "revision": model_options.model_revision,
        "token": model_options.token,
        "trust_remote_code": model_options.trust_remote_code,
    }
    if label_schema is not None:
        config_kwargs["num_labels"] = len(label_schema.label_list)

    return AutoConfig.from_pretrained(
        model_options.config_name or model_options.model_name_or_path,
        **config_kwargs,
    )


def load_fast_tokenizer(model_options: ModelOptions, config: transformers.PretrainedConfig) -> PreTrainedTokenizerFast:
    tokenizer_name_or_path = model_options.tokenizer_name or model_options.model_name_or_path
    tokenizer_kwargs: Dict[str, Any] = {
        "cache_dir": model_options.cache_dir,
        "use_fast": True,
        "revision": model_options.model_revision,
        "token": model_options.token,
        "trust_remote_code": model_options.trust_remote_code,
    }
    if getattr(config, "model_type", None) in TOKENIZER_TYPES_REQUIRING_PREFIX_SPACE:
        tokenizer_kwargs["add_prefix_space"] = True

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError("This project requires a fast tokenizer for offset-based alignment.")
    return tokenizer


def load_token_classification_model(
    model_options: ModelOptions,
    config: transformers.PretrainedConfig,
) -> transformers.PreTrainedModel:
    return AutoModelForTokenClassification.from_pretrained(
        model_options.model_name_or_path,
        from_tf=model_options.model_name_or_path.endswith(".ckpt"),
        config=config,
        cache_dir=model_options.cache_dir,
        revision=model_options.model_revision,
        token=model_options.token,
        trust_remote_code=model_options.trust_remote_code,
        ignore_mismatched_sizes=model_options.ignore_mismatched_sizes,
    )


def resolve_effective_max_seq_length(
    tokenizer: PreTrainedTokenizerFast,
    config: transformers.PretrainedConfig,
    requested_max_seq_length: Optional[int],
) -> int:
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if requested_max_seq_length is not None:
        if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 100_000 and requested_max_seq_length > tokenizer_limit:
            LOGGER.warning(
                "Requested max_seq_length=%d exceeds tokenizer.model_max_length=%d. Using %d instead.",
                requested_max_seq_length,
                tokenizer_limit,
                tokenizer_limit,
            )
            return tokenizer_limit
        return requested_max_seq_length

    candidates: List[int] = []
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 100_000:
        candidates.append(tokenizer_limit)

    position_limit = getattr(config, "max_position_embeddings", None)
    if isinstance(position_limit, int) and 0 < position_limit < 100_000:
        candidates.append(position_limit)

    if candidates:
        return min(candidates)

    LOGGER.warning(
        "Could not infer a reasonable model max length. Falling back to %d.",
        REASONABLE_MAX_LENGTH_FALLBACK,
    )
    return REASONABLE_MAX_LENGTH_FALLBACK


def select_subset(dataset: Dataset, limit: Optional[int]) -> Dataset:
    if limit is None:
        return dataset
    return dataset.select(range(min(len(dataset), limit)))


def attach_example_indices(dataset: Dataset) -> Dataset:
    if INTERNAL_EXAMPLE_INDEX_COLUMN in dataset.column_names:
        return dataset
    return dataset.add_column(INTERNAL_EXAMPLE_INDEX_COLUMN, list(range(len(dataset))))


def full_tokenize_texts(
    texts: Sequence[str],
    tokenizer: PreTrainedTokenizerFast,
) -> List[List[Tuple[int, int]]]:
    encodings = tokenizer(
        list(texts),
        add_special_tokens=False,
        return_offsets_mapping=True,
        padding=False,
        truncation=False,
    )
    return [
        [tuple(map(int, offset)) for offset in sample_offsets]
        for sample_offsets in encodings["offset_mapping"]
    ]


def build_full_token_label_map(
    offsets: Sequence[Tuple[int, int]],
    entities: Sequence[Entity],
    label_schema: LabelSchema,
    alignment_mode: str,
    *,
    overlap_policy: str = "last",
) -> Dict[Tuple[int, int], int]:
    if alignment_mode not in {"strict", "expand", "skip"}:
        raise ValueError(f"Unsupported alignment_mode={alignment_mode!r}. Use one of: strict, expand, skip.")
    if overlap_policy not in {"last", "error"}:
        raise ValueError("overlap_policy must be one of: last, error.")

    owner_by_offset: Dict[Tuple[int, int], str] = {}
    owner_label: Dict[str, str] = {}

    normalized_entities = flatten_entities_for_flat_ner(entities, overlap_policy=overlap_policy)
    for entity_index, entity in enumerate(normalized_entities):
        covered_offsets: List[Tuple[int, int]] = []
        misaligned = False

        for token_start, token_end in offsets:
            if token_end <= token_start:
                continue
            overlap = min(token_end, entity.end) - max(token_start, entity.start)
            if overlap <= 0:
                continue

            fully_inside = token_start >= entity.start and token_end <= entity.end
            if alignment_mode == "expand":
                covered_offsets.append((token_start, token_end))
            else:
                if fully_inside:
                    covered_offsets.append((token_start, token_end))
                else:
                    misaligned = True

        if alignment_mode == "strict" and misaligned:
            raise ValueError(
                f"Entity {entity.label!r} at {entity.start}-{entity.end} does not align to tokenizer offsets."
            )
        if alignment_mode == "skip" and misaligned:
            continue
        if not covered_offsets:
            if alignment_mode == "strict":
                raise ValueError(
                    f"Entity {entity.label!r} at {entity.start}-{entity.end} does not align to any tokenizer offsets."
                )
            continue

        owner_id = f"{entity_index}:{entity.label}:{entity.start}:{entity.end}"
        owner_label[owner_id] = entity.label

        if overlap_policy == "error":
            conflicting = [
                offset
                for offset in covered_offsets
                if offset in owner_by_offset and owner_by_offset[offset] != owner_id
            ]
            if conflicting:
                first = conflicting[0]
                raise ValueError(
                    "Overlapping entities are not representable in flat BIO tagging when overlap_policy='error': "
                    f"token span {first} receives multiple entities."
                )

        for offset in covered_offsets:
            owner_by_offset[offset] = owner_id

    label_map: Dict[Tuple[int, int], int] = {}
    previous_owner_id: Optional[str] = None
    for token_start, token_end in offsets:
        if token_end <= token_start:
            previous_owner_id = None
            continue
        offset = (token_start, token_end)
        owner_id = owner_by_offset.get(offset)
        if owner_id is None:
            previous_owner_id = None
            continue
        prefix = "I-" if owner_id == previous_owner_id else "B-"
        label_map[offset] = label_schema.encode(prefix + owner_label[owner_id])
        previous_owner_id = owner_id

    return label_map


def tokenize_labeled_examples(
    examples: Dict[str, List[Any]],
    *,
    tokenizer: PreTrainedTokenizerFast,
    text_column: str,
    entities_column: str,
    label_schema: LabelSchema,
    max_seq_length: int,
    stride: int,
    padding: str | bool,
    alignment_mode: str,
    overlap_policy: str,
) -> Dict[str, List[Any]]:
    texts = [str(text) for text in examples[text_column]]
    full_offsets_batch = full_tokenize_texts(texts, tokenizer=tokenizer)

    full_label_maps: List[Dict[Tuple[int, int], int]] = []
    for offsets, raw_entities in zip(full_offsets_batch, examples[entities_column]):
        entities = canonicalize_entities(raw_entities, overlap_policy=overlap_policy)
        full_label_maps.append(
            build_full_token_label_map(
                offsets=offsets,
                entities=entities,
                label_schema=label_schema,
                alignment_mode=alignment_mode,
                overlap_policy=overlap_policy,
            )
        )

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_seq_length,
        stride=stride,
        padding=padding,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )

    overflow_to_sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    special_tokens_masks = tokenized.pop("special_tokens_mask")
    offset_mappings = tokenized.pop("offset_mapping")

    labels: List[List[int]] = []
    for encoded_index, source_index in enumerate(overflow_to_sample_mapping):
        chunk_labels: List[int] = []
        full_label_map = full_label_maps[int(source_index)]
        for offset, is_special in zip(offset_mappings[encoded_index], special_tokens_masks[encoded_index]):
            token_start, token_end = int(offset[0]), int(offset[1])
            if is_special or token_end <= token_start:
                chunk_labels.append(IGNORE_INDEX)
                continue
            chunk_labels.append(full_label_map.get((token_start, token_end), label_schema.encode("O")))
        labels.append(chunk_labels)

    tokenized["labels"] = labels
    return tokenized


def tokenize_prediction_examples(
    examples: Dict[str, List[Any]],
    *,
    tokenizer: PreTrainedTokenizerFast,
    text_column: str,
    example_index_column: str,
    max_seq_length: int,
    stride: int,
    padding: str | bool,
) -> Dict[str, List[Any]]:
    texts = [str(text) for text in examples[text_column]]
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_seq_length,
        stride=stride,
        padding=padding,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    overflow_to_sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    tokenized["sample_index"] = [int(examples[example_index_column][index]) for index in overflow_to_sample_mapping]
    return tokenized


def preprocess_train_dataset(
    raw_dataset: Dataset,
    *,
    columns: DatasetColumns,
    tokenizer: PreTrainedTokenizerFast,
    label_schema: LabelSchema,
    max_seq_length: int,
    stride: int,
    padding: str | bool,
    alignment_mode: str,
    overlap_policy: str,
    num_proc: Optional[int],
    overwrite_cache: bool,
    limit: Optional[int],
    training_args: TrainingArguments,
) -> Tuple[Dataset, Dataset]:
    raw_subset = attach_example_indices(select_subset(raw_dataset, limit))
    tokenized = raw_subset.map(
        lambda batch: tokenize_labeled_examples(
            batch,
            tokenizer=tokenizer,
            text_column=columns.text,
            entities_column=columns.entities,  # type: ignore[arg-type]
            label_schema=label_schema,
            max_seq_length=max_seq_length,
            stride=stride,
            padding=padding,
            alignment_mode=alignment_mode,
            overlap_policy=overlap_policy,
        ),
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=not overwrite_cache,
        remove_columns=raw_subset.column_names,
        desc="Tokenizing labeled dataset with raw-text offsets",
    )
    return raw_subset, tokenized


def preprocess_prediction_dataset(
    raw_dataset: Dataset,
    *,
    columns: DatasetColumns,
    tokenizer: PreTrainedTokenizerFast,
    max_seq_length: int,
    stride: int,
    padding: str | bool,
    num_proc: Optional[int],
    overwrite_cache: bool,
    limit: Optional[int],
) -> Tuple[Dataset, Dataset]:
    raw_subset = attach_example_indices(select_subset(raw_dataset, limit))
    tokenized = raw_subset.map(
        lambda batch: tokenize_prediction_examples(
            batch,
            tokenizer=tokenizer,
            text_column=columns.text,
            example_index_column=INTERNAL_EXAMPLE_INDEX_COLUMN,
            max_seq_length=max_seq_length,
            stride=stride,
            padding=padding,
        ),
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=not overwrite_cache,
        remove_columns=raw_subset.column_names,
        desc="Tokenizing prediction dataset with raw-text offsets",
    )
    return raw_subset, tokenized


def build_trainer(
    *,
    model: transformers.PreTrainedModel,
    training_args: TrainingArguments,
    tokenizer: PreTrainedTokenizerFast,
    data_collator: DataCollatorForTokenClassification,
    compute_metrics: Optional[Callable[[Tuple[Any, Any]], Dict[str, float]]],
    train_dataset: Optional[Dataset],
    eval_dataset: Optional[Dataset],
) -> Trainer:
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }

    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    return Trainer(**trainer_kwargs)


def infer_last_checkpoint(training_args: TrainingArguments) -> Optional[str]:
    from transformers.trainer_utils import get_last_checkpoint

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and os.listdir(training_args.output_dir):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to train from scratch."
            )
    return last_checkpoint


def maybe_log_and_save_metrics(trainer: Trainer, split_name: str, metrics: Dict[str, Any]) -> None:
    trainer.log_metrics(split_name, metrics)
    trainer.save_metrics(split_name, metrics)


def normalize_strategy_name(value: Any) -> str:
    if value is None:
        return "no"
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def has_active_eval_strategy(training_args: TrainingArguments) -> bool:
    strategy = getattr(training_args, "eval_strategy", None)
    if strategy is None:
        strategy = getattr(training_args, "evaluation_strategy", None)
    return normalize_strategy_name(strategy) != "no"


def normalize_training_args_for_train_only(
    training_args: TrainingArguments,
    *,
    has_validation_data: bool,
) -> None:
    if training_args.load_best_model_at_end and not has_validation_data:
        raise ValueError("`load_best_model_at_end=True` requires validation data.")
    if has_active_eval_strategy(training_args) and not has_validation_data:
        raise ValueError("An evaluation strategy was requested, but no validation dataset was provided.")


def reconstruct_entities_from_offsets(
    *,
    text: str,
    full_offsets: Sequence[Tuple[int, int]],
    predicted_label_ids_by_offset: Dict[Tuple[int, int], int],
    label_schema: LabelSchema,
    passage_offset: int,
) -> List[Dict[str, Any]]:
    entities: List[Dict[str, Any]] = []
    active_label: Optional[str] = None
    active_start: Optional[int] = None
    active_end: Optional[int] = None

    for offset in full_offsets:
        token_start, token_end = offset
        if token_end <= token_start:
            continue

        label_id = predicted_label_ids_by_offset.get(offset, label_schema.encode("O"))
        tag = label_schema.id_to_label[int(label_id)]

        if tag == "O":
            if active_label is not None:
                entities.append(
                    {
                        "label": active_label,
                        "start_local": int(active_start),
                        "end_local": int(active_end),
                        "start": passage_offset + int(active_start),
                        "end": passage_offset + int(active_end),
                        "text": text[int(active_start):int(active_end)],
                    }
                )
                active_label = None
                active_start = None
                active_end = None
            continue

        if "-" in tag:
            prefix, entity_label = tag.split("-", 1)
        else:
            prefix, entity_label = "B", tag

        starts_new = (
            prefix == "B"
            or active_label is None
            or active_label != entity_label
        )

        if starts_new:
            if active_label is not None:
                entities.append(
                    {
                        "label": active_label,
                        "start_local": int(active_start),
                        "end_local": int(active_end),
                        "start": passage_offset + int(active_start),
                        "end": passage_offset + int(active_end),
                        "text": text[int(active_start):int(active_end)],
                    }
                )
            active_label = entity_label
            active_start = token_start
            active_end = token_end
        else:
            active_end = token_end

    if active_label is not None:
        entities.append(
            {
                "label": active_label,
                "start_local": int(active_start),
                "end_local": int(active_end),
                "start": passage_offset + int(active_start),
                "end": passage_offset + int(active_end),
                "text": text[int(active_start):int(active_end)],
            }
        )

    return entities


def reconstruct_prediction_outputs(
    *,
    raw_predict_dataset: Dataset,
    tokenized_predict_dataset: Dataset,
    prediction_logits: Sequence[Sequence[Sequence[float]]],
    label_schema: LabelSchema,
    tokenizer: PreTrainedTokenizerFast,
    columns: DatasetColumns,
) -> List[Dict[str, Any]]:
    aggregated_logits: Dict[int, Dict[Tuple[int, int], List[float]]] = {}
    aggregated_counts: Dict[int, Dict[Tuple[int, int], int]] = {}

    for chunk_index, chunk_logits in enumerate(prediction_logits):
        sample_index = int(tokenized_predict_dataset[chunk_index]["sample_index"])
        offset_mapping = tokenized_predict_dataset[chunk_index]["offset_mapping"]
        special_tokens_mask = tokenized_predict_dataset[chunk_index]["special_tokens_mask"]

        sample_logits = aggregated_logits.setdefault(sample_index, {})
        sample_counts = aggregated_counts.setdefault(sample_index, {})

        for token_logits, offset, is_special in zip(chunk_logits, offset_mapping, special_tokens_mask):
            token_start, token_end = int(offset[0]), int(offset[1])
            if is_special or token_end <= token_start:
                continue

            key = (token_start, token_end)
            if key not in sample_logits:
                sample_logits[key] = [float(value) for value in token_logits]
                sample_counts[key] = 1
            else:
                sample_logits[key] = [
                    old_value + float(new_value)
                    for old_value, new_value in zip(sample_logits[key], token_logits)
                ]
                sample_counts[key] += 1

    outputs: List[Dict[str, Any]] = []
    for raw_index, example in enumerate(raw_predict_dataset):
        text = str(example[columns.text])
        full_offsets = full_tokenize_texts([text], tokenizer=tokenizer)[0]
        sample_logits = aggregated_logits.get(raw_index, {})
        sample_counts = aggregated_counts.get(raw_index, {})
        predicted_label_ids_by_offset: Dict[Tuple[int, int], int] = {}

        for offset in full_offsets:
            if offset not in sample_logits:
                continue
            count = max(sample_counts.get(offset, 1), 1)
            averaged = [value / count for value in sample_logits[offset]]
            best_label = max(range(len(averaged)), key=lambda index: averaged[index])
            predicted_label_ids_by_offset[offset] = int(best_label)

        passage_offset = int(example[columns.passage_offset]) if columns.passage_offset and columns.passage_offset in example else 0
        predicted_entities = reconstruct_entities_from_offsets(
            text=text,
            full_offsets=full_offsets,
            predicted_label_ids_by_offset=predicted_label_ids_by_offset,
            label_schema=label_schema,
            passage_offset=passage_offset,
        )

        entry: Dict[str, Any] = {
            "id": example[columns.record_id] if columns.record_id and columns.record_id in example else raw_index,
            "text": text,
            "passage_offset": passage_offset,
            "predicted_entities": predicted_entities,
        }
        if columns.document_id and columns.document_id in example:
            entry["document_id"] = example[columns.document_id]
        if columns.passage_id and columns.passage_id in example:
            entry["passage_id"] = example[columns.passage_id]
        outputs.append(entry)

    return outputs


def save_prediction_outputs(output_dir: str, prediction_entries: List[Dict[str, Any]]) -> None:
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "predictions.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(prediction_entries, handle, ensure_ascii=False, indent=2)

    jsonl_path = os.path.join(output_dir, "predictions.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for entry in prediction_entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def iter_predicted_entities(prediction_entries: Sequence[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for entry in prediction_entries:
        document_id = str(entry.get("document_id", ""))
        for entity in entry.get("predicted_entities", []) or []:
            yield {
                "document_id": document_id,
                "label": str(entity["label"]),
                "start": int(entity["start"]),
                "end": int(entity["end"]),
            }


def write_predictions_to_bioc_xml(
    input_xml_path: str | Path,
    output_xml_path: str | Path,
    prediction_entries: Sequence[Dict[str, Any]],
) -> None:
    import bioc

    entities_by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for entity in iter_predicted_entities(prediction_entries):
        entities_by_doc.setdefault(str(entity["document_id"]), []).append(entity)

    for doc_entities in entities_by_doc.values():
        doc_entities.sort(key=lambda item: (int(item["start"]), int(item["end"]), str(item["label"])))

    bioc_load = getattr(bioc, "load", None)
    bioc_dump = getattr(bioc, "dump", None)
    if bioc_load is None or bioc_dump is None:
        from bioc import biocxml

        bioc_load = biocxml.load
        bioc_dump = biocxml.dump

    with open(input_xml_path, "r", encoding="utf-8") as handle:
        collection = bioc_load(handle)

    annotation_index = 1
    for document in collection.documents:
        document_entities = list(entities_by_doc.get(str(document.id), []))
        entity_pointer = 0
        for passage in document.passages:
            passage.annotations = []
            passage.relations = []
            passage_text = passage.text or ""
            passage_start = int(passage.offset or 0)
            passage_end = passage_start + len(passage_text)

            while entity_pointer < len(document_entities) and int(document_entities[entity_pointer]["end"]) <= passage_start:
                entity_pointer += 1

            local_pointer = entity_pointer
            while local_pointer < len(document_entities):
                entity = document_entities[local_pointer]
                start = int(entity["start"])
                end = int(entity["end"])
                if start >= passage_end:
                    break
                if passage_start <= start < end <= passage_end:
                    annotation = bioc.BioCAnnotation()
                    annotation.id = f"T{annotation_index}"
                    annotation.infons["type"] = str(entity["label"])
                    relative_start = start - passage_start
                    relative_end = end - passage_start
                    annotation.text = passage_text[relative_start:relative_end] if passage_text else ""
                    annotation.add_location(bioc.BioCLocation(start, end - start))
                    passage.add_annotation(annotation)
                    annotation_index += 1
                local_pointer += 1

    output_path = Path(output_xml_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        bioc_dump(collection, handle)
