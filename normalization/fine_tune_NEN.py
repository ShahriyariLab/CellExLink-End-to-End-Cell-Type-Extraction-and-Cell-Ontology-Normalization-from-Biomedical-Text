#!/usr/bin/env python3
"""
Fine-tune SapBERT (or another HF encoder) for biomedical name normalization / entity linking
using concept-labeled names.

The training file must contain one positive pair per line:

    concept_id||entity_name_1||entity_name_2

"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as hf_logging


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
hf_logging.set_verbosity_info()
hf_logging.enable_default_handler()
hf_logging.enable_explicit_format()


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Fine-tune SapBERT on CellLink training pairs.")
    parser.add_argument(
        "--train-pairs",
        type=Path,
        default=base_dir / "sapbert_data" / "sapbert_training_pairs.txt",
        help="Training pairs file in concept_id||name1||name2 format.",
    )
    parser.add_argument(
        "--dictionary-tsv",
        type=Path,
        default=base_dir / "sapbert_data" / "cell_ontology_plus_mentions.tsv",
        help="Combined concept dictionary TSV kept in run metadata.",
    )
    parser.add_argument(
        "--base-model",
        default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        help="Base Hugging Face model to fine-tune.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "models" / "sapbert_finetuned",
        help="Directory where the final fine-tuned model will be saved.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--labels-per-batch", type=int, default=16)
    parser.add_argument("--examples-per-label", type=int, default=2)
    parser.add_argument("--batches-per-epoch", type=int, default=1000)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-concepts", type=int, default=None)
    parser.add_argument("--max-names-per-concept", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--use-stemming", action="store_true")
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--keep-ambiguous-names", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TextPreprocessor:
    def __init__(self, lowercase: bool = False, use_stemming: bool = False):
        self.lowercase = lowercase
        self.use_stemming = use_stemming
        self._stemmer = None
        if use_stemming:
            try:
                try:
                    from .plural_stemmer import normalize_text as plural_normalize_text  # type: ignore
                except ImportError:
                    from plural_stemmer import normalize_text as plural_normalize_text  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "--use-stemming was set, but plural_stemmer.normalize_text could not be imported. "
                ) from exc
            self._stemmer = plural_normalize_text

    def __call__(self, text: str) -> str:
        text = " ".join(text.strip().split())
        if self.lowercase:
            text = text.lower()
        if self._stemmer is not None:
            text = self._stemmer(text)
        return text


@dataclass
class PairLoadStats:
    concepts_before_filtering: int
    concepts_after_filtering: int
    names_before_filtering: int
    names_after_filtering: int
    ambiguous_names_dropped: int
    concepts_dropped_for_low_support: int


def load_concept_to_names_from_pair_file(
    path: Path,
    preprocess: Callable[[str], str],
    max_concepts: int | None = None,
    max_names_per_concept: int | None = None,
    drop_ambiguous_names: bool = True,
    min_names_per_concept: int = 2,
) -> Tuple[Dict[str, List[str]], PairLoadStats]:
    concept_to_names_sets: Dict[str, set[str]] = defaultdict(set)

    with path.open("r", encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                concept_id, left, right = line.split("||", 2)
            except ValueError as exc:
                raise ValueError(f"Invalid pair format on line {line_number}: {line}") from exc

            left = preprocess(left)
            right = preprocess(right)
            if left:
                concept_to_names_sets[concept_id].add(left)
            if right:
                concept_to_names_sets[concept_id].add(right)

    if max_concepts is not None:
        kept_items = list(concept_to_names_sets.items())[:max_concepts]
        concept_to_names_sets = defaultdict(set, {k: v for k, v in kept_items})

    names_before = sum(len(v) for v in concept_to_names_sets.values())
    concepts_before = len(concept_to_names_sets)

    ambiguous_names_dropped = 0
    if drop_ambiguous_names:
        name_to_concepts: Dict[str, set[str]] = defaultdict(set)
        for concept_id, names in concept_to_names_sets.items():
            for name in names:
                name_to_concepts[name].add(concept_id)

        ambiguous_names = {name for name, concepts in name_to_concepts.items() if len(concepts) > 1}
        ambiguous_names_dropped = len(ambiguous_names)
        if ambiguous_names:
            for concept_id in list(concept_to_names_sets.keys()):
                concept_to_names_sets[concept_id] = {
                    name for name in concept_to_names_sets[concept_id] if name not in ambiguous_names
                }

    concept_to_names: Dict[str, List[str]] = {}
    concepts_dropped_for_low_support = 0
    for concept_id, names_set in concept_to_names_sets.items():
        names = sorted(names_set)
        if max_names_per_concept is not None:
            names = names[:max_names_per_concept]
        if len(names) < min_names_per_concept:
            concepts_dropped_for_low_support += 1
            continue
        concept_to_names[concept_id] = names

    stats = PairLoadStats(
        concepts_before_filtering=concepts_before,
        concepts_after_filtering=len(concept_to_names),
        names_before_filtering=names_before,
        names_after_filtering=sum(len(v) for v in concept_to_names.values()),
        ambiguous_names_dropped=ambiguous_names_dropped,
        concepts_dropped_for_low_support=concepts_dropped_for_low_support,
    )
    return concept_to_names, stats


class ConceptNameDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[int]):
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        self.texts = list(texts)
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return {"text": self.texts[idx], "label": self.labels[idx]}


class ConceptBatchSampler(BatchSampler):
    def __init__(
        self,
        indices_by_label: Dict[int, List[int]],
        labels_per_batch: int,
        examples_per_label: int,
        batches_per_epoch: int,
        seed: int = 42,
    ):
        if labels_per_batch < 2:
            raise ValueError("labels_per_batch must be >= 2")
        if examples_per_label < 2:
            raise ValueError("examples_per_label must be >= 2")
        eligible = {label: idxs for label, idxs in indices_by_label.items() if len(idxs) >= examples_per_label}
        if len(eligible) < labels_per_batch:
            raise ValueError(
                f"Need at least {labels_per_batch} labels with >= {examples_per_label} examples each; "
                f"only found {len(eligible)}."
            )
        self.indices_by_label = eligible
        self.label_list = sorted(eligible.keys())
        self.labels_per_batch = labels_per_batch
        self.examples_per_label = examples_per_label
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterable[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        for _ in range(self.batches_per_epoch):
            batch_labels = rng.sample(self.label_list, self.labels_per_batch)
            batch_indices: List[int] = []
            for label in batch_labels:
                candidates = self.indices_by_label[label]
                if len(candidates) == self.examples_per_label:
                    chosen = list(candidates)
                else:
                    chosen = rng.sample(candidates, self.examples_per_label)
                batch_indices.extend(chosen)
            rng.shuffle(batch_indices)
            yield batch_indices

    def __len__(self) -> int:
        return self.batches_per_epoch


class SapBERTEncoder(nn.Module):
    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name_or_path)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        return outputs.last_hidden_state[:, 0, :]


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, p=2, dim=1)
        device = embeddings.device
        batch_size = embeddings.size(0)

        labels = labels.view(-1, 1)
        positive_mask = torch.eq(labels, labels.T).to(device)
        logits_mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        positive_mask = positive_mask & logits_mask

        logits = torch.matmul(embeddings, embeddings.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        positives_per_anchor = positive_mask.sum(dim=1)
        valid = positives_per_anchor > 0
        if not torch.any(valid):
            raise RuntimeError("No positive pairs in batch. Increase examples_per_label.")

        mean_log_prob_pos = (log_prob * positive_mask).sum(dim=1) / positives_per_anchor.clamp_min(1)
        return -mean_log_prob_pos[valid].mean()


def build_training_arrays(concept_to_names: Dict[str, List[str]]) -> Tuple[List[str], List[int], Dict[str, int]]:
    concept_ids = sorted(concept_to_names.keys())
    concept_to_label = {concept_id: idx for idx, concept_id in enumerate(concept_ids)}
    texts: List[str] = []
    labels: List[int] = []
    for concept_id in concept_ids:
        label = concept_to_label[concept_id]
        for name in concept_to_names[concept_id]:
            texts.append(name)
            labels.append(label)
    return texts, labels, concept_to_label


def save_hf_checkpoint(save_dir: Path, model: SapBERTEncoder, tokenizer, extra_metadata: Dict[str, object]) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)
    with (save_dir / "training_metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(extra_metadata, fp, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()

    if args.examples_per_label < 2:
        raise ValueError("--examples-per-label must be >= 2")
    if args.labels_per_batch < 2:
        raise ValueError("--labels-per-batch must be >= 2")

    if args.train_batch_size is not None:
        expected_batch_size = args.labels_per_batch * args.examples_per_label
        if args.train_batch_size != expected_batch_size:
            raise ValueError(
                f"--train-batch-size ({args.train_batch_size}) must equal "
                f"labels_per_batch * examples_per_label ({expected_batch_size})"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    preprocess = TextPreprocessor(lowercase=args.lowercase, use_stemming=args.use_stemming)
    train_concept_to_names, train_stats = load_concept_to_names_from_pair_file(
        path=args.train_pairs,
        preprocess=preprocess,
        max_concepts=args.max_train_concepts,
        max_names_per_concept=args.max_names_per_concept,
        drop_ambiguous_names=not args.keep_ambiguous_names,
        min_names_per_concept=args.examples_per_label,
    )
    if not train_concept_to_names:
        raise ValueError("No usable training concepts remain after filtering.")

    texts, labels, concept_to_label = build_training_arrays(train_concept_to_names)
    label_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    dataset = ConceptNameDataset(texts=texts, labels=labels)
    batch_sampler = ConceptBatchSampler(
        indices_by_label=label_to_indices,
        labels_per_batch=args.labels_per_batch,
        examples_per_label=args.examples_per_label,
        batches_per_epoch=args.batches_per_epoch,
        seed=args.seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = SapBERTEncoder(args.base_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def collate_fn(batch: Sequence[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        batch_texts = [item["text"] for item in batch]
        batch_labels = torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        enc["labels"] = batch_labels
        return enc

    train_loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = SupervisedContrastiveLoss(temperature=args.temperature)

    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    metadata: Dict[str, object] = {
        "base_model": args.base_model,
        "train_pairs": str(args.train_pairs),
        "dictionary_tsv": str(args.dictionary_tsv),
        "output_dir": str(args.output_dir),
        "epochs": args.epochs,
        "labels_per_batch": args.labels_per_batch,
        "examples_per_label": args.examples_per_label,
        "train_batch_size": args.labels_per_batch * args.examples_per_label,
        "batches_per_epoch": args.batches_per_epoch,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "max_length": args.max_length,
        "seed": args.seed,
        "amp": amp_enabled,
        "train_stats": train_stats.__dict__,
        "num_training_texts": len(texts),
        "num_training_concepts": len(concept_to_label),
    }

    print(f"Base model: {args.base_model}")
    print(f"Output dir: {args.output_dir}")
    print(f"Device: {device}")
    print(f"Training concepts: {len(train_concept_to_names)}")
    print(f"Training names: {len(texts)}")
    print(f"Batch size: {args.labels_per_batch * args.examples_per_label}")

    train_history: List[Dict[str, float]] = []

    for epoch in range(args.epochs):
        model.train()
        batch_sampler.set_epoch(epoch)
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            labels_tensor = batch.pop("labels").to(device, non_blocking=True)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            autocast_ctx = torch.amp.autocast("cuda") if amp_enabled else contextlib.nullcontext()
            with autocast_ctx:
                embeddings = model(**batch)
                loss = criterion(embeddings, labels_tensor)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.item())

            if args.logging_steps and step % args.logging_steps == 0:
                avg_loss = epoch_loss / step
                print(
                    f"epoch={epoch + 1}/{args.epochs} "
                    f"batch={step}/{len(train_loader)} "
                    f"loss={loss.item():.6f} avg_loss={avg_loss:.6f}"
                )

        epoch_avg_loss = epoch_loss / max(len(train_loader), 1)
        train_history.append({"epoch": epoch + 1, "loss": epoch_avg_loss})
        print(f"Epoch {epoch + 1} finished. avg_loss={epoch_avg_loss:.6f}")

    metadata["train_history"] = train_history
    save_hf_checkpoint(args.output_dir, model, tokenizer, metadata)
    print(f"Saved final model to {args.output_dir}")
    logger.info("Saved final model to %s", args.output_dir)

    print("Training finished.")
    logger.info("Training finished successfully.")
    print("Use this in normalize.py:")
    print(f'    MODEL_NAMES = {{"SapBERT": "{args.output_dir}"}}')
    logger.info('Use this in normalize.py: MODEL_NAMES = {"SapBERT": "%s"}', args.output_dir)


if __name__ == "__main__":
    main()


# Example:
#   python fine_tune_NEN.py \
#     --train-pairs sapbert_data/sapbert_training_pairs.txt \
#     --dictionary-tsv sapbert_data/cell_ontology_plus_mentions.tsv \
#     --output-dir models/sapbert_finetuned
