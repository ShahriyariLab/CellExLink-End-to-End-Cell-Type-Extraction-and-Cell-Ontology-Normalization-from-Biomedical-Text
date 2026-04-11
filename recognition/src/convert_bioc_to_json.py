from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence

import bioc

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EntitySpan:
    start: int  # local passage offset
    end: int    # local passage offset
    label: str
    ann_id: str = ""
    text: str = ""

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"Invalid entity span: {self.start}..{self.end}")


@dataclass(frozen=True)
class PassageRecord:
    record_id: int
    document_id: str
    passage_id: int
    passage_offset: int
    text: str
    entities: List[EntitySpan]


def iter_input_files(paths: Sequence[str | Path]) -> Iterator[Path]:
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            yield path
        elif path.is_dir():
            for child in sorted(p for p in path.rglob("*") if p.is_file()):
                yield child
        else:
            raise FileNotFoundError(f"Input path does not exist: {path}")


def spans_overlap(left: EntitySpan, right: EntitySpan) -> bool:
    return max(left.start, right.start) < min(left.end, right.end)


def count_overlapping_pairs(entities: Sequence[EntitySpan]) -> int:
    count = 0
    for index, left in enumerate(entities):
        for right in entities[index + 1 :]:
            if spans_overlap(left, right):
                count += 1
    return count


def first_overlapping_pair(entities: Sequence[EntitySpan]) -> tuple[EntitySpan, EntitySpan] | None:
    for index, left in enumerate(entities):
        for right in entities[index + 1 :]:
            if spans_overlap(left, right):
                return left, right
    return None


def extract_entities(passage, *, overlap_policy: str = "last") -> List[EntitySpan]:
    if overlap_policy not in {"last", "error"}:
        raise ValueError("overlap_policy must be one of: last, error.")

    entities: List[EntitySpan] = []
    passage_text = passage.text or ""
    passage_offset = int(passage.offset or 0)

    for ann in passage.annotations:
        label = str(ann.infons.get("type", "Unknown"))
        ann_text = ann.text or ""
        locations = list(getattr(ann, "locations", []) or [])

        if not locations:
            total = getattr(ann, "total_span", None)
            if total is None:
                LOGGER.warning("Annotation %s has no locations and no total span; skipping.", ann.id)
                continue
            locations = [type("_Loc", (), {"offset": total.offset, "length": total.length})()]

        if len(locations) > 1:
            LOGGER.warning(
                "Annotation %s has %d locations; treating each location as a separate flat entity in file order.",
                ann.id,
                len(locations),
            )

        for location_index, loc in enumerate(locations):
            absolute_start = int(loc.offset)
            absolute_end = absolute_start + int(loc.length)

            if absolute_end <= absolute_start:
                LOGGER.warning("Skipping zero-length/invalid annotation %s at %d-%d", ann.id, absolute_start, absolute_end)
                continue

            if absolute_start < passage_offset or absolute_end > passage_offset + len(passage_text):
                LOGGER.warning(
                    "Skipping annotation %s because it falls outside passage bounds: %d-%d not in [%d, %d).",
                    ann.id,
                    absolute_start,
                    absolute_end,
                    passage_offset,
                    passage_offset + len(passage_text),
                )
                continue

            local_start = absolute_start - passage_offset
            local_end = absolute_end - passage_offset
            actual_text = passage_text[local_start:local_end]
            if ann_text and actual_text and ann_text != actual_text:
                LOGGER.warning(
                    "Annotation text mismatch at absolute offset %d: expected %r, found %r",
                    absolute_start,
                    ann_text,
                    actual_text,
                )

            entities.append(
                EntitySpan(
                    start=local_start,
                    end=local_end,
                    label=label,
                    ann_id=f"{ann.id}:{location_index}" if len(locations) > 1 else str(ann.id),
                    text=ann_text or actual_text,
                )
            )

    overlap_count = count_overlapping_pairs(entities)
    if overlap_count:
        if overlap_policy == "error":
            pair = first_overlapping_pair(entities)
            assert pair is not None
            raise ValueError(
                "Overlapping entities found in BioC XML, but overlap_policy='error': "
                f"{pair[0].label!r} at {pair[0].start}-{pair[0].end} overlaps "
                f"{pair[1].label!r} at {pair[1].start}-{pair[1].end}"
            )
        LOGGER.warning(
            "Detected %d overlapping entity pair(s) in passage offset %d. File order is preserved; training with overlap_policy='last' will let later annotations override earlier ones on overlapping tokenizer spans.",
            overlap_count,
            passage_offset,
        )

    return entities


def iter_passage_records(
    srcs: Sequence[str | Path],
    *,
    include_entities: bool,
    overlap_policy: str = "last",
) -> Iterator[PassageRecord]:
    record_id = 0
    for src in iter_input_files(srcs):
        LOGGER.info("Reading %s", src)
        with bioc.biocxml.iterparse(str(src)) as reader:
            _ = reader.get_collection_info()
            for document in reader:
                for passage_index, passage in enumerate(document.passages):
                    text = passage.text or ""
                    if not text:
                        continue

                    entities = extract_entities(passage, overlap_policy=overlap_policy) if include_entities else []
                    yield PassageRecord(
                        record_id=record_id,
                        document_id=str(document.id),
                        passage_id=passage_index,
                        passage_offset=int(passage.offset or 0),
                        text=text,
                        entities=entities,
                    )
                    record_id += 1


def convert_bioc_to_json(
    srcs: Sequence[str | Path],
    dest: str | Path,
    *,
    include_entities: bool = True,
    overlap_policy: str = "last",
) -> int:
    output_path = Path(dest)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_records = 0
    with output_path.open("w", encoding="utf-8") as sink:
        for record in iter_passage_records(srcs, include_entities=include_entities, overlap_policy=overlap_policy):
            payload = {
                "id": record.record_id,
                "document_id": record.document_id,
                "passage_id": record.passage_id,
                "passage_offset": record.passage_offset,
                "text": record.text,
            }
            if include_entities:
                payload["entities"] = [
                    {"start": entity.start, "end": entity.end, "label": entity.label, "text": entity.text}
                    for entity in record.entities
                ]
            sink.write(json.dumps(payload, ensure_ascii=False) + "\n")
            num_records += 1

    LOGGER.info(
        "Wrote %d passage records to %s (include_entities=%s, overlap_policy=%s).",
        num_records,
        output_path,
        include_entities,
        overlap_policy,
    )
    return num_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert BioC XML into JSONL passage records for offset-based NER training/prediction."
    )
    parser.add_argument("inputs", nargs="+", help="Input BioC XML files or directories.")
    parser.add_argument("--output", required=True, help="Destination JSONL file.")
    parser.add_argument(
        "--mode",
        choices=["train", "predict"],
        default="train",
        help="`train` keeps BioC entities as supervision labels.",
    )
    parser.add_argument(
        "--overlap-policy",
        choices=["last", "error"],
        default="last",
        help="How train-mode handles overlapping spans. `last` preserves annotation order and lets later annotations override earlier ones on overlapping tokenizer spans. Ignored in predict mode.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    include_entities = args.mode == "train"
    count = convert_bioc_to_json(
        args.inputs,
        args.output,
        include_entities=include_entities,
        overlap_policy=args.overlap_policy,
    )
    print(f"Wrote {count} passage records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
