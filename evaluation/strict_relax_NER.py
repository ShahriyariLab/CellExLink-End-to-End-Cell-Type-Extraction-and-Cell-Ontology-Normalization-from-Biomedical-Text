from __future__ import annotations

import argparse
import collections
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import bioc

LOG_FORMAT = "[%(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
LOGGER = logging.getLogger(__name__)
WORD_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)
DEFAULT_EXCLUDED_ANNOTATION_TYPES = {"cell_vague"}


@dataclass(frozen=True, order=True)
class Location:
    offset: int
    length: int

    @property
    def end(self) -> int:
        return self.offset + self.length


@dataclass(frozen=True)
class SpanEntity:
    passage_key: str
    entity_type: str
    locations: Tuple[Location, ...]
    text: str

    @property
    def signature(self) -> Tuple[str, str, Tuple[Location, ...]]:
        return self.passage_key, self.entity_type, self.locations


@dataclass(frozen=True)
class PassageRecord:
    offset: int
    text: str
    token_spans: Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class EvaluationCounts:
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass(frozen=True)
class EvaluationResult:
    precision: float
    recall: float
    f_score: float


@dataclass(frozen=True)
class LoadedPath:
    annotations: Tuple[SpanEntity, ...]
    passages: Dict[str, PassageRecord]



def _iter_xml_files(path: Path) -> Iterator[Path]:
    if path.is_file():
        if path.suffix.lower() != ".xml":
            raise ValueError(f"Expected a BioC XML file, got: {path}")
        yield path
        return

    if not path.is_dir():
        raise FileNotFoundError(f"Path is not a file or directory: {path}")

    for xml_path in sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".xml"):
        yield xml_path



def _normalize_annotation_types(annotation_types: Optional[Sequence[str] | str]) -> Optional[Set[str]]:
    if annotation_types is None:
        return None

    if isinstance(annotation_types, str):
        value = annotation_types.strip()
        if not value or value.lower() == "none":
            return None
        return set(value.split())

    cleaned = {value for value in annotation_types if value and value.lower() != "none"}
    return cleaned or None


def _should_skip_annotation_type(entity_type: str, allowed_types: Optional[Set[str]]) -> bool:
    if allowed_types is not None:
        return entity_type not in allowed_types
    return entity_type in DEFAULT_EXCLUDED_ANNOTATION_TYPES



def _source_prefix(root_path: Path, xml_path: Path) -> str:
    if root_path.is_file():
        return ""
    return str(xml_path.relative_to(root_path))



def _passage_key(source_prefix: str, document_id: Optional[str], passage: bioc.BioCPassage) -> str:
    base_document_id = document_id or "unknown_document"
    passage_id = passage.infons.get("passage_id")
    base_key = passage_id if passage_id else f"{base_document_id}:{passage.offset}"
    return f"{source_prefix}::{base_key}" if source_prefix else base_key



def _token_spans_for_text(text: str, offset: int) -> Tuple[Tuple[int, int], ...]:
    return tuple((offset + match.start(), offset + match.end()) for match in WORD_PATTERN.finditer(text))



def _normalize_annotation_text(annotation: bioc.BioCAnnotation, passage: bioc.BioCPassage) -> str:
    if annotation.text:
        return annotation.text

    parts: List[str] = []
    passage_text = passage.text or ""
    for location in annotation.locations:
        start = location.offset - passage.offset
        end = start + location.length
        parts.append(passage_text[start:end])
    return " ".join(parts)



def _annotation_locations(annotation: bioc.BioCAnnotation) -> Tuple[Location, ...]:
    locations = tuple(sorted((Location(loc.offset, loc.length) for loc in annotation.locations), key=lambda item: (item.offset, item.length)))
    if not locations:
        raise ValueError(f"Annotation has no locations: {annotation!r}")
    return locations



def _location_text(passage_text: str, passage_offset: int, locations: Sequence[Location]) -> str:
    parts = []
    for location in locations:
        start = location.offset - passage_offset
        end = start + location.length
        parts.append(passage_text[start:end])
    return " ".join(parts)



def load_bioc_annotations(
    path: str | Path,
    annotation_types: Optional[Sequence[str] | str] = None,
) -> LoadedPath:
    root_path = Path(path).resolve()
    allowed_types = _normalize_annotation_types(annotation_types)

    annotations: List[SpanEntity] = []
    passages: Dict[str, PassageRecord] = {}

    for xml_path in _iter_xml_files(root_path):
        source_prefix = _source_prefix(root_path, xml_path)
        LOGGER.info("Reading %s", xml_path)
        with open(xml_path, "r", encoding="utf-8") as handle:
            collection = bioc.load(handle)

        for document in collection.documents:
            document_id = document.id
            for passage in document.passages:
                passage_text = passage.text or ""
                passage_key = _passage_key(source_prefix, document_id, passage)
                passage_record = PassageRecord(
                    offset=passage.offset,
                    text=passage_text,
                    token_spans=_token_spans_for_text(passage_text, passage.offset),
                )
                passages[passage_key] = passage_record

                passage_end = passage.offset + len(passage_text)
                for annotation in passage.annotations:
                    entity_type = annotation.infons.get("type")
                    if entity_type is None:
                        LOGGER.warning("Skipping annotation without type in %s / %s", xml_path, passage_key)
                        continue
                    if _should_skip_annotation_type(entity_type, allowed_types):
                        continue

                    locations = _annotation_locations(annotation)
                    if any(loc.length <= 0 for loc in locations):
                        LOGGER.warning("Skipping zero-length annotation %s in %s", annotation.id, passage_key)
                        continue
                    if any(loc.offset < passage.offset or loc.end > passage_end for loc in locations):
                        LOGGER.warning("Skipping out-of-passage annotation %s in %s", annotation.id, passage_key)
                        continue

                    annotation_text = _normalize_annotation_text(annotation, passage)
                    extracted_text = _location_text(passage_text, passage.offset, locations)
                    if annotation_text != extracted_text:
                        LOGGER.warning(
                            "Annotation text mismatch in %s / %s / %s: %r vs %r",
                            xml_path,
                            passage_key,
                            annotation.id,
                            annotation_text,
                            extracted_text,
                        )

                    annotations.append(
                        SpanEntity(
                            passage_key=passage_key,
                            entity_type=entity_type,
                            locations=locations,
                            text=annotation_text,
                        )
                    )

    return LoadedPath(tuple(annotations), passages)



def _spans_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return start_a < end_b and start_b < end_a



def _has_shared_token(ref_entity: SpanEntity, pred_entity: SpanEntity, passages: Dict[str, PassageRecord]) -> bool:
    if ref_entity.passage_key != pred_entity.passage_key:
        return False

    passage = passages.get(ref_entity.passage_key)
    if passage is None:
        return False

    for ref_location in ref_entity.locations:
        for pred_location in pred_entity.locations:
            if not _spans_overlap(ref_location.offset, ref_location.end, pred_location.offset, pred_location.end):
                continue
            for token_start, token_end in passage.token_spans:
                if _spans_overlap(ref_location.offset, ref_location.end, token_start, token_end) and _spans_overlap(
                    pred_location.offset, pred_location.end, token_start, token_end
                ):
                    return True
    return False



def _build_match_graph(
    references: Sequence[SpanEntity],
    predictions: Sequence[SpanEntity],
    passages: Dict[str, PassageRecord],
    prediction_text_blacklist: Set[str],
) -> Dict[int, List[int]]:
    edges: Dict[int, List[int]] = {index: [] for index in range(len(references))}
    for ref_index, reference in enumerate(references):
        for pred_index, prediction in enumerate(predictions):
            if prediction.text.lower() in prediction_text_blacklist:
                continue
            if reference.entity_type != prediction.entity_type:
                continue
            if _has_shared_token(reference, prediction, passages):
                edges[ref_index].append(pred_index)
    return edges



def _maximum_bipartite_matching(edges: Dict[int, List[int]], num_predictions: int) -> int:
    matched_prediction_to_ref: List[Optional[int]] = [None] * num_predictions

    def dfs(ref_index: int, seen_predictions: Set[int]) -> bool:
        for pred_index in edges.get(ref_index, []):
            if pred_index in seen_predictions:
                continue
            seen_predictions.add(pred_index)
            current_ref = matched_prediction_to_ref[pred_index]
            if current_ref is None or dfs(current_ref, seen_predictions):
                matched_prediction_to_ref[pred_index] = ref_index
                return True
        return False

    matches = 0
    for ref_index in edges:
        if dfs(ref_index, set()):
            matches += 1
    return matches



def compute_strict_metrics(
    reference_annotations: Sequence[SpanEntity],
    predicted_annotations: Sequence[SpanEntity],
) -> EvaluationResult:
    reference_signatures = {annotation.signature for annotation in reference_annotations}
    predicted_signatures = {annotation.signature for annotation in predicted_annotations}

    true_positives = len(reference_signatures & predicted_signatures)
    false_positives = len(predicted_signatures - reference_signatures)
    false_negatives = len(reference_signatures - predicted_signatures)

    return metrics_from_counts(EvaluationCounts(true_positives, false_positives, false_negatives))



def compute_relaxed_metrics(
    reference_annotations: Sequence[SpanEntity],
    predicted_annotations: Sequence[SpanEntity],
    passages: Dict[str, PassageRecord],
    prediction_text_blacklist: Optional[Iterable[str]] = None,
) -> EvaluationResult:
    blacklist = {item.lower() for item in (prediction_text_blacklist or [])}

    refs_by_passage: Dict[str, List[SpanEntity]] = collections.defaultdict(list)
    preds_by_passage: Dict[str, List[SpanEntity]] = collections.defaultdict(list)

    for annotation in reference_annotations:
        refs_by_passage[annotation.passage_key].append(annotation)
    for annotation in predicted_annotations:
        preds_by_passage[annotation.passage_key].append(annotation)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    all_passages = set(refs_by_passage) | set(preds_by_passage)
    for passage_key in all_passages:
        references = refs_by_passage.get(passage_key, [])
        predictions = preds_by_passage.get(passage_key, [])
        if not references:
            false_positives += len(predictions)
            continue
        if not predictions:
            false_negatives += len(references)
            continue

        edges = _build_match_graph(references, predictions, passages, blacklist)
        matches = _maximum_bipartite_matching(edges, len(predictions))
        true_positives += matches
        false_negatives += len(references) - matches
        false_positives += len(predictions) - matches

    return metrics_from_counts(EvaluationCounts(true_positives, false_positives, false_negatives))



def metrics_from_counts(counts: EvaluationCounts) -> EvaluationResult:
    if counts.true_positives == 0:
        return EvaluationResult(0.0, 0.0, 0.0)

    precision_denominator = counts.true_positives + counts.false_positives
    recall_denominator = counts.true_positives + counts.false_negatives
    if precision_denominator == 0 or recall_denominator == 0:
        return EvaluationResult(0.0, 0.0, 0.0)

    precision = counts.true_positives / precision_denominator
    recall = counts.true_positives / recall_denominator
    if precision + recall == 0.0:
        f_score = 0.0
    else:
        f_score = 2.0 * precision * recall / (precision + recall)
    return EvaluationResult(precision, recall, f_score)



def verify_passage_sets(reference_passages: Dict[str, PassageRecord], predicted_passages: Dict[str, PassageRecord]) -> List[str]:
    errors: List[str] = []

    reference_keys = set(reference_passages)
    predicted_keys = set(predicted_passages)

    missing = sorted(reference_keys - predicted_keys)
    extra = sorted(predicted_keys - reference_keys)
    if missing:
        errors.append(f"Prediction path is missing passages: {', '.join(missing)}")
    if extra:
        errors.append(f"Prediction path contains extra passages: {', '.join(extra)}")

    for passage_key in sorted(reference_keys & predicted_keys):
        reference_passage = reference_passages[passage_key]
        predicted_passage = predicted_passages[passage_key]
        if reference_passage.offset != predicted_passage.offset:
            errors.append(
                f"Passage offsets do not match for {passage_key}: {reference_passage.offset} != {predicted_passage.offset}"
            )
        if reference_passage.text != predicted_passage.text:
            errors.append(f"Passage text does not match for {passage_key}")

    return errors



def drop_predictions_for_unknown_passages(
    predictions: Sequence[SpanEntity],
    predicted_passages: Dict[str, PassageRecord],
    reference_passages: Dict[str, PassageRecord],
) -> Tuple[List[SpanEntity], Dict[str, PassageRecord]]:
    allowed_passages = set(reference_passages)
    filtered_predictions = [annotation for annotation in predictions if annotation.passage_key in allowed_passages]
    filtered_passages = {key: value for key, value in predicted_passages.items() if key in allowed_passages}
    return filtered_predictions, filtered_passages



def evaluate_paths(
    reference_path: str | Path,
    prediction_path: str | Path,
    evaluation_method: str = "strict",
    annotation_types: Optional[Sequence[str] | str] = None,
    verify_documents: bool = True,
    skip_extra_pred_passages: bool = False,
    prediction_text_blacklist: Optional[Sequence[str]] = None,
) -> EvaluationResult:
    reference = load_bioc_annotations(reference_path, annotation_types=annotation_types)
    predictions = load_bioc_annotations(prediction_path, annotation_types=annotation_types)

    predicted_annotations = list(predictions.annotations)
    predicted_passages = dict(predictions.passages)
    if skip_extra_pred_passages:
        predicted_annotations, predicted_passages = drop_predictions_for_unknown_passages(
            predicted_annotations,
            predicted_passages,
            reference.passages,
        )

    if verify_documents:
        verification_errors = verify_passage_sets(reference.passages, predicted_passages)
        if verification_errors:
            for error in verification_errors:
                LOGGER.error(error)
            raise ValueError("Reference and prediction passages do not match.")

    if evaluation_method == "strict":
        return compute_strict_metrics(reference.annotations, predicted_annotations)
    if evaluation_method == "relax":
        return compute_relaxed_metrics(
            reference.annotations,
            predicted_annotations,
            reference.passages,
            prediction_text_blacklist=prediction_text_blacklist,
        )

    raise ValueError(f"Unknown evaluation method: {evaluation_method}")



def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate BioC NER predictions against reference BioC XML.")
    parser.add_argument("--reference_path", "-r", type=str, required=True, help="Reference BioC XML file or directory.")
    parser.add_argument("--prediction_path", "-p", type=str, required=True, help="Prediction BioC XML file or directory.")
    parser.add_argument(
        "--evaluation_method",
        "-m",
        choices={"strict", "relax"},
        required=True,
        help="Use exact span/type matching ('strict') or token-overlap matching ('relax').",
    )
    parser.add_argument(
        "--annotation_type",
        "-a",
        type=str,
        default="None",
        help=(
            "Whitespace-separated annotation types to keep. Use 'None' to keep the default set, "
            "which excludes cell_vague unless it is explicitly requested."
        ),
    )
    parser.add_argument(
        "--logging_level",
        "-l",
        type=str,
        default="INFO",
        help="Logging level: critical, error, warning, info, debug.",
    )
    parser.add_argument(
        "--no_document_verification",
        dest="verify_documents",
        action="store_const",
        const=False,
        default=True,
        help="Do not verify that passage IDs, offsets, and texts match.",
    )
    parser.add_argument(
        "--skip_extra_pred_passages",
        action="store_true",
        help="Ignore predicted passages that are not present in the reference input.",
    )
    parser.add_argument(
        "--prediction_text_blacklist",
        nargs="*",
        default=[],
        help="Prediction texts that can never match in relaxed evaluation. They still count as false positives if unmatched.",
    )
    return parser



def cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.logging_level.upper(), format=LOG_FORMAT)
    result = evaluate_paths(
        reference_path=args.reference_path,
        prediction_path=args.prediction_path,
        evaluation_method=args.evaluation_method,
        annotation_types=args.annotation_type,
        verify_documents=args.verify_documents,
        skip_extra_pred_passages=args.skip_extra_pred_passages,
        prediction_text_blacklist=args.prediction_text_blacklist,
    )
    print(f"P = {result.precision:.3f}, R = {result.recall:.3f}, F = {result.f_score:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
