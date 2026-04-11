import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import bioc


DEFAULT_INPUT_PATH = Path("../../model_outputs/test_output.xml")
DEFAULT_MODEL_NAMES = ["SapBERT_finetuned"]


PassageKey = str
LocationTuple = Tuple[Tuple[int, int], ...]


def load_collection(input_path: Path):
    with open(input_path, "r", encoding="utf-8") as readfp:
        return bioc.load(readfp)


def get_passage_key(doc, passage) -> str:
    passage_id = passage.infons.get("passage_id")
    if passage_id not in [None, ""]:
        return passage_id

    article_id = passage.infons.get("article-id_pmid")
    if article_id not in [None, ""]:
        return article_id

    return f"{doc.id}:{passage.offset}"


def get_annotation_locations(ann) -> LocationTuple:
    return tuple(sorted((location.offset, location.length) for location in ann.locations))


def get_annotation_text(ann) -> str:
    return (ann.text or "").strip()


def clean_identifier_text(identifier_text: str) -> str:
    return (
        identifier_text.strip()
        .replace("(skos:exact)", "")
        .replace("(skos:related)", "")
        .strip()
    )


def split_identifier_field(identifier_text: str) -> List[str]:
    cleaned = clean_identifier_text(identifier_text)
    parts = [part.strip() for part in re.split(r"[;,]", cleaned)]
    return [part for part in parts if part and part != "-" and part.lower() != "none"]


def exact_ids_only_iterator(bioc_collection):
    for doc in bioc_collection.documents:
        for passage in doc.passages:
            for ann in passage.annotations:
                identifier_i = ann.infons.get("identifier")
                if not identifier_i:
                    continue

                identifier_text = identifier_i.strip()
                if not identifier_text or "none" in identifier_text.lower():
                    continue
                if ";" in identifier_text or "," in identifier_text:
                    continue
                if "(skos:related)" in identifier_text.lower():
                    continue

                normalized_id = clean_identifier_text(identifier_text)
                if normalized_id and normalized_id != "-":
                    yield doc, passage, ann, (normalized_id,)


def singleID_iterator(bioc_collection):
    for doc in bioc_collection.documents:
        for passage in doc.passages:
            for ann in passage.annotations:
                identifier_i = ann.infons.get("identifier")
                if not identifier_i:
                    continue

                identifier_text = identifier_i.strip()
                if not identifier_text or "none" in identifier_text.lower():
                    continue
                if ";" in identifier_text or "," in identifier_text:
                    continue

                normalized_id = clean_identifier_text(identifier_text)
                if normalized_id and normalized_id != "-":
                    yield doc, passage, ann, (normalized_id,)


def all_labels_iterator(bioc_collection):
    for doc in bioc_collection.documents:
        for passage in doc.passages:
            for ann in passage.annotations:
                identifier_i = ann.infons.get("identifier")
                if not identifier_i:
                    continue

                all_ids = split_identifier_field(identifier_i)
                if all_ids:
                    yield doc, passage, ann, all_ids


ITERATORS = {
    "exactIDsOnly_iterator": exact_ids_only_iterator,
    "singleID_iterator": singleID_iterator,
    "allLabels_iterator": all_labels_iterator,
}


def build_score_key(model_name: str, rank: int) -> str:
    return f"{model_name}_identifier_score_{rank}"


def get_prediction_tuples(
    doc,
    passage,
    ann,
    model_name: str,
    max_k: int = 10,
    include_locations: bool = False,
    include_text: bool = False,
    score_threshold: Optional[float] = None,
):
    tuples = []
    passage_key = get_passage_key(doc, passage)
    locations = get_annotation_locations(ann) if include_locations else None
    ann_text = get_annotation_text(ann)

    for i in range(max_k):
        key = f"{model_name}_id_{i}"
        if key not in ann.infons:
            continue

        pred_id = (ann.infons[key] or "").strip()
        if pred_id.lower() in ["", "-", "none"]:
            continue

        if score_threshold is not None:
            score_text = ann.infons.get(build_score_key(model_name, i))
            if score_text in [None, ""]:
                continue
            try:
                if float(score_text) < score_threshold:
                    continue
            except ValueError:
                continue

        if include_locations and include_text:
            tuples.append((passage_key, ann.infons["type"], locations, pred_id, ann_text))
        elif include_locations:
            tuples.append((passage_key, ann.infons["type"], locations, pred_id))
        elif include_text:
            tuples.append((passage_key, ann.infons["type"], pred_id, ann_text))
        else:
            tuples.append((passage_key, ann.infons["type"], pred_id))

    return tuples


def calculate_metrics_from_sets(ref_tuples: Set[tuple], pred_tuples: Set[tuple]) -> Dict[str, float]:
    matches = len(ref_tuples.intersection(pred_tuples))
    precision_denominator = len(pred_tuples)
    recall_denominator = len(ref_tuples)

    precision = matches / precision_denominator if precision_denominator else 0.0
    recall = matches / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision": precision,
        "precision_numerator": matches,
        "precision_denominator": precision_denominator,
        "recall": recall,
        "recall_numerator": matches,
        "recall_denominator": recall_denominator,
        "f1": f1,
    }


def get_reference_tuples(
    reference_collection,
    iterator_name: str,
    entity_types: Sequence[str],
    include_locations: bool = False,
) -> Set[tuple]:
    iterator = ITERATORS[iterator_name]
    ref_tuples = set()

    for doc, passage, ann, ref_ids_i in iterator(reference_collection):
        if ann.infons.get("type") not in entity_types:
            continue

        passage_key = get_passage_key(doc, passage)
        if include_locations:
            locations = get_annotation_locations(ann)
            ref_tuples.update((passage_key, ann.infons["type"], locations, ref_id) for ref_id in ref_ids_i)
        else:
            ref_tuples.update((passage_key, ann.infons["type"], ref_id) for ref_id in ref_ids_i)

    return ref_tuples


def iter_prediction_annotations(prediction_collection, entity_types: Sequence[str]):
    for doc in prediction_collection.documents:
        for passage in doc.passages:
            for ann in passage.annotations:
                if ann.infons.get("type") in entity_types:
                    yield doc, passage, ann


def evaluate_iterator(
    reference_collection,
    iterator_name: str,
    model_names: Sequence[str],
    entity_types: Sequence[str],
    score_mode: str = "gold_mention_normalize",
    prediction_collection=None,
    score_threshold: Optional[float] = None,
):
    results = []
    prediction_collection = prediction_collection or reference_collection
    include_locations = score_mode == "end_to_end"
    include_text = False

    for model_name in model_names:
        ref_tuples = get_reference_tuples(
            reference_collection,
            iterator_name,
            entity_types,
            include_locations=include_locations,
        )

        top1_pred_tuples = set()
        top5_pred_tuples = set()
        top10_pred_tuples = set()

        prediction_iterator = (
            iter_prediction_annotations(prediction_collection, entity_types)
            if include_locations
            else ITERATORS[iterator_name](prediction_collection)
        )

        for item in prediction_iterator:
            if include_locations:
                doc, passage, ann = item
            else:
                doc, passage, ann, _ = item
                if ann.infons.get("type") not in entity_types:
                    continue

            top10_tuples_i = get_prediction_tuples(
                doc,
                passage,
                ann,
                model_name,
                max_k=10,
                include_locations=include_locations,
                include_text=include_text,
                score_threshold=score_threshold,
            )
            if not top10_tuples_i:
                continue

            top1_pred_tuples.add(top10_tuples_i[0])
            top5_pred_tuples.update(top10_tuples_i[:5])
            top10_pred_tuples.update(top10_tuples_i)

        model_result = {"model_name": model_name, "topk": []}
        for k, pred_set in [("1", top1_pred_tuples), ("5", top5_pred_tuples), ("10", top10_pred_tuples)]:
            metrics = calculate_metrics_from_sets(ref_tuples, pred_set)

            model_result["topk"].append(
                {
                    "k": k,
                    "score_mode": score_mode,
                    "precision": metrics["precision"],
                    "precision_numerator": metrics["precision_numerator"],
                    "precision_denominator": metrics["precision_denominator"],
                    "recall": metrics["recall"],
                    "recall_numerator": metrics["recall_numerator"],
                    "recall_denominator": metrics["recall_denominator"],
                    "f1": metrics["f1"],
                }
            )

        results.append(model_result)

    return results


def print_results(iterator_name: str, entity_types: Sequence[str], results):
    print(f"Iterator: {iterator_name}")
    print(f"Entity types: {list(entity_types)}")
    for model_result in results:
        print(model_result["model_name"])
        for topk_result in model_result["topk"]:
            print(
                f"  top-{topk_result['k']}: precision {topk_result['precision']:.3f} "
                f"({topk_result['precision_numerator']}/{topk_result['precision_denominator']}), "
                f"recall {topk_result['recall']:.3f} "
                f"({topk_result['recall_numerator']}/{topk_result['recall_denominator']}), "
                f"F1 {topk_result['f1']:.3f}"
            )
        print()


def run_eval(
    reference_path: Path = DEFAULT_INPUT_PATH,
    model_names: Optional[Sequence[str]] = None,
    score_mode: str = "gold_mention_normalize",
    prediction_path: Optional[Path] = None,
    score_threshold: Optional[float] = None,
):
    if model_names is None:
        model_names = DEFAULT_MODEL_NAMES

    reference_collection = load_collection(reference_path)
    prediction_collection = load_collection(prediction_path) if prediction_path is not None else None
    all_results = []

    for iterator_name in ["exactIDsOnly_iterator", "allLabels_iterator"]:
        for entity_types in [["cell_type"]]:
            results = evaluate_iterator(
                reference_collection=reference_collection,
                iterator_name=iterator_name,
                model_names=model_names,
                entity_types=entity_types,
                score_mode=score_mode,
                prediction_collection=prediction_collection,
                score_threshold=score_threshold,
            )
            all_results.append((iterator_name, entity_types, results))

    return all_results


def main():
    all_results = run_eval()
    for iterator_name, entity_types, results in all_results:
        print_results(iterator_name, entity_types, results)


if __name__ == "__main__":
    main()
