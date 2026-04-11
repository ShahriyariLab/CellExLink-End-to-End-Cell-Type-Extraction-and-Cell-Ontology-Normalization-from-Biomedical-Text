from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import bioc
import pyobo
import spacy
from scispacy.linking import EntityLinker

MODEL_NAME = "scispacy"
DEFAULT_TOPN = 10
DEFAULT_ONTOLOGY_PREFIX = os.environ.get("PYOBO_ONTOLOGY_PREFIX", "cl")
DEFAULT_SCORE_THRESHOLD = float(os.environ.get("PYOBO_SCORE_THRESHOLD", "0.0"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SciSpaCy/PyOBO entity linking on BioC XML annotations."
    )
    parser.add_argument("--input-xml", type=Path, required=True, help="Input BioC XML file with NER mentions.")
    parser.add_argument("--output-xml", type=Path, required=True, help="Output BioC XML file with linked IDs.")
    parser.add_argument(
        "--ontology-prefix",
        default=DEFAULT_ONTOLOGY_PREFIX,
        help="PyOBO ontology prefix to link against.",
    )
    parser.add_argument("--topn", type=int, default=DEFAULT_TOPN, help="Maximum number of candidates to keep.")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        help="Minimum linker score to keep a candidate.",
    )
    parser.add_argument(
        "--disable-abbreviations",
        action="store_true",
        help="Disable abbreviation expansion from passage text.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information while linking.",
    )
    parser.add_argument(
        "--debug-max-print",
        type=int,
        default=200,
        help="Maximum number of detailed annotation debug blocks to print.",
    )
    return parser.parse_args()


def debug_print(message: str, enabled: bool) -> None:
    if enabled:
        print(message, flush=True)


def normalize_identifier_case(identifier: str) -> str:
    if not identifier:
        return identifier

    identifier = identifier.strip()
    return re.sub(r"^cl[_:]", "CL:", identifier, flags=re.IGNORECASE)


def load_celltype_linker(ontology_prefix: str, topn: int, debug: bool) -> EntityLinker:
    debug_print(f"[DEBUG] Loading linker for ontology: {ontology_prefix}", debug)
    try:
        linker = pyobo.get_scispacy_entity_linker(
            ontology_prefix,
            filter_for_definitions=False,
            max_entities_per_mention=topn,
        )
        debug_print("[DEBUG] Loaded linker via pyobo.get_scispacy_entity_linker", debug)
        return linker
    except Exception as exc:
        print(
            f"High-level PyOBO linker construction failed for '{ontology_prefix}': {exc!r}",
            file=sys.stderr,
            flush=True,
        )

    try:
        kb = pyobo.get_scispacy_knowledgebase(ontology_prefix)
        linker = EntityLinker.from_kb(
            kb,
            filter_for_definitions=False,
            max_entities_per_mention=topn,
        )
        debug_print("[DEBUG] Loaded linker via fallback EntityLinker.from_kb", debug)
        return linker
    except Exception as exc:
        print(
            f"Fallback KB-based linker construction also failed for '{ontology_prefix}': {exc!r}",
            file=sys.stderr,
            flush=True,
        )
        raise RuntimeError(
            f"Could not load PyOBO/scispaCy linker for ontology '{ontology_prefix}'."
        ) from exc


def build_linking_nlp():
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns([{"label": "MENTION", "pattern": [{"TEXT": {"REGEX": ".+"}}]}])
    return nlp


def build_abbreviation_nlp():
    try:
        import scispacy.abbreviation 
    except ImportError:
        print(
            "Warning: scispacy.abbreviation is unavailable; continuing without abbreviation expansion.",
            file=sys.stderr,
            flush=True,
        )
        return None

    nlp = spacy.blank("en")
    try:
        nlp.add_pipe("abbreviation_detector")
    except Exception as exc:
        print(
            f"Warning: could not load abbreviation_detector ({exc}); continuing without abbreviation expansion.",
            file=sys.stderr,
            flush=True,
        )
        return None
    return nlp


def get_passage_abbreviation_map(passage_text: str, abbr_nlp, debug: bool):
    if abbr_nlp is None:
        return {}

    passage_text = (passage_text or "").strip()
    if not passage_text:
        return {}

    try:
        doc = abbr_nlp(passage_text)
    except Exception as exc:
        debug_print(f"[DEBUG] Abbreviation parsing failed: {exc!r}", debug)
        return {}

    abbr_map = {}
    abbreviations = getattr(doc._, "abbreviations", []) or []

    for abbr in abbreviations:
        short_form = ""
        long_form = ""

        if hasattr(abbr, "_"):
            short_form = str(abbr).strip()
            long_form_obj = getattr(abbr._, "long_form", None)
            if long_form_obj is not None:
                long_form = str(long_form_obj).strip()
        elif isinstance(abbr, dict):
            short_form = str(abbr.get("short_form") or abbr.get("abbr") or abbr.get("text") or "").strip()
            long_form = str(
                abbr.get("long_form") or abbr.get("long") or abbr.get("definition") or ""
            ).strip()
        elif isinstance(abbr, (tuple, list)) and len(abbr) >= 2:
            short_form = str(abbr[0]).strip()
            long_form = str(abbr[1]).strip()

        if short_form and long_form and short_form not in abbr_map:
            abbr_map[short_form] = long_form

    if abbr_map:
        debug_print(f"[DEBUG] Abbreviation map: {abbr_map}", debug)

    return abbr_map


def ent_has_kb_ents(ent) -> bool:
    return hasattr(ent._, "kb_ents") and bool(ent._.kb_ents)


def normalize_with_pyobo(linker, mention_text, mention_nlp, topn, score_threshold, debug, abbreviation_map=None):
    abbreviation_map = abbreviation_map or {}
    mention_text = (mention_text or "").strip()
    if not mention_text:
        debug_print("[DEBUG] Empty mention text encountered", debug)
        return []

    candidate_texts = []
    expanded = abbreviation_map.get(mention_text)
    if expanded and expanded != mention_text:
        candidate_texts.append(expanded)
    candidate_texts.append(mention_text)

    seen_ids = set()
    all_results = []

    debug_print(f"[DEBUG] Linking mention: {mention_text!r}", debug)
    debug_print(f"[DEBUG] Candidate texts: {candidate_texts}", debug)

    for text in candidate_texts:
        text = text.strip()
        if not text:
            continue

        doc = mention_nlp(text)
        if not doc.ents and len(doc) > 0:
            span = doc.char_span(0, len(text), label="MENTION", alignment_mode="expand")
            if span is not None:
                doc.ents = [span]

        debug_print(f"[DEBUG] Pre-link entities for {text!r}: {[ent.text for ent in doc.ents]}", debug)

        try:
            doc = linker(doc)
        except Exception as exc:
            print(f"Linking failed for mention {text!r}: {exc!r}", file=sys.stderr, flush=True)
            continue

        debug_print(f"[DEBUG] Post-link entities for {text!r}: {[ent.text for ent in doc.ents]}", debug)

        if not doc.ents:
            debug_print(f"[DEBUG] No entities returned for: {text!r}", debug)
            continue

        ent = doc.ents[0]
        if not ent_has_kb_ents(ent):
            debug_print(f"[DEBUG] No kb_ents for: {text!r}", debug)
            continue

        debug_print(f"[DEBUG] Raw candidates for {text!r}: {ent._.kb_ents[:topn]}", debug)

        for identifier, score in ent._.kb_ents:
            if score < score_threshold:
                continue

            normalized_identifier = normalize_identifier_case(identifier)
            canonical_name = normalized_identifier
            kb_entry = linker.kb.cui_to_entity.get(identifier)
            if kb_entry is None:
                kb_entry = linker.kb.cui_to_entity.get(identifier.lower())
            if kb_entry is None:
                kb_entry = linker.kb.cui_to_entity.get(normalized_identifier)
            if kb_entry is None:
                kb_entry = linker.kb.cui_to_entity.get(normalized_identifier.lower())
            if kb_entry is not None and getattr(kb_entry, "canonical_name", None):
                canonical_name = kb_entry.canonical_name

            if normalized_identifier not in seen_ids:
                seen_ids.add(normalized_identifier)
                all_results.append((normalized_identifier, canonical_name, float(score)))

    all_results.sort(key=lambda item: item[2], reverse=True)
    if all_results:
        debug_print(f"[DEBUG] Final ranked hits for {mention_text!r}: {all_results[:topn]}", debug)
    else:
        debug_print(f"[DEBUG] No matches found for {mention_text!r}", debug)
    return all_results


def get_annotation_text(annotation: bioc.BioCAnnotation, passage: bioc.BioCPassage) -> str:
    text = (annotation.text or "").strip()
    if text:
        return text

    if not annotation.locations or passage.text is None:
        return ""

    location = annotation.locations[0]
    start = location.offset - passage.offset
    end = start + location.length
    if start < 0 or end > len(passage.text):
        return ""
    return passage.text[start:end].strip()


def process_collection(input_xml, output_xml, linker, abbr_nlp, mention_nlp, args) -> None:
    with input_xml.open("r", encoding="utf-8") as readfp:
        collection = bioc.load(readfp)

    mention_cache = {}
    processed_annotations = 0
    matched_annotations = 0
    missing_annotations = 0
    debug_counter = 0

    for document in collection.documents:
        debug_print(f"[DEBUG] Processing document id={document.id}", args.debug)

        for passage in document.passages:
            if not passage.infons.get("annotatable", True):
                debug_print("[DEBUG] Skipping non-annotatable passage", args.debug)
                continue

            abbreviation_map = get_passage_abbreviation_map(passage.text or "", abbr_nlp, args.debug)
            abbr_items = tuple(sorted(abbreviation_map.items()))

            for annotation in passage.annotations:
                if annotation.infons.get("type") == "cell_vague":
                    debug_print("[DEBUG] Skipping cell_vague annotation", args.debug)
                    continue

                processed_annotations += 1
                mention_text = get_annotation_text(annotation, passage)
                gold_identifier = annotation.infons.get("identifier", "")

                if args.debug and debug_counter < args.debug_max_print:
                    debug_print(
                        f"\n[DEBUG] Annotation #{processed_annotations}"
                        f"\n  type={annotation.infons.get('type')}"
                        f"\n  mention={mention_text!r}"
                        f"\n  gold={gold_identifier!r}",
                        args.debug,
                    )
                    debug_counter += 1

                if not mention_text:
                    missing_annotations += 1
                    debug_print("[DEBUG] Missing mention text", args.debug)
                    continue

                cache_key = (mention_text, abbr_items)
                if cache_key not in mention_cache:
                    mention_cache[cache_key] = normalize_with_pyobo(
                        linker=linker,
                        mention_text=mention_text,
                        mention_nlp=mention_nlp,
                        topn=args.topn,
                        score_threshold=args.score_threshold,
                        debug=args.debug,
                        abbreviation_map=abbreviation_map,
                    )
                else:
                    debug_print(f"[DEBUG] Cache hit for mention: {mention_text!r}", args.debug)

                normalized_hits = mention_cache[cache_key]
                if not normalized_hits:
                    missing_annotations += 1
                    debug_print(
                        f"[DEBUG] NO MATCH for mention={mention_text!r} gold={gold_identifier!r}",
                        args.debug,
                    )
                    continue

                matched_annotations += 1
                debug_print(f"[DEBUG] MATCH for mention={mention_text!r} -> {normalized_hits[:3]}", args.debug)

                for rank, (identifier, canonical_name, score) in enumerate(normalized_hits[: args.topn]):
                    annotation.infons[f"{MODEL_NAME}_id_{rank}"] = identifier
                    annotation.infons[f"{MODEL_NAME}_identifier_name_{rank}"] = canonical_name
                    annotation.infons[f"{MODEL_NAME}_identifier_score_{rank}"] = f"{score:.6f}"

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    with output_xml.open("w", encoding="utf-8") as writefp:
        bioc.dump(collection, writefp)

    print(f"PyOBO ontology prefix: {args.ontology_prefix}", flush=True)
    print(f"Unique mention/context queries: {len(mention_cache)}", flush=True)
    print(
        "Processed {} annotations, matched {}, missing {}".format(
            processed_annotations,
            matched_annotations,
            missing_annotations,
        ),
        flush=True,
    )
    print(f"Saved output to {output_xml}", flush=True)


def main() -> None:
    args = parse_args()
    if not args.input_xml.is_file():
        raise FileNotFoundError(f"Missing input XML: {args.input_xml}")

    use_abbreviations = not args.disable_abbreviations
    print(f"Input XML: {args.input_xml}", flush=True)
    print(f"Output XML: {args.output_xml}", flush=True)
    print(f"Ontology prefix: {args.ontology_prefix}", flush=True)
    print(f"TopN: {args.topn}", flush=True)
    print(f"Score threshold: {args.score_threshold}", flush=True)
    print(f"Abbreviations: {'ON' if use_abbreviations else 'OFF'}", flush=True)

    linker = load_celltype_linker(args.ontology_prefix, args.topn, args.debug)
    abbr_nlp = build_abbreviation_nlp() if use_abbreviations else None
    mention_nlp = build_linking_nlp()

    process_collection(
        input_xml=args.input_xml,
        output_xml=args.output_xml,
        linker=linker,
        abbr_nlp=abbr_nlp,
        mention_nlp=mention_nlp,
        args=args,
    )


if __name__ == "__main__":
    main()


# Example:
#   python predict_NEN.py \
#     --input-xml ../model_outputs/scispacy/predictions.xml \
#     --output-xml ../model_outputs/scispacy/normalized.xml
