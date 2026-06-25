"""Normalize detected cell mentions to Cell Ontology identifiers.

Input:
- A Cell Ontology dictionary in JSONL format.
- An optional abbreviation TSV file.
- One or more BioC XML files that already contain mention annotations.

Output:
- BioC XML files with CL identifiers and matched ontology names written into
  the annotation metadata.
"""

import json
import argparse
import re
import time
from pathlib import Path
from collections import Counter, defaultdict

import bioc
import numpy as np
import torch
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
try:
    from .plural_stemmer import normalize_text as plural_normalize_text
except ImportError:  # pragma: no cover
    from plural_stemmer import normalize_text as plural_normalize_text

try:
    import pyab3p
except ImportError:  # pragma: no cover
    pyab3p = None


DEFAULT_TOPN = 1
AMBIGUOUS_TOPN = 5

ABBREVIATION_HEADER = [
    "short_form",
    "matched_cl_id",
]

DASH_PATTERN = r"[\-\u2010\u2011\u2012\u2013\u2014\u2212]"


def normalize_text(text):
    return " ".join(str(text).casefold().split())


def normalize_abbreviation_key(text):
    text = str(text).strip()
    text = re.sub(DASH_PATTERN, "", text)
    text = re.sub(r"\s+", "", text)
    return text


def abbreviation_variant_keys(text):
    key = normalize_abbreviation_key(text)
    variants = []
    seen = set()

    def add(x):
        if x and x not in seen:
            seen.add(x)
            variants.append(x)

    add(key)

    if key.endswith("s") and len(key) > 1:
        add(key[:-1])
    else:
        add(key + "s")

    return variants


def abbreviation_threshold_for(mention_text):
    n = len(normalize_abbreviation_key(mention_text))
    if n <= 4:
        return 1.0
    if n <= 7:
        return 0.95
    return 0.90


def ab3p_fuzzy_threshold_for(short_form_text):
    n = len(normalize_abbreviation_key(short_form_text))
    if n <= 4:
        return 1.0
    if n <= 7:
        return 0.95
    return 0.90


def token_jaccard(left, right):
    left_tokens = set(normalize_text(left).split())
    right_tokens = set(normalize_text(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def sequence_ratio(left, right):
    from difflib import SequenceMatcher
    return SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio()


def abbreviation_sequence_ratio(left, right):
    from difflib import SequenceMatcher
    left_norm = normalize_abbreviation_key(left)
    right_norm = normalize_abbreviation_key(right)
    if not left_norm or not right_norm:
        return 0.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def has_parenthetical_relation(left, right):
    left = str(left)
    right = str(right)

    if "(" in left and ")" in left:
        inside = re.findall(r"\(([^()]*)\)", left)
        if any(normalize_text(x) == normalize_text(right) for x in inside):
            return 1.0

    if "(" in right and ")" in right:
        inside = re.findall(r"\(([^()]*)\)", right)
        if any(normalize_text(x) == normalize_text(left) for x in inside):
            return 1.0

    return 0.0


def is_abbreviation_like(text):
    raw = str(text).strip()
    if not raw:
        return False

    compact = normalize_abbreviation_key(raw)
    if len(compact) <= 1:
        return False

    has_upper = any(ch.isupper() for ch in raw)
    has_digit = any(ch.isdigit() for ch in raw)
    has_symbol = any(ch in "+/-" for ch in raw)
    no_space_shortish = len(compact) <= 12 and " " not in raw

    return has_upper or has_digit or has_symbol or no_space_shortish


def is_sentence_transformers_model(model_name_or_path):
    model_path = Path(model_name_or_path)
    return model_path.is_dir() and (model_path / "modules.json").is_file()


def load_encoder(model_name_or_path):
    if is_sentence_transformers_model(model_name_or_path):
        from sentence_transformers import SentenceTransformer

        print("Loading sentence-transformers model:", model_name_or_path)
        model = SentenceTransformer(model_name_or_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device:", device)
        model = model.to(device)
        return {
            "kind": "sentence_transformers",
            "model": model,
            "device": device,
        }

    print("Loading transformers model:", model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = model.to(device)
    model.eval()
    return {
        "kind": "transformers",
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
    }


def encode_names(encoder, names, batch_size=128):
    all_reps = []

    if encoder["kind"] == "sentence_transformers":
        model = encoder["model"]
        for i in tqdm(np.arange(0, len(names), batch_size), desc="Encoding names"):
            batch_embeddings = model.encode(
                names[i:i + batch_size],
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,
            )
            all_reps.append(batch_embeddings)
    else:
        tokenizer = encoder["tokenizer"]
        model = encoder["model"]
        device = encoder["device"]

        with torch.no_grad():
            for i in tqdm(np.arange(0, len(names), batch_size), desc="Encoding names"):
                toks = tokenizer.batch_encode_plus(
                    names[i:i + batch_size],
                    padding="max_length",
                    max_length=32,
                    truncation=True,
                    return_tensors="pt",
                )
                toks = {k: v.to(device) for k, v in toks.items()}
                output = model(**toks)
                cls_rep = output[0][:, 0, :]
                all_reps.append(cls_rep.cpu().detach().numpy())

    if len(all_reps) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    return np.concatenate(all_reps, axis=0)


def encode_queries(encoder, queries):
    if encoder["kind"] == "sentence_transformers":
        return encoder["model"].encode(
            queries,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )

    tokenizer = encoder["tokenizer"]
    model = encoder["model"]
    device = encoder["device"]

    toks = tokenizer.batch_encode_plus(
        queries,
        padding="max_length",
        max_length=32,
        truncation=True,
        return_tensors="pt",
    )
    toks = {k: v.to(device) for k, v in toks.items()}

    with torch.no_grad():
        output = model(**toks)

    return output[0][:, 0, :].cpu().detach().numpy()


def topk(array, k, axis=-1, sorted=True):
    partitioned_ind = (
        np.argpartition(array, -k, axis=axis)
        .take(indices=range(-k, 0), axis=axis)
    )
    partitioned_scores = np.take_along_axis(array, partitioned_ind, axis=axis)

    if sorted:
        sorted_trunc_ind = np.flip(np.argsort(partitioned_scores, axis=axis), axis=axis)
        ind = np.take_along_axis(partitioned_ind, sorted_trunc_ind, axis=axis)
        scores = np.take_along_axis(partitioned_scores, sorted_trunc_ind, axis=axis)
    else:
        ind = partitioned_ind
        scores = partitioned_scores

    return scores, ind


def load_terms(filename):
    """Load ontology names and synonyms from the JSONL dictionary file.

    Input:
    - `filename`: path to the ontology JSONL file.

    Output:
    - `term_entries`: flat list used for embedding search.
    - `concept_metadata`: grouped metadata for each ontology concept.
    """
    term_entries = []
    concept_metadata = {}

    with open(filename, "r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON on line {line_no} in {filename}: {exc}") from exc

            identifier = record.get("norm_concept_id")
            preferred_label = record.get("norm_preferred_label")
            synonyms = record.get("synonyms", []) or []
            namespace = record.get("namespace", "")

            if not identifier or not preferred_label:
                continue

            if identifier not in concept_metadata:
                concept_metadata[identifier] = {
                    "preferred_label": preferred_label,
                    "synonyms": set(),
                    "names": set(),
                    "namespace": namespace,
                }

            concept_metadata[identifier]["names"].add(preferred_label)

            for syn in synonyms:
                if syn:
                    concept_metadata[identifier]["synonyms"].add(syn)
                    concept_metadata[identifier]["names"].add(syn)

            term_entries.append({
                "name": plural_normalize_text(preferred_label),
                "raw_name": preferred_label,
                "identifier": identifier,
                "preferred_label": preferred_label,
                "is_preferred": True,
            })

            for syn in synonyms:
                if syn:
                    term_entries.append({
                        "name": plural_normalize_text(syn),
                        "raw_name": syn,
                        "identifier": identifier,
                        "preferred_label": preferred_label,
                        "is_preferred": False,
                    })

    return term_entries, concept_metadata


def classify_abbreviation_path(abbr_path):
    if abbr_path in [None, "", "."]:
        return None

    path = Path(abbr_path)
    if not path.is_file():
        return None

    if path.suffix != ".tsv":
        return "other"

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            fields = line.split("\t")
            if fields[:2] == ABBREVIATION_HEADER:
                return "short_form_identifier_tsv"
            return "other"

    return "other"


def load_abbreviation_identifier_lookup(abbr_paths, verbose=True):
    """Read trusted short-form to identifier mappings from TSV files.

    Input:
    - `abbr_paths`: one TSV path or a list of TSV paths.
    - `verbose`: whether to print loading details.

    Output:
    - A lookup dictionary containing direct and ambiguous abbreviation matches.
    """
    if isinstance(abbr_paths, str):
        abbr_paths = [abbr_paths]

    row_counts = Counter()
    key_to_candidates = defaultdict(list)

    for abbr_path in abbr_paths:
        if classify_abbreviation_path(abbr_path) != "short_form_identifier_tsv":
            continue

        path = Path(abbr_path)
        if verbose:
            print("Loading abbreviation lookup from", path)

        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle):
                line = line.rstrip("\n")
                if not line:
                    continue

                fields = line.split("\t")
                if line_no == 0 and fields[:2] == ABBREVIATION_HEADER:
                    continue
                if len(fields) < 2:
                    continue

                short_form, matched_cl_id = fields[:2]
                short_form = short_form.strip()
                matched_cl_id = matched_cl_id.strip()

                if not short_form:
                    continue
                if matched_cl_id in ["", "-", "None", "none"]:
                    continue

                first_id = re.split(r"[,;]", matched_cl_id)[0].strip()
                if first_id in ["", "-", "None", "none"]:
                    continue

                key = normalize_abbreviation_key(short_form)
                if not key:
                    continue

                candidate = {
                    "short_form": short_form,
                    "key": key,
                    "identifier": first_id,
                }

                key_to_candidates[key].append(candidate)
                row_counts[key] += 1

    direct_lookup = {}
    ambiguous_candidates = {}

    for key, candidates in key_to_candidates.items():
        unique_ids = {c["identifier"] for c in candidates}
        if len(unique_ids) == 1:
            best = max(candidates, key=lambda c: len(c["short_form"]))
            direct_lookup[key] = (best["short_form"], best["identifier"])
        else:
            ambiguous_candidates[key] = candidates

    all_keys = sorted(key_to_candidates.keys())

    if verbose and key_to_candidates:
        print(
            "Loaded {} safe abbreviation mappings; {} ambiguous abbreviation keys".format(
                len(direct_lookup),
                len(ambiguous_candidates),
            )
        )

    return {
        "direct_lookup": direct_lookup,
        "ambiguous_candidates": ambiguous_candidates,
        "all_keys": all_keys,
        "key_to_candidates": key_to_candidates,
        "row_counts": row_counts,
    }


def get_document_key(document, passage):
    key = (
        document.id
        or passage.infons.get("article-id_pmid")
        or passage.infons.get("passage_id")
    )
    if key is None:
        raise ValueError("Could not determine a stable document key for normalization")
    return str(key)


def get_mention_records(input_filename):
    """Collect unique mention texts from a BioC XML file.

    Input:
    - `input_filename`: BioC XML file with mention annotations.

    Output:
    - A set of unique `(document_key, normalized_mention)` pairs.
    - A mapping from document key to full document text.
    - The total number of annotated mentions seen in the file.
    """
    mentions = set()
    document_text_by_key = {}
    annotation_count = 0

    print("Loading mention texts from file", input_filename)
    with open(input_filename, "r", encoding="utf-8") as fp:
        input_collection = bioc.load(fp)

    for document in input_collection.documents:
        document_key = None
        document_passages = []

        for passage in document.passages:
            if not passage.infons.get("annotatable", True):
                continue

            passage_text = passage.text or ""
            if not passage_text:
                continue

            if document_key is None:
                document_key = get_document_key(document, passage)

            document_passages.append(passage_text)

            for annotation in passage.annotations:
                if annotation.infons["type"] != "cell_vague":
                    annotation_count += 1
                    mentions.add(
                        (
                            document_key,
                            plural_normalize_text(annotation.text or ""),
                        )
                    )

        if document_key is not None:
            document_text_by_key[document_key] = "\n".join(document_passages)

    print(
        "Collected {} unique mentions from {} annotations in {}".format(
            len(mentions),
            annotation_count,
            input_filename,
        )
    )
    return mentions, document_text_by_key, annotation_count


def normalize_pyab3p_output(results):
    if isinstance(results, dict):
        return [(short_form, long_form) for short_form, long_form in results.items()]

    if isinstance(results, list):
        pairs = []
        for item in results:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                short_form = str(item[0]).strip()
                long_form = str(item[1]).strip()
            elif isinstance(item, dict):
                short_form = str(
                    item.get("short_form")
                    or item.get("short")
                    or item.get("abbr")
                    or item.get("abbreviation")
                    or ""
                ).strip()
                long_form = str(
                    item.get("long_form")
                    or item.get("long")
                    or item.get("expansion")
                    or ""
                ).strip()
            else:
                short_form = str(getattr(item, "short_form", getattr(item, "short", ""))).strip()
                long_form = str(getattr(item, "long_form", getattr(item, "long", ""))).strip()

            if short_form and long_form:
                pairs.append((short_form, long_form))
        return pairs

    return []


def build_document_abbreviation_lookup(document_text_by_key, target_document_keys, verbose=True):
    """Use Ab3P to find short-form and long-form pairs inside documents.

    Input:
    - `document_text_by_key`: full text for each document.
    - `target_document_keys`: documents that need abbreviation help.
    - `verbose`: whether to print progress details.

    Output:
    - A lookup from document key to abbreviation expansions found in that document.
    """
    if not target_document_keys:
        return {}

    if pyab3p is None:
        if verbose:
            print("pyab3p is not installed; skipping document-context abbreviation expansion.")
        return {}

    detector = pyab3p.Ab3p()
    lookup = {}

    for document_key in tqdm(sorted(target_document_keys), desc="Running Ab3P over full documents"):
        document_text = document_text_by_key.get(document_key, "")
        if not document_text.strip():
            continue

        try:
            results = detector.get_abbrs(document_text)
        except Exception as exc:
            if verbose:
                print(f"WARN Ab3P failed for document {document_key}: {exc}")
            continue

        pairs = normalize_pyab3p_output(results)
        if not pairs:
            continue

        lookup[document_key] = {
            normalize_abbreviation_key(short_form): long_form.strip()
            for short_form, long_form in pairs
            if short_form and long_form
        }

    if verbose:
        print("Built Ab3P document lookup for {} documents".format(len(lookup)))
    return lookup


def get_topn_term_results(distances, term_entries, k):
    top_scores, top_indices = topk(-distances, k)
    results = []

    for ii in range(len(top_indices[0])):
        idx = top_indices[0][ii]
        score = float(top_scores[0][ii])
        entry = term_entries[idx]
        results.append({
            "name": entry["name"],
            "raw_name": entry["raw_name"],
            "identifier": entry["identifier"],
            "preferred_label": entry["preferred_label"],
            "is_preferred": entry["is_preferred"],
            "embedding_score": score,
        })

    return results


def retrieve_term_candidates(encoder, dictionary_embeddings, term_entries, query_text, topn, initial_k=None):
    """Retrieve the best ontology concepts for one mention text.

    Input:
    - Encoded ontology embeddings, ontology term entries, and one query string.
    - `topn`: number of concepts to keep after grouping synonyms.

    Output:
    - A list of the highest-scoring concept candidates for the query.
    """
    if initial_k is None:
        initial_k = max(topn * 10, 20)

    query_rep = encode_queries(encoder, [query_text])
    dists = cdist(query_rep, dictionary_embeddings, metric="cosine")

    synonym_hits = get_topn_term_results(dists, term_entries, initial_k)

    best_by_concept = {}
    for hit in synonym_hits:
        identifier = hit["identifier"]
        prev = best_by_concept.get(identifier)
        if prev is None or hit["embedding_score"] > prev["embedding_score"]:
            best_by_concept[identifier] = hit

    concept_hits = sorted(
        best_by_concept.values(),
        key=lambda x: x["embedding_score"],
        reverse=True,
    )

    return concept_hits[:topn]


def rerank_term_candidates(query_text, candidates, concept_metadata):
    reranked = []
    query_norm = normalize_text(query_text)

    for cand in candidates:
        identifier = cand["identifier"]
        meta = concept_metadata.get(identifier, {})
        names = set(meta.get("names", set()))
        preferred_label = meta.get("preferred_label", cand["preferred_label"])

        best_name_overlap = 0.0
        exact_synonym_match = 0.0
        best_parenthetical = 0.0
        best_seq = 0.0

        for name in names:
            if query_norm == normalize_text(name):
                exact_synonym_match = 1.0
            best_name_overlap = max(best_name_overlap, token_jaccard(query_text, name))
            best_parenthetical = max(best_parenthetical, has_parenthetical_relation(query_text, name))
            best_seq = max(best_seq, sequence_ratio(query_text, name))

        preferred_overlap = token_jaccard(query_text, preferred_label)

        final_score = (
            1.00 * cand["embedding_score"]
            + 0.35 * exact_synonym_match
            + 0.20 * best_name_overlap
            + 0.15 * preferred_overlap
            + 0.10 * best_parenthetical
            + 0.05 * best_seq
            + (0.03 if cand["is_preferred"] else 0.0)
        )

        reranked.append({
            **cand,
            "rerank_score": final_score,
            "exact_synonym_match": exact_synonym_match,
            "token_overlap": max(best_name_overlap, preferred_overlap),
            "parenthetical_match": best_parenthetical,
            "sequence_ratio": best_seq,
        })

    reranked.sort(key=lambda x: (x["rerank_score"], x["embedding_score"]), reverse=True)
    return reranked


def prepare_abbreviation_index(encoder, abbreviation_lookup):
    abbr_keys = abbreviation_lookup["all_keys"]
    if not abbr_keys:
        return None

    abbr_embeddings = encode_names(encoder, abbr_keys)
    return {
        "keys": abbr_keys,
        "embeddings": abbr_embeddings,
    }


def find_best_abbreviation_key(
    encoder,
    abbreviation_index,
    abbreviation_lookup,
    mention_text,
    mention_embedding_cache=None,
    result_cache=None,
):
    if not abbreviation_lookup or not abbreviation_index:
        return None

    if mention_embedding_cache is None:
        mention_embedding_cache = {}
    if result_cache is None:
        result_cache = {}

    cache_key = mention_text
    if cache_key in result_cache:
        return result_cache[cache_key]

    mention_key = normalize_abbreviation_key(mention_text)
    if not mention_key:
        result_cache[cache_key] = None
        return None

    threshold = abbreviation_threshold_for(mention_text)

    if threshold == 1.0:
        if mention_key in abbreviation_lookup["key_to_candidates"]:
            result = {
                "matched_key": mention_key,
                "score": 1.0,
                "method": "exact_short_abbreviation",
            }
            result_cache[cache_key] = result
            return result

        result_cache[cache_key] = None
        return None

    if mention_key in mention_embedding_cache:
        query_rep = mention_embedding_cache[mention_key]
    else:
        query_rep = encode_queries(encoder, [mention_key])
        mention_embedding_cache[mention_key] = query_rep

    dists = cdist(query_rep, abbreviation_index["embeddings"], metric="cosine")
    best_idx = int(np.argmin(dists[0]))
    best_score = float(1.0 - dists[0][best_idx])
    best_key = abbreviation_index["keys"][best_idx]

    if best_score >= threshold:
        result = {
            "matched_key": best_key,
            "score": best_score,
            "method": "encoder_abbreviation_match",
        }
        result_cache[cache_key] = result
        return result

    result_cache[cache_key] = None
    return None


def find_ab3p_long_form_with_fallback(doc_lookup, matched_key):
    if not doc_lookup:
        return None, None, None, None

    exact_hit = doc_lookup.get(matched_key)
    if exact_hit:
        return exact_hit, "ab3p_exact_key", matched_key, 1.0

    for variant_key in abbreviation_variant_keys(matched_key):
        if variant_key == matched_key:
            continue
        variant_hit = doc_lookup.get(variant_key)
        if variant_hit:
            return variant_hit, "ab3p_exact_variant", variant_key, 1.0

    threshold = ab3p_fuzzy_threshold_for(matched_key)
    best_key = None
    best_score = -1.0

    for doc_key in doc_lookup.keys():
        score = abbreviation_sequence_ratio(matched_key, doc_key)
        if score > best_score:
            best_score = score
            best_key = doc_key

    if best_key is not None and best_score >= threshold:
        return doc_lookup[best_key], "ab3p_fuzzy_shortform", best_key, float(best_score)

    return None, None, None, None


def resolve_abbreviation(
    encoder,
    abbreviation_index,
    abbreviation_lookup,
    document_abbreviation_lookup,
    mention_text,
    document_key,
    mention_embedding_cache=None,
    result_cache=None,
):
    """Resolve a short mention through abbreviation rules before normal retrieval.

    Input:
    - Encoder state, abbreviation indexes, document abbreviation lookups, and one mention.

    Output:
    - A dictionary describing the resolved abbreviation match, or `None` if no
      safe abbreviation-based match is available.
    """
    match = find_best_abbreviation_key(
        encoder=encoder,
        abbreviation_index=abbreviation_index,
        abbreviation_lookup=abbreviation_lookup,
        mention_text=mention_text,
        mention_embedding_cache=mention_embedding_cache,
        result_cache=result_cache,
    )
    if match is None:
        return None

    matched_key = match["matched_key"]
    score = match["score"]
    method = match["method"]

    direct_match = abbreviation_lookup["direct_lookup"].get(matched_key)
    if direct_match is not None:
        return {
            "type": "direct",
            "matched_key": matched_key,
            "score": score,
            "method": method,
            "short_form": direct_match[0],
            "identifier": direct_match[1],
        }

    candidates = abbreviation_lookup["ambiguous_candidates"].get(matched_key, [])
    if not candidates:
        return None

    doc_lookup = document_abbreviation_lookup.get(document_key, {})
    expanded_long_form, ab3p_method, ab3p_matched_key, ab3p_score = find_ab3p_long_form_with_fallback(
        doc_lookup,
        matched_key,
    )

    if not expanded_long_form:
        return {
            "type": "ambiguous_unresolved",
            "matched_key": matched_key,
            "score": score,
            "method": method,
        }

    return {
        "type": "ambiguous_long_form",
        "matched_key": matched_key,
        "score": score,
        "method": method,
        "expanded_long_form": expanded_long_form,
        "ab3p_method": ab3p_method,
        "ab3p_matched_key": ab3p_matched_key,
        "ab3p_match_score": ab3p_score,
    }


def get_jsonl_total_tokens(json_path):
    total_tokens = 0
    with open(json_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {json_path}") from exc

            tokens = record.get("tokens", [])
            if isinstance(tokens, list):
                total_tokens += len(tokens)
    return total_tokens


def format_runtime_summary(
    inference_elapsed,
    total_mentions_processed,
    total_tokens_for_reporting=None,
):
    print("\n=== EL Predict-Only Runtime Summary ===")
    print(f"Predict-only wall time: {inference_elapsed:.6f} seconds")
    print(f"Total mentions processed: {total_mentions_processed}")

    if total_mentions_processed > 0 and inference_elapsed > 0:
        mentions_per_second = total_mentions_processed / inference_elapsed
        ms_per_mention = (inference_elapsed * 1000.0) / total_mentions_processed
        print(f"Throughput: {mentions_per_second:.4f} mentions/second")
        print(f"Latency: {ms_per_mention:.4f} ms/mention")
    elif total_mentions_processed > 0:
        print("Throughput: inf mentions/second")
        print("Latency: 0.0000 ms/mention")

    if total_tokens_for_reporting is not None and total_tokens_for_reporting > 0:
        ms_per_1k_tokens = (inference_elapsed * 1000.0) / (total_tokens_for_reporting / 1000.0)
        print(f"Latency: {ms_per_1k_tokens:.4f} ms/1k tokens")


def run_inference(
    model_name,
    term_entries,
    concept_metadata,
    mentions,
    abbreviation_lookup=None,
    document_abbreviation_lookup=None,
    el_warmup_runs=1,
):
    """Normalize all unique mentions by retrieving the best ontology concepts.

    Input:
    - Model path, ontology term data, unique mentions, and optional abbreviation help.

    Output:
    - A dictionary mapping each mention to its best normalization result.
    - A runtime summary dictionary with elapsed time and mention count.
    """
    encoder = load_encoder(model_name)

    print("Encoding CL dictionary names/synonyms")
    dictionary_names = [entry["name"] for entry in term_entries]
    dictionary_embeddings = encode_names(encoder, dictionary_names)

    abbreviation_index = None
    if abbreviation_lookup and abbreviation_lookup["all_keys"]:
        print("Encoding abbreviation short forms with the same encoder")
        abbreviation_index = prepare_abbreviation_index(encoder, abbreviation_lookup)

    mentions_list = list(mentions)
    if el_warmup_runs < 0:
        raise ValueError("el_warmup_runs must be >= 0")

    def run_mentions_loop(collect_outputs=True):
        """Run one pass over all mentions, optionally keeping the predictions."""
        local_normalized = {} if collect_outputs else None
        query_cache = {}
        mention_embedding_cache = {}
        abbreviation_result_cache = {}
        stats = Counter()

        for document_id, mention_text in tqdm(mentions_list, desc="Running EL inference", leave=False):
            result_record = None

            if is_abbreviation_like(mention_text) and abbreviation_lookup:
                abbr_resolution = resolve_abbreviation(
                    encoder=encoder,
                    abbreviation_index=abbreviation_index,
                    abbreviation_lookup=abbreviation_lookup,
                    document_abbreviation_lookup=document_abbreviation_lookup or {},
                    mention_text=mention_text,
                    document_key=document_id,
                    mention_embedding_cache=mention_embedding_cache,
                    result_cache=abbreviation_result_cache,
                )

                if abbr_resolution is not None:
                    if abbr_resolution["type"] == "direct":
                        result_record = [{
                            "name": abbr_resolution["short_form"],
                            "identifier": abbr_resolution["identifier"],
                            "embedding_score": abbr_resolution["score"],
                            "final_score": abbr_resolution["score"],
                            "source": "abbreviation_direct",
                            "abbreviation_method": abbr_resolution["method"],
                        }]
                        stats["abbreviation_direct"] += 1

                    elif abbr_resolution["type"] == "ambiguous_long_form":
                        query_text = plural_normalize_text(abbr_resolution["expanded_long_form"])
                        cache_key = ("ambiguous_long_form", query_text)

                        if cache_key in query_cache:
                            candidates = query_cache[cache_key]
                        else:
                            retrieved = retrieve_term_candidates(
                                encoder,
                                dictionary_embeddings,
                                term_entries,
                                query_text,
                                AMBIGUOUS_TOPN,
                                initial_k=50,
                            )
                            candidates = rerank_term_candidates(
                                query_text,
                                retrieved,
                                concept_metadata,
                            )
                            query_cache[cache_key] = candidates

                        if candidates:
                            result_record = [{
                                "name": candidates[0]["raw_name"],
                                "identifier": candidates[0]["identifier"],
                                "embedding_score": candidates[0]["embedding_score"],
                                "final_score": candidates[0]["rerank_score"],
                                "source": "abbreviation_ambiguous_via_long_form",
                                "abbreviation_method": abbr_resolution["method"],
                                "expanded_long_form": abbr_resolution["expanded_long_form"],
                                "ab3p_method": abbr_resolution.get("ab3p_method"),
                                "ab3p_matched_key": abbr_resolution.get("ab3p_matched_key"),
                                "ab3p_match_score": abbr_resolution.get("ab3p_match_score"),
                            }]
                            stats["abbreviation_ambiguous_via_long_form"] += 1

                            ab3p_method = abbr_resolution.get("ab3p_method")
                            if ab3p_method:
                                stats[f"{ab3p_method}"] += 1
                        else:
                            stats["abbreviation_ambiguous_unresolved"] += 1

                    elif abbr_resolution["type"] == "ambiguous_unresolved":
                        stats["abbreviation_ambiguous_unresolved"] += 1

            if result_record is None:
                query_text = plural_normalize_text(mention_text)
                cache_key = ("normal", query_text)

                if cache_key in query_cache:
                    candidates = query_cache[cache_key]
                else:
                    retrieved = retrieve_term_candidates(
                        encoder,
                        dictionary_embeddings,
                        term_entries,
                        query_text,
                        DEFAULT_TOPN,
                        initial_k=20,
                    )
                    candidates = rerank_term_candidates(
                        query_text,
                        retrieved,
                        concept_metadata,
                    )
                    query_cache[cache_key] = candidates

                if candidates:
                    result_record = [{
                        "name": candidates[0]["raw_name"],
                        "identifier": candidates[0]["identifier"],
                        "embedding_score": candidates[0]["embedding_score"],
                        "final_score": candidates[0]["rerank_score"],
                        "source": "model_normal",
                    }]
                    stats["model_normal"] += 1
                else:
                    if collect_outputs:
                        local_normalized[(document_id, mention_text)] = None
                    stats["no_candidate"] += 1
                    continue

            if collect_outputs:
                local_normalized[(document_id, mention_text)] = result_record

        return local_normalized, stats

    for warmup_idx in range(el_warmup_runs):
        print(f"EL warmup run {warmup_idx + 1}/{el_warmup_runs}")
        _warmup_outputs, _warmup_stats = run_mentions_loop(collect_outputs=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_start = time.perf_counter()

    normalized, stats = run_mentions_loop(collect_outputs=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_elapsed = time.perf_counter() - inference_start

    print(f"\n=== EL Predict-Only Runtime Summary ({model_name}) ===")
    format_runtime_summary(
        inference_elapsed=inference_elapsed,
        total_mentions_processed=len(mentions_list),
    )

    print("Inference summary for model {}:".format(model_name))
    for key, value in sorted(stats.items()):
        print("  {} = {}".format(key, value))
    print("  total_mentions = {}".format(len(mentions_list)))

    del encoder
    return normalized, {
        "elapsed_seconds": float(inference_elapsed),
        "total_mentions_processed": int(len(mentions_list)),
    }


def process_collection(
    input_filename,
    models_results,
    output_filename,
    include_identifier_scores=False,
):
    """Write normalization results back into a BioC XML collection.

    Input:
    - One input BioC XML file, model predictions, and one output path.

    Output:
    - A new BioC XML file with normalization fields added to each mention annotation.
    """
    print("Processing file", input_filename, "to", output_filename)

    with open(input_filename, "r", encoding="utf-8") as fp:
        bioc_collection = bioc.load(fp)

    processed_annotations = 0
    matched_annotations = 0
    missing_annotations = 0
    warning_budget = 10

    for document in bioc_collection.documents:
        for passage in document.passages:
            passage_text = passage.text or ""
            if not passage.infons.get("annotatable", True) or len(passage_text) == 0:
                continue

            document_key = get_document_key(document, passage)

            for annotation in passage.annotations:
                if annotation.infons["type"] != "cell_vague":
                    processed_annotations += 1
                    normalized_text = plural_normalize_text(annotation.text or "")
                    found_match = False

                    for model_name, normalized in models_results.items():
                        topn_results = normalized.get((document_key, normalized_text))
                        if not topn_results:
                            if warning_budget > 0:
                                print(
                                    "WARN: No normalized identifier found for document_key = {} mention text = {}".format(
                                        document_key,
                                        normalized_text,
                                    )
                                )
                                warning_budget -= 1
                        else:
                            found_match = True
                            best = topn_results[0]

                            annotation.infons[model_name + "_id_0"] = best["identifier"]
                            annotation.infons[model_name + "_identifier_name_0"] = best["name"]
                            if include_identifier_scores:
                                annotation.infons[model_name + "_identifier_score_0"] = float(
                                    best.get("final_score", best.get("embedding_score", 0.0))
                                )

                            if "expanded_long_form" in best:
                                annotation.infons[model_name + "_expanded_long_form"] = best["expanded_long_form"]
                            if "ab3p_method" in best and best["ab3p_method"] is not None:
                                annotation.infons[model_name + "_ab3p_method"] = best["ab3p_method"]
                            if "ab3p_matched_key" in best and best["ab3p_matched_key"] is not None:
                                annotation.infons[model_name + "_ab3p_matched_key"] = best["ab3p_matched_key"]
                            if "ab3p_match_score" in best and best["ab3p_match_score"] is not None:
                                annotation.infons[model_name + "_ab3p_match_score"] = float(best["ab3p_match_score"])

                    if found_match:
                        matched_annotations += 1
                    else:
                        missing_annotations += 1

    with open(output_filename, "w", encoding="utf-8") as fp:
        bioc.dump(bioc_collection, fp)

    print(
        "Processed {} annotations, matched {}, missing {}".format(
            processed_annotations,
            matched_annotations,
            missing_annotations,
        )
    )


def paths_to_filenames(input_paths, output_paths):
    """Expand matching input and output path lists for file or directory mode.

    Input:
    - Input paths and output paths from the command line.

    Output:
    - Two aligned filename lists used by the main normalization loop.
    """
    new_input_paths = []
    new_output_paths = []

    for input_path, output_path in zip(input_paths, output_paths):
        input_path = str(input_path)
        output_path = str(output_path)

        if Path(input_path).is_dir() and Path(output_path).is_dir():
            new_input_paths += [
                str(Path(input_path) / filename.name)
                for filename in sorted(Path(input_path).iterdir())
                if filename.is_file()
            ]
            new_output_paths += [
                str(Path(output_path) / filename.name)
                for filename in sorted(Path(input_path).iterdir())
                if filename.is_file()
            ]
        elif Path(input_path).is_file() and not Path(output_path).is_dir():
            new_input_paths.append(input_path)
            new_output_paths.append(output_path)
        else:
            raise Exception("both input and output path must be either directory or file")

    return new_input_paths, new_output_paths


def main(
    term_filename,
    abbr_paths,
    input_paths,
    output_paths,
    model_names,
    abbr_verbose=True,
    el_warmup_runs=1,
    include_identifier_scores=False,
):
    """Run ontology normalization for one file or a batch of BioC XML files.

    Input:
    - Dictionary path, abbreviation path(s), input/output path(s), and model settings.

    Output:
    - Writes normalized BioC XML files to disk.
    """
    if isinstance(abbr_paths, str):
        abbr_paths = [abbr_paths]
    if isinstance(input_paths, str):
        input_paths = [input_paths]
    if isinstance(output_paths, str):
        output_paths = [output_paths]

    input_paths, output_paths = paths_to_filenames(input_paths, output_paths)

    abbreviation_lookup = load_abbreviation_identifier_lookup(abbr_paths, verbose=abbr_verbose)

    print("Loading CL dictionary JSONL")
    term_entries, concept_metadata = load_terms(term_filename)
    print(
        "Loaded {} dictionary entries across {} concepts".format(
            len(term_entries),
            len(concept_metadata),
        )
    )

    mentions = set()
    document_text_by_key = {}
    total_annotation_count = 0

    for input_filename, _ in zip(input_paths, output_paths):
        file_mentions, file_document_text_by_key, file_annotation_count = get_mention_records(input_filename)
        mentions.update(file_mentions)
        document_text_by_key.update(file_document_text_by_key)
        total_annotation_count += file_annotation_count

    print(f"Total unique mentions for EL inference: {len(mentions)}")
    print(f"Total annotation mentions in corpus: {total_annotation_count}")

    ambiguous_keys = set(abbreviation_lookup["ambiguous_candidates"].keys()) if abbreviation_lookup else set()

    target_document_keys = set()
    if ambiguous_keys:
        for document_key, mention_text in mentions:
            mention_key = normalize_abbreviation_key(mention_text)
            if mention_key in ambiguous_keys or is_abbreviation_like(mention_text):
                target_document_keys.add(document_key)

    document_abbreviation_lookup = build_document_abbreviation_lookup(
        document_text_by_key,
        target_document_keys,
        verbose=abbr_verbose,
    )

    models_results = {}
    for model_nickname, model_fullname in model_names.items():
        model_results, _runtime_summary = run_inference(
            model_fullname,
            term_entries,
            concept_metadata,
            mentions,
            abbreviation_lookup=abbreviation_lookup,
            document_abbreviation_lookup=document_abbreviation_lookup,
            el_warmup_runs=el_warmup_runs,
        )
        models_results[model_nickname] = model_results
        print(
            "Model {} produced {} normalized mention entries".format(
                model_nickname,
                len(models_results[model_nickname]),
            )
        )

    print("Going to process files")
    for input_filename, output_filename in zip(input_paths, output_paths):
        process_collection(
            input_filename,
            models_results,
            output_filename,
            include_identifier_scores=include_identifier_scores,
        )

    print("Done.")


def parse_args():
    """Read command-line arguments for the normalization stage.

    Input:
    - Values passed on the command line.

    Output:
    - An `argparse.Namespace` with dictionary, abbreviation, input, output,
      and runtime settings.
    """
    parser = argparse.ArgumentParser(description="Normalize CellLink mentions in BioC XML.")
    parser.add_argument("term_filename", help="Cell ontology JSONL file.")
    parser.add_argument("abbr_paths", nargs="?", default="", help="Abbreviations TSV file.")
    parser.add_argument("input_paths", help="Input BioC XML file or directory.")
    parser.add_argument("output_paths", help="Output BioC XML file or directory.")
    parser.add_argument(
        "--model-path",
        default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        help="Model name or local model directory.",
    )
    parser.add_argument(
        "--abbr-verbose",
        action="store_true",
        help="Print abbreviation loading and Ab3P details.",
    )
    parser.add_argument(
        "--disable-abbreviations",
        action="store_true",
        help="Ignore abbreviation file and abbreviation matching.",
    )
    parser.add_argument(
        "--el-warmup-runs",
        type=int,
        default=1,
        help="Number of EL warmup runs before timing mention normalization.",
    )
    parser.add_argument(
        "--include-identifier-scores",
        action="store_true",
        help="Write `*_identifier_score_0` fields for evaluation workflows.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_name = Path(str(args.model_path)).name
    abbr_paths = "" if args.disable_abbreviations else args.abbr_paths

    main(
        term_filename=args.term_filename,
        abbr_paths=abbr_paths,
        input_paths=args.input_paths,
        output_paths=args.output_paths,
        model_names={model_name: args.model_path},
        abbr_verbose=args.abbr_verbose,
        el_warmup_runs=args.el_warmup_runs,
        include_identifier_scores=args.include_identifier_scores,
    )
