import argparse
from importlib import util
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
EVAL_PATH = BASE_DIR / "eval.py"
DEFAULT_MODEL_NAMES = ["SapBERT_finetuned"]

DATASET_CONFIGS = {
    "celllink": {
        "entity_type_sets": [
            ["cell_phenotype"],
            ["cell_hetero"],
            ["cell_type"],
            ["cell_phenotype", "cell_hetero"],
        ],
        "iterator_names": ["exactIDsOnly_iterator", "allLabels_iterator"],
    },
    "other": {
        "entity_type_sets": [["cell_type"]],
        "iterator_names": ["singleID_iterator"],
    },
}


def load_eval_module():
    spec = util.spec_from_file_location("el_eval_module", EVAL_PATH)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run EL evaluation for selected top-k outputs.")
    parser.add_argument(
        "--reference-path",
        type=Path,
        required=True,
        help="Reference BioC XML path for evaluation.",
    )
    parser.add_argument(
        "--prediction-path",
        type=Path,
        required=True,
        help="Prediction BioC XML path for evaluation.",
    )
    parser.add_argument(
        "--dataset",
        choices=["celllink", "other"],
        default="other",
        help="Which dataset-style iterator configuration to use.",
    )
    parser.add_argument(
        "--model-names",
        nargs="+",
        default=DEFAULT_MODEL_NAMES,
        help="One or more model name prefixes to score from the prediction XML.",
    )
    parser.add_argument(
        "--topk",
        choices=["top1", "top5", "top10", "all"],
        default="top1",
        help="Which top-k results to print.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Only score predictions whose identifier confidence is at least this value.",
    )
    parser.add_argument(
        "--score-mode",
        choices=["gold_mention_normalize", "end_to_end", "relax_end_to_end"],
        default="gold_mention_normalize",
        help="Score normalization on gold mentions, exact end-to-end matches, or relaxed end-to-end matches.",
    )
    parser.add_argument(
        "--prediction-text-blacklist",
        nargs="*",
        default=[],
        help=(
            "Prediction annotation texts that must never match during approximate span+normalization "
            "evaluation. They remain predictions, so they count as false positives if unmatched."
        ),
    )
    return parser


def filter_results(results, topk_choice):
    if topk_choice == "all":
        return results

    topk_map = {"top1": "1", "top5": "5", "top10": "10"}
    selected_k = topk_map[topk_choice]

    filtered = []
    for model_result in results:
        filtered.append(
            {
                "model_name": model_result["model_name"],
                "topk": [
                    topk_result
                    for topk_result in model_result["topk"]
                    if topk_result["k"] == selected_k
                ],
            }
        )
    return filtered


def main():
    args = build_arg_parser().parse_args()
    eval_module = load_eval_module()
    dataset_config = DATASET_CONFIGS[args.dataset]
    score_mode = args.score_mode

    reference_collection = eval_module.load_collection(args.reference_path)
    prediction_collection = eval_module.load_collection(args.prediction_path)

    for iterator_name in dataset_config["iterator_names"]:
        print("=" * 80)
        print(iterator_name)
        print("=" * 80)

        if args.threshold is not None:
            print(f"Applying confidence threshold >= {args.threshold}")

        if score_mode == "relax_end_to_end" and args.prediction_text_blacklist:
            print(
                "Blacklisted predicted texts will not be allowed to match in approx scoring: "
                f"{args.prediction_text_blacklist}"
            )

        for entity_types in dataset_config["entity_type_sets"]:
            results = eval_module.evaluate_iterator(
                reference_collection=reference_collection,
                iterator_name=iterator_name,
                model_names=args.model_names,
                entity_types=entity_types,
                score_mode=score_mode,
                prediction_collection=prediction_collection,
                prediction_text_blacklist=args.prediction_text_blacklist,
                score_threshold=args.threshold,
            )
            eval_module.print_results(
                iterator_name,
                entity_types,
                filter_results(results, args.topk),
            )


if __name__ == "__main__":
    main()


