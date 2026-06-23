# Training and Evaluation

This document is for training runs and evaluation. 

## Notes about Data 

All datasets are used for named entity recognition evaluation. However, only `BioID`, `CRAFT`, and `CellLink` are used for named entity normalization and end-to-end evaluation, because the remaining datasets do not provide Cell Ontology identifier ground-truth labels.

## Important note on the CellLink corpus

For CellLink data, the original test split does not provide ground-truth normalization labels. Therefore, this repository uses the validation split for evaluation. The CellLink data provided in this repository for evaluation correspond to the validation split from the original source. Vague cell-type populations are excluded for both training and evaluation. Although the data in this repository include `cell_vague` annotations, the NER fine-tuning and evaluation code exclude them.


## Fine-Tuning

### NER fine-tuning

```bash
python recognition/src/run_fine_tune_NER.py \
  --train-xml dataset \
  --output-dir recognition/models/CellExLink-bioformer16L \
  --model-name-or-path bioformers/bioformer-16L
```

- When `--train-xml` is set to the `dataset` folder, the script merges the training data from the files under that directory by simple concatenation.

### NEN fine-tuning

```bash
python normalization/fine_tune_NEN.py \
  --train-pairs normalization/sapbert_training_pairs.txt \
  --output-dir normalization/models/CellExLink-Sapbert
```

## Evaluation

### NER-only evaluation

```bash
python evaluation/strict_relax_NER.py \
  --reference_path dataset/BioID/test.xml \
  --prediction_path path/to/ner_predictions.xml \
  --evaluation_method strict
```

Supported evaluation styles are `strict` and `relax`.

By default, `strict_relax_NER.py` excludes `cell_vague` annotations.

### NEN evaluation

```bash
python evaluation/run_eval.py \
  --dataset other \
  --reference-path dataset/BioID/test.xml \
  --prediction-path path/to/normalized.xml \
  --model-names CellExLink-Sapbert \
  --score-mode end_to_end
```

Useful `--score-mode` values:

- `gold_mention_normalize`: score normalization on gold mentions only
- `end_to_end`: score exact-span recognition and normalization jointly
- `--dataset celllink`: use the CellLink evaluation setting

### Runtime evaluation

```bash
python evaluation/run_time_eval.py \
  --task ner \
  --input-xml dataset/JNLPBA/test.xml \
  --model-path models/CellExLink-bioformer16L
```

Task choices:

- `--task ner`: runtime for mention recognition
- `--task el`: runtime for normalization

## Other Baselines

Baseline are documented in [../other_baselines/README.md](../other_baselines/README.md).
