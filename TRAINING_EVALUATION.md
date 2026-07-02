# Training and Evaluation

This guide covers data preparation, model training, and evaluation for CellExLink.

## Data Notes

All datasets in the repository can be used for named entity recognition evaluation.
Only `BioID`, `CRAFT`, and `CellLink` are used for named entity normalization and end-to-end evaluation, because the other datasets do not include Cell Ontology ground-truth identifiers.

## CellLink Notes

For CellLink, the original test split does not include gold normalization labels.
Because of that, this repository uses the validation split for normalization and end-to-end evaluation.

`cell_vague` annotations are excluded from training and evaluation.

## Data Preparation for NER

Use [recognition/src/data_processing.py](/Users/alma/CellExLink-End-to-End-Cell-Type-Extraction-and-Cell-Ontology-Normalization-from-Biomedical-Text/recognition/src/data_processing.py) if you want to prepare one combined BioC XML training file before running training.

This script:

- reads one BioC XML file or many corpus folders
- simply concatenates the `train.xml` training sets under `dataset/`
- writes one combined BioC XML file

Example:

```bash
python recognition/src/data_processing.py \
  dataset \
  --output recognition/data/train_all.xml
```

## Fine-Tuning

### NER Fine-Tuning

Recommended workflow with explicit preprocessing:

```bash
python recognition/src/data_processing.py \
  dataset \
  --output recognition/data/train_all.xml

python recognition/src/run_fine_tune_NER.py \
  --train-xml recognition/data/train_all.xml \
  --output-dir recognition/models/CellExLink-bioformer16L \
  --model-name-or-path bioformers/bioformer-16L
```

You can also train directly from BioC XML:

```bash
python recognition/src/run_fine_tune_NER.py \
  --train-xml dataset \
  --output-dir recognition/models/CellExLink-bioformer16L \
  --model-name-or-path bioformers/bioformer-16L
```

### What Joint Fine-Tuning Does

With the repository's standard `dataset/<corpus>/train.xml` layout, passing `--train-xml dataset` performs the following steps:

1. The converter selects `train.xml` from each direct corpus directory under `dataset/`. 
2. During training conversion, annotations whose type is `cell_vague` are skipped. 
3. CellLink labels `cell_phenotype` and `cell_hetero` are normalized to the shared `cell_type` label.
4. It concatenates the training data from the `train.xml` files. 

### NEN Fine-Tuning

```bash
python normalization/fine_tune_NEN.py \
  --train-pairs normalization/sapbert_training_pairs.txt \
  --output-dir normalization/models/CellExLink-Sapbert
```

## Evaluation

### NER-Only Evaluation

```bash
python evaluation/strict_relax_NER.py \
  --reference_path dataset/BioID/test.xml \
  --prediction_path path/to/ner_predictions.xml \
  --evaluation_method strict
```

Supported evaluation styles:

- `strict`
- `relax`

By default, `strict_relax_NER.py` excludes `cell_vague` annotations.

### NEN Evaluation

```bash
python evaluation/run_eval.py \
  --dataset other \
  --reference-path dataset/BioID/test.xml \
  --prediction-path path/to/normalized.xml \
  --model-names CellExLink-Sapbert \
  --score-mode end_to_end
```

Useful `--score-mode` values:

- `gold_mention_normalize`
- `end_to_end`

Use `--dataset celllink` for the CellLink evaluation setting.

### Runtime Evaluation

```bash
python evaluation/run_time_eval.py \
  --task ner \
  --input-xml dataset/JNLPBA/test.xml \
  --model-path models/CellExLink-bioformer16L
```

Task choices:

- `--task ner` for mention recognition runtime
- `--task el` for normalization runtime

## Other Baselines

Baseline methods are documented in [other_baselines/README.md](/Users/alma/CellExLink-End-to-End-Cell-Type-Extraction-and-Cell-Ontology-Normalization-from-Biomedical-Text/other_baselines/README.md).
