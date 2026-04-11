# CellExLink


### Prepraration

```bash
conda create -n CellExlink python=3.12
conda activate CellExlink
git clone https://github.com/CellExLink
cd CellExLink
pip install -r requirements.txt
```
This repository provides the code required to fine-tune the CellExLink NER and NEN models. Alternatively, the fine-tuned checkpoints can be downloaded directly from Hugging Face.
Download the fine-tuned checkpoints from Hugging Face and place them in the appropriate directories.

For the CellExLink NER component, download the checkpoint from [here](https://huggingface.co/almire/CellExLink-bioformer16L) and place it under `recognition/models/`. or 

```bash
huggingface-cli download almire/CellExLink-bioformer16L --local-dir ./recognition/models/CellExLink-bioformer16L
```

For the CellExLink NEN component, download the checkpoint from [here](https://huggingface.co/almire/CellExLink-Sapbert) and place it under `normalization/models/`. or 

```bash
huggingface-cli download almire/CellExLink-Sapbert --local-dir ./normalization/models/CellExLink-Sapbert
```

### Using CellExLink

#### NER prediction
```bash
from recognition import predict_ner

predict_ner(
    model_path="recognition/models/CellExLink-bioformer16L",
    input_xml="dataset/BioID/test.xml",
    output_dir="model_outputs/ner",
    output_xml="model_outputs/ner/ner_predictions.xml",
)
```
#### NEN prediction
```bash
from normalization import normalize_bioc

normalize_bioc(
    input_xml="model_outputs/ner/ner_predictions.xml",
    #input_xml="dataset/BioID/test.xml"  # for gold mention normalization
    output_xml="model_outputs/normalized.xml",
    model_path="normalization/models/CellExLink-Sapbert",
)
```
### Fine-tuning
Fine-tune the NER model:

```bash
python recognition/src/run_fine_tune.py \
  --train-xml dataset/train_data.xml \
  --output-dir recognition/models/CellExLink-bioformer16L
  --model-name-or-path bioformers/bioformer-16L 
```
Fine-tune the NEN model:

```bash
python normalization/fine_tune_NEN.py \
  --train-pairs normalization/train_data/sapbert_training_pairs.txt \
  --output-dir normalization/models/CellExLink-Sapbertx
```

### Evaluation
Run NER-only evaluation:

```bash
python evaluation/run_eval_NER_only.py \
  --reference-path dataset/BioID/test.xml \
  --prediction-path model_outputs/ner/ner_predictions.xml \
  --evaluation-method strict
```

Evaluation method can be `strict` or `relax`.

Run NEN evaluation:

```bash
python evaluation/run_eval.py \
  --dataset other \
  --reference-path dataset/BioID/test.xml \
  --prediction-path model_outputs/normalized.xml \
  --model-names CellExLink-Sapbert \
  --score-mode end_to_end
```

- `--score-mode gold_mention_normalize`: evaluates only the correctness of the normalized identifier for gold-standard mentions.
- `--score-mode end_to_end`: evaluates both exact span matching and the correctness of the normalized identifier.

Run time evaluation:

```bash

python evaluation/run_time_eval.py \
  --task ner \
  --input-xml dataset/JNLPBA/test.xml \
  --model-path recognition/models/celllink_bioformer 
```
- `--task ner`: evaluation for named entity recognition (NER)
- `--task el`: evaluation for named entity normalization (NEN)


### Datasets
The CellLink, BioID, AnatEM, CRAFT, and JNLPBA datasets used in this project are available from the original Zenodo record: [link](https://zenodo.org/records/18090009).

Download the datasets from the original source and place them in the `dataset/` directory before running training or evaluation.

Use of these datasets is subject to the license and terms specified by the original source. Please refer to the Zenodo record for citation and licensing information.