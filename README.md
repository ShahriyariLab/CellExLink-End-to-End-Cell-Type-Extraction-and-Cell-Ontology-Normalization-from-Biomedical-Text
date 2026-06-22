# CellExLink: End-to-End Cell-Type Recognition and Normalization in Biomedical Text

`CellExLink` is a Python pipeline for end-to-end cell-type extraction from biomedical text.

The inference pipeline has two stages:

1. `NER`: detect cell-type mention spans
2. `NEN`: assign Cell Ontology (`CL`) identifiers to detected mentions

## Installation

```bash
git clone https://github.com/ShahriyariLab/CellExLink-End-to-End-Cell-Type-Extraction-and-Cell-Ontology-Normalization-from-Biomedical-Text.git CellExLink
cd CellExLink

conda create -n cellexlink python=3.12 -y
conda activate cellexlink
python -m pip install -r requirements.txt
```

## Model Download
The default model checkpoints used by the recognition and ontology-linking components are hosted on Hugging Face and are not stored in this repository.

Default checkpoints:

- Recognition model: [`almire/CellExLink-bioformer16L`](https://huggingface.co/almire/CellExLink-bioformer16L)
- Ontology-linking embedding-retrieval model: [`almire/CellExLink-Sapbert`](https://huggingface.co/almire/CellExLink-Sapbert)

Download the default checkpoints with:

```bash
python -m pip install huggingface_hub
python download_models.py
```

This creates:

```text
models/
  CellExLink-bioformer16L/
  CellExLink-Sapbert/
  models.json
```

## Usage

CellExLink accepts unannotated [BioC XML](https://bioc.sourceforge.net/) documents, including:

- PubMed titles and abstracts obtained through the NCBI BioC PubMed API
- Eligible PMC full-text articles obtained through the NCBI BioC PMC API

Example input files are provided in:

```text
examples/bioc_abstracts/
examples/bioc_fulltext/
```

Run CellExLink on PubMed abstracts

```bash
python prediction_script.py \
    examples/bioc_abstracts \
    --output-root examples/bioc_abstracts_annotated
```

Run CellExLink on PMC full-text articles

```bash
python prediction_script.py \
    examples/bioc_fulltext \
    --output-root examples/bioc_fulltext_annotated
```

The command runs the complete end-to-end workflow. It detects cell-type mentions and assigns a Cell Ontology identifier to each detected span. The original BioC document structure and text are preserved, and the predictions are added as BioC XML annotations.

For example, an input passage containing:

```xml
<passage>
  <infon key="type">title</infon>
  <offset>0</offset>
  <text>B lymphocytes: how they develop and function.</text>
</passage>
```
is returned with an annotation such as:

```xml
<annotation id="T1">
  <infon key="type">cell_type</infon>
  <infon key="CellExLink-Sapbert_id_0">CL:0000236</infon>
  <infon key="CellExLink-Sapbert_identifier_name_0">B lymphocyte</infon>
  <location offset="0" length="13"/>
  <text>B lymphocytes</text>
</annotation>
```

Precomputed outputs for the supplied examples are available in:

```text
examples/bioc_abstracts_annotated/
examples/bioc_fulltext_annotated/
```

Run the end-to-end pipeline on a single BioC XML file:

```bash
python prediction_script.py examples/input_bioc.xml
```
Precomputed output is available in:

```tesxt
examples/input_bioc_normalized.xml
```

## Additional Guidance

Additional repository guides are available for developers for training and benchmark evaluation.

- [Training and benchmark evaluation](docs/TRAINING_EVALUATION.md)
- [Baseline methods](other_baselines/README.md)


## Data

The training and test datasets shared in this repository are available from Zenodo and the associated paper resource: <https://doi.org/10.5281/zenodo.18090009>


Please follow the dataset's own license, citation, and redistribution terms.

## Questions and Issues

For usage questions or bug reports, please open a GitHub issue and include:

- the command or script you ran
- the input format
- the relevant error message
- your operating system and Python version
