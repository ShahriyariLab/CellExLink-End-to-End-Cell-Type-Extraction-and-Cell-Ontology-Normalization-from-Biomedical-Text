### scispacy
scispaCy requires a separate Python environment with the following dependencies:
- `en_ner_craft_md`
- `Python < 3.11`
- `scipy < 1.11`
#### NER prediction
```bash
python other_baselines/scispacy/predict_NER.py \
  --input-xml dataset/CRAFT/test.xml \
  --output-xml model_outputs/scispacy/predictions.xml
```

#### NEN prediction

```bash
python other_baselines/scispacy/predict_NEN.py \
  --input-xml model_outputs/scispacy/predictions.xml \
  --output-xml model_outputs/scispacy/normalized.xml
```

### BERN2 and VANER2

BERN2 and VANER2 require separate Python environments. For local installation instructions, refer to the official repositories:

- [BERN2](https://github.com/dmis-lab/BERN2)
- [VANER2](https://github.com/ZhuLab-Fudan/VANER2/tree/main)

For BERN2, we separated the NER and NEN steps in our experiments. After generating predictions, follow the same evaluation process provided in our codebase. 