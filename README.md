# CSC394 Challenge 1: Partial Freezing for Multilingual PoS Tagging

This repository contains a small, reproducible comparison of full fine-tuning and partial freezing for Universal Dependencies Chinese GSD PoS tagging.

## Research question

Does freezing the embeddings and the lower half of a multilingual DistilBERT encoder change UPOS accuracy on a fixed Chinese GSD development split, compared with fine-tuning every parameter?

## Method

- Dataset: official UD Chinese-GSD Simplified train and dev CoNLL-U files.
- Model: `distilbert/distilbert-base-multilingual-cased`.
- Task: predict one UPOS label per UD word. Only the first subword for each word contributes to the loss and accuracy.
- Comparison: both runs start from a fresh copy of the same pretrained checkpoint and use the same sampled train/dev examples, seed, batch size, learning rate, and number of epochs.
- Partial-freezing choice: freeze embeddings and encoder layers 0–2 (three of DistilBERT's six layers); train layers 3–5 and the classification head. This keeps lower-level multilingual representations fixed while allowing upper contextual layers to adapt.
- Metric: word-level UPOS accuracy on the official development split. The development data is not used for gradient updates.

The notebook runs both experiments and saves the measured scores to `results.json`. It does not include fabricated or pre-filled accuracy values.

## Run

Python 3.10–3.12 is recommended. From this folder:

```bash
python -m pip install -r requirements.txt
jupyter notebook partial_freezing_pos_tagging.ipynb
```

Run all notebook cells from top to bottom. The first run downloads the dataset and model weights. A CUDA GPU is recommended; CPU execution is supported but may take substantially longer.

The notebook is the main, step-by-step entry point. `run_experiment.py` contains the implementation and can also be run directly:

```bash
python run_experiment.py --max-train 600 --max-dev 150 --epochs 3 --seed 42
```

## Results

No accuracy claims are made until the notebook or script has completed successfully. Add the two values from `results.json` to the report and explain the observed difference after the run.

## Limitations

This is a small, single-seed experiment on one language and one UD treebank. It is useful as a controlled challenge experiment, not as evidence that partial freezing is generally better. A stronger follow-up would repeat several seeds, report variance, and evaluate additional languages.

## References

- Nivre, J. et al. (2020). *Universal Dependencies v2: An Evergrowing Multilingual Treebank Collection.* LREC.
- Sanh, V. et al. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.* NeurIPS EMC² Workshop.
- Hugging Face Transformers documentation: https://huggingface.co/docs/transformers/tasks/token_classification
