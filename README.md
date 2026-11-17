# CSC394 Challenge 1: Partial Freezing for Multilingual PoS Tagging

A reproducible comparison of full fine-tuning and partial freezing for Universal Dependencies Chinese GSD UPOS tagging.

## Experiment

- Data: official UD Chinese GSD Simplified train/dev files; fixed-seed samples of 600 and 150 sentences.
- Model: distilbert/distilbert-base-multilingual-cased.
- Metric: word-level UPOS accuracy, counting the first subword only.
- Baseline: train every parameter. Frozen condition: freeze embeddings and DistilBERT layers 0–2; train layers 3–5 and the classification head.
- Shared setup: seed 42, AdamW, learning rate 2e-5, batch size 16, three epochs, fresh pretrained checkpoint per condition.

## Measured results

The reference run used a Tesla T4. Full fine-tuning reached **87.55%** (3,283/3,750 words); partial freezing reached **85.57%** (3,209/3,750 words), a 1.97 percentage-point gap on this sample. The frozen setup trained 21.3M of 134.7M parameters. This is one seed and one language, so it is a preliminary comparison rather than a general conclusion.

See [report.md](report.md) for interpretation and limitations, and [results.json](results.json) for exact metrics and loss values.

## Run

Open [the notebook in Colab](https://colab.research.google.com/github/billchenziang33/csc394-pos-partial-freezing/blob/main/partial_freezing_pos_tagging.ipynb) and run each cell in order. The notebook installs Transformers, checks out the script, downloads the public model and UD data, and saves results to results.json.

To run locally with Python 3.10–3.12:

~~~bash
python -m pip install -r requirements.txt
python run_experiment.py --max-train 600 --max-dev 150 --epochs 3 --batch-size 16 --learning-rate 2e-5 --seed 42
~~~

A CUDA GPU is recommended; CPU execution works but may take longer.

## Sources

- [Universal Dependencies — Chinese GSD Simplified](https://github.com/UniversalDependencies/UD_Chinese-GSDSimp)
- [DistilBERT multilingual cased](https://huggingface.co/distilbert/distilbert-base-multilingual-cased)
- [Hugging Face token-classification guide](https://huggingface.co/docs/transformers/tasks/token_classification)
- Sanh et al. (2019), [DistilBERT paper](https://arxiv.org/abs/1910.01108)
