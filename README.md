# CSC394 Challenge 1: Partial Freezing for Multilingual PoS Tagging

This project compares full fine-tuning with partial freezing for word-level UPOS tagging on Universal Dependencies Chinese GSD Simplified.

## Experiment

- Data: 600 training and 150 development sentences per seed, sampled from the official UD train/dev splits.
- Model: distilbert/distilbert-base-multilingual-cased.
- Metric: word-level UPOS accuracy; only the first subword is scored.
- Baseline: fine-tune all parameters. Partial freezing: freeze embeddings and DistilBERT layers 0–2; train layers 3–5 and the classification head.
- Pairing: for each seed, both conditions use the same sampled train/dev data, matching classifier-head initialization, and matching training shuffle order. Each condition starts from the same pretrained checkpoint.
- Training setup: AdamW, learning rate 2e-5, batch size 16, three epochs.

## Three-seed results

Mean and sample standard deviation are calculated from the three per-seed accuracies; dev word counts differ slightly because each seed samples different sentences.

| Condition | Mean word accuracy | Sample SD | Mean training time | Trainable parameters |
| --- | ---: | ---: | ---: | ---: |
| Full fine-tuning | 88.42% | 1.02 pp | 17.7 s | 134.7M / 134.7M |
| Freeze embeddings + layers 0–2 | 86.56% | 1.03 pp | 9.4 s | 21.3M / 134.7M |

Full fine-tuning scored higher in all three paired runs. The average paired gap (partial freezing minus full fine-tuning) was **−1.87 percentage points**. This is still a small, single-language experiment and does not establish that one method is generally better.

| Seed | Full fine-tuning | Partial freezing | Paired gap |
| ---: | ---: | ---: | ---: |
| 42 | 87.55% | 85.57% | −1.97 pp |
| 43 | 89.55% | 87.63% | −1.91 pp |
| 44 | 88.18% | 86.46% | −1.72 pp |

See [report.md](report.md) for the interpretation and limitations, and [results.json](results.json) for per-seed metrics and training losses.

## Run

Open [the notebook in Colab](https://colab.research.google.com/github/billchenziang33/csc394-pos-partial-freezing/blob/main/partial_freezing_pos_tagging.ipynb) and run its cells. To repeat the three paired seeds locally:

~~~bash
python -m pip install -r requirements.txt
python run_repeated_experiments.py --seeds 42 43 44
~~~

To run only one seed, use the command python run_experiment.py --seed 42. A CUDA GPU is recommended; CPU execution may take longer.

## Sources

- [Universal Dependencies — Chinese GSD Simplified](https://github.com/UniversalDependencies/UD_Chinese-GSDSimp)
- [DistilBERT multilingual cased](https://huggingface.co/distilbert/distilbert-base-multilingual-cased)
- [Hugging Face token-classification guide](https://huggingface.co/docs/transformers/tasks/token_classification)
- Sanh et al. (2019), [DistilBERT paper](https://arxiv.org/abs/1910.01108)
