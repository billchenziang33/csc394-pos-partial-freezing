# CSC394 Challenge 1 — Partial Freezing for Chinese PoS Tagging

## Research question

How does freezing lower layers of a multilingual model affect word-level UPOS accuracy on Chinese, compared with full fine-tuning?

## Data and setup

I used the official Universal Dependencies Chinese GSD Simplified train and development splits. For each of three seeds (42, 43, 44), I sampled 600 training sentences and 150 development sentences. Each seed therefore has a different fixed sample. The resulting development sets contain 3,750, 3,817, and 3,899 word-level UPOS labels.

The model is distilbert/distilbert-base-multilingual-cased with a token-classification head. For words split into multiple subwords, loss and accuracy use only the first subword; special tokens and continuation subwords are ignored.

Each seed is a paired comparison: both conditions use the same sampled sentences, a fresh copy of the same pretrained checkpoint, matching classifier-head initialization, and the same training shuffle order. The baseline updates every parameter. The partial-freezing condition keeps the embeddings and the first three of DistilBERT's six layers fixed, and trains the upper three layers and classification head. Both use AdamW, learning rate 2e-5, batch size 16, and three epochs.

## Results

The table reports the mean and sample standard deviation of accuracy across the three seeds. The mean training time is an average of the recorded runs on a Tesla T4.

| Condition | Mean word accuracy | Sample SD | Mean training time | Trainable parameters |
| --- | ---: | ---: | ---: | ---: |
| Full fine-tuning | 88.42% | 1.02 percentage points | 17.7 s | 134,747,153 / 134,747,153 |
| Freeze embeddings + layers 0–2 | 86.56% | 1.03 percentage points | 9.4 s | 21,276,689 / 134,747,153 |

| Seed | Development words | Full fine-tuning | Partial freezing | Partial minus full |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 3,750 | 87.55% | 85.57% | −1.97 pp |
| 43 | 3,817 | 89.55% | 87.63% | −1.91 pp |
| 44 | 3,899 | 88.18% | 86.46% | −1.72 pp |

Full fine-tuning scored higher in all three paired runs. The average paired gap was −1.87 percentage points. The partially frozen model trained about 15.8% of the parameters and took about 47% less time in these short runs. These timings only describe this Colab T4 setup and should not be generalized to larger workloads or other hardware.

## Interpretation

On these samples, updating all layers gave better accuracy. One possible explanation is that the lower layers also benefited from adapting to the Chinese treebank. Partial freezing reduced the number of trainable parameters and run time, but did not preserve the full-fine-tuning accuracy in these runs. Three seeds make the comparison less dependent on one random sample, but the evidence is still limited.

## Limitations and next steps

This is one language, one model, one small subset size, and three seeds. Each seed changes both the sentence sample and the model's random initialization, so the reported standard deviation reflects both sources of variation. The results do not establish that full fine-tuning is generally better, or that freezing is ineffective. A useful next step would be to hold the data split fixed while varying only model seeds, then repeat the comparison on another language and a larger sample.

## Reproduction

Install requirements and run the three paired seeds:

~~~bash
python -m pip install -r requirements.txt
python run_repeated_experiments.py --seeds 42 43 44
~~~

The wrapper saves raw per-seed metrics and the mean/sample standard deviation to results.json. The single-seed experiment can also be run with python run_experiment.py --seed 42. The Colab notebook contains the executed run and a short results summary.

## Sources

- Challenge prompt: [Partial Freezing of MLMs for PoS Tagging](https://docs.google.com/document/d/1F0yPVs7YxExzqPiCESockILN-6e5NWRxtUNda0WuOqc)
- Data: [Universal Dependencies — Chinese GSD Simplified](https://github.com/UniversalDependencies/UD_Chinese-GSDSimp)
- Model: [DistilBERT multilingual cased](https://huggingface.co/distilbert/distilbert-base-multilingual-cased)
- Token alignment: [Hugging Face token classification guide](https://huggingface.co/docs/transformers/tasks/token_classification)
- Model background: Sanh et al. (2019), [DistilBERT](https://arxiv.org/abs/1910.01108)
