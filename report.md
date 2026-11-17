# CSC394 Challenge 1 — Partial Freezing for Chinese PoS Tagging

## Research question

How does freezing lower layers of a multilingual language model affect its UPOS tagging accuracy on a small Chinese treebank sample, compared with fine-tuning the full model?

## Data and setup

I used the official Universal Dependencies Chinese GSD Simplified treebank. I sampled 600 training sentences and 150 development sentences with a fixed seed. The development sample contains 3,750 word-level UPOS labels. I used `distilbert/distilbert-base-multilingual-cased` with a token-classification head. For Chinese words split into multiple subwords, the loss and accuracy use only the first subword; special tokens and continuation subwords are ignored.

Both conditions started from a fresh copy of the same pretrained checkpoint and used the same sampled data, seed (42), optimizer (AdamW), learning rate (`2e-5`), batch size (16), and three epochs. The baseline fine-tuned all parameters. The partial-freezing condition kept the embeddings and the first three of DistilBERT's six transformer layers fixed, while training the upper three layers and classification head.

## Results

| Condition | Development word accuracy | Correct / total words | Trainable parameters | Mean training loss by epoch |
| --- | ---: | ---: | ---: | --- |
| Full fine-tuning | 87.55% | 3,283 / 3,750 | 134,747,153 / 134,747,153 | 1.9214, 0.7462, 0.4111 |
| Freeze embeddings + layers 0–2 | 85.57% | 3,209 / 3,750 | 21,276,689 / 134,747,153 | 2.1683, 1.0663, 0.5588 |

In this run, partial freezing scored 1.97 percentage points below full fine-tuning (74 fewer correct development words). It trained about 15.8% of the model's parameters. The script recorded 16.6 seconds for the full-fine-tuning condition and 8.2 seconds for partial freezing on a Tesla T4; these timings describe this small run only.

## Interpretation

Full fine-tuning performed better on this sample. One possible explanation is that updating all layers let the model adapt its lower-level representations to the Chinese treebank, while freezing those layers limited that adaptation. The smaller trainable parameter count also makes the frozen setup cheaper to optimize. This experiment alone cannot tell whether the accuracy gap is stable or caused partly by the sample and seed.

## Limitations and next steps

This is a single-language, single-seed experiment using a subset of the official train and development splits. It does not establish that full fine-tuning is generally better or that partial freezing is ineffective. I would next repeat both conditions with several seeds, report mean and standard deviation, and test another language to see whether the result changes with dataset size and language structure.

## Reproduction

Run `partial_freezing_pos_tagging.ipynb` from top to bottom, or install `requirements.txt` and run:

```bash
python run_experiment.py --max-train 600 --max-dev 150 --epochs 3 --batch-size 16 --learning-rate 2e-5 --seed 42
```

The run writes the measured configuration and scores to `results.json`.

## Sources

- Challenge prompt: [Partial Freezing of MLMs for PoS Tagging](https://docs.google.com/document/d/1F0yPVs7YxExzqPiCESockILN-6e5NWRxtUNda0WuOqc)
- Data: [Universal Dependencies — Chinese GSD Simplified](https://github.com/UniversalDependencies/UD_Chinese-GSDSimp)
- Model: [DistilBERT multilingual cased](https://huggingface.co/distilbert/distilbert-base-multilingual-cased)
- Token classification guide: [Hugging Face Transformers](https://huggingface.co/docs/transformers/tasks/token_classification)
- Sanh et al. (2019), [DistilBERT](https://arxiv.org/abs/1910.01108)
