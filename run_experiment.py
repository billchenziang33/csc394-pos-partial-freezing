"""Compare full fine-tuning with partial freezing for UD Chinese UPOS tagging."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import requests
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer


MODEL_NAME = "distilbert/distilbert-base-multilingual-cased"
UD_BASE = "https://raw.githubusercontent.com/UniversalDependencies/UD_Chinese-GSDSimp/master"
FILES = {
    "train": "zh_gsdsimp-ud-train.conllu",
    "dev": "zh_gsdsimp-ud-dev.conllu",
}
UPOS = (
    "ADJ ADP ADV AUX CCONJ DET INTJ NOUN NUM PART PRON PROPN PUNCT SCONJ"
    " SYM VERB X"
).split()
LABEL_TO_ID = {label: i for i, label in enumerate(UPOS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}


def download_conllu(split: str, data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / FILES[split]
    if not path.exists():
        response = requests.get(f"{UD_BASE}/{FILES[split]}", timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path


def read_conllu(path: Path) -> list[tuple[list[str], list[str]]]:
    sentences: list[tuple[list[str], list[str]]] = []
    words: list[str] = []
    tags: list[str] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if not line:
                if words:
                    sentences.append((words, tags))
                    words, tags = [], []
                continue
            if line.startswith("#"):
                continue
            columns = line.split("\t")
            # UD multiword tokens (e.g. 1-2) and empty nodes (e.g. 3.1)
            # are not syntactic words and therefore have no word-level UPOS target.
            if len(columns) >= 4 and columns[0].isdigit():
                form, upos = columns[1], columns[3]
                if upos in LABEL_TO_ID:
                    words.append(form)
                    tags.append(upos)
        if words:
            sentences.append((words, tags))
    return sentences


def sample_sentences(rows, limit: int, seed: int):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows[: min(limit, len(rows))]


def collate(batch, tokenizer):
    words_batch = [row[0] for row in batch]
    tags_batch = [row[1] for row in batch]
    encoded = tokenizer(
        words_batch,
        is_split_into_words=True,
        truncation=True,
        max_length=256,
        padding=True,
        return_tensors="pt",
    )
    labels = []
    for batch_index, tags in enumerate(tags_batch):
        word_ids = encoded.word_ids(batch_index=batch_index)
        previous_word_id = None
        aligned = []
        for word_id in word_ids:
            if word_id is None or word_id == previous_word_id:
                aligned.append(-100)
            else:
                aligned.append(LABEL_TO_ID[tags[word_id]])
            previous_word_id = word_id
        labels.append(aligned)
    encoded["labels"] = torch.tensor(labels, dtype=torch.long)
    return encoded


def freeze_lower_half(model):
    for parameter in model.distilbert.embeddings.parameters():
        parameter.requires_grad = False
    for layer in model.distilbert.transformer.layer[:3]:
        for parameter in layer.parameters():
            parameter.requires_grad = False


def train_and_evaluate(train_rows, dev_rows, tokenizer, mode, args, device):
    # Calling from_pretrained here creates a fresh model for each condition.
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(UPOS), id2label=ID_TO_LABEL, label2id=LABEL_TO_ID
    )
    if mode == "partial_freezing":
        freeze_lower_half(model)
    model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate
    )
    train_loader = DataLoader(
        train_rows,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate(batch, tokenizer),
    )
    started = time.time()
    model.train()
    epoch_losses = []
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
            total_loss += output.loss.item()
        epoch_losses.append(total_loss / max(1, len(train_loader)))
        print(f"{mode} | epoch {epoch + 1}/{args.epochs} | loss {epoch_losses[-1]:.4f}")

    dev_loader = DataLoader(
        dev_rows,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate(batch, tokenizer),
    )
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in dev_loader:
            labels = batch["labels"]
            inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
            predictions = model(**inputs).logits.argmax(dim=-1).cpu()
            mask = labels.ne(-100)
            correct += predictions.eq(labels).logical_and(mask).sum().item()
            total += mask.sum().item()
    elapsed = time.time() - started
    result = {
        "condition": mode,
        "dev_word_accuracy": correct / total if total else None,
        "correct_words": correct,
        "dev_words": total,
        "epoch_mean_training_loss": epoch_losses,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "elapsed_seconds": round(elapsed, 1),
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-train", type=int, default=600)
    parser.add_argument("--max-dev", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results", type=Path, default=Path("results.json"))
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train = read_conllu(download_conllu("train", args.data_dir))
    dev = read_conllu(download_conllu("dev", args.data_dir))
    train = sample_sentences(train, args.max_train, args.seed)
    dev = sample_sentences(dev, args.max_dev, args.seed + 1)
    print(f"Sampled sentences: train={len(train)}, dev={len(dev)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    results = []
    for mode in ("full_finetuning", "partial_freezing"):
        # Reset RNG so both conditions use the same data order and comparable head init.
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        results.append(train_and_evaluate(train, dev, tokenizer, mode, args, device))

    payload = {
        "model": MODEL_NAME,
        "dataset": "UD_Chinese-GSDSimp",
        "seed": args.seed,
        "max_train_sentences": args.max_train,
        "max_dev_sentences": args.max_dev,
        "actual_train_sentences": len(train),
        "actual_dev_sentences": len(dev),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "frozen_components": ["embeddings", "transformer layers 0-2"],
        "device": str(device),
        "results": results,
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved measured results to {args.results}")


if __name__ == "__main__":
    main()
