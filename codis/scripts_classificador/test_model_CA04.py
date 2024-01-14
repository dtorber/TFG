import json
import gzip
import argparse

from typing import List, Tuple, Dict

import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score
)

MAX_LENGTH = 512
BATCH_SIZE = 20

def predict(
    model_path: str, samples: List[str], num_labels: int
) -> Tuple[List[int], List[float]]:
    print(f"Loading model: {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=num_labels
    ).to('cuda')

    print(f"Loading tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, model_max_length=MAX_LENGTH
    )

    predictions: List[int] = []
    probabilities: List[float] = []

    for i in range(0, len(samples), BATCH_SIZE):
        ids = tokenizer(
            samples[i:i+BATCH_SIZE], truncation=True, max_length=MAX_LENGTH,
            return_tensors="pt", padding=True
        ).to('cuda')

        outs = model(**ids)
        probs = outs.logits.softmax(dim=1)
        classes = probs.argmax(axis=1)

        preds = [class_id.item() for class_id in classes]
        prob_preds = [probs[i][pred].item() for i, pred in enumerate(preds)]

        predictions.extend(preds)
        probabilities.extend(prob_preds)

    del model
    del tokenizer
    torch.cuda.empty_cache()

    return predictions, probabilities


NUM_SOURCES = 9
EXCLUDED = {7, 8, 9}


def main(args) -> None:
    key_to_id = {
        f"CA{i+1:02d}": i \
        for i in range(NUM_SOURCES) if i+1 not in EXCLUDED \
            or args.with_excluded
    }
    id_to_key = {value:key for key, value in key_to_id.items()}

    texts: List[str] = []
    references: List[int] = []
    noticies: List[Dict] = list()

    with open(args.samples_path, 'rb') as f:
        ifile = gzip.GzipFile(fileobj=f)
        for line in ifile:
            sample = json.loads(line.decode("utf-8").strip())
            source = sample[args.label_key]
            if source == "CA04":
                texts.append(sample[args.text_key])
                noticies.append(sample)

                if not isinstance(source, int):
                    references.append(key_to_id[source])
                else:
                    references.append(source)
                break

    predictions, _ = predict(args.model_path, texts, NUM_SOURCES - len(EXCLUDED))

    with open("resultats_classificador_CA04/noticies_CA03.txt", "w") as f:
        print(predictions)
        for pred in predictions:
            print(id_to_key[pred])

        with open("resultats_classificador_CA04/eixida_raw.out", "w") as f2:
            f2.write(str(predictions))

        for i, pred in enumerate(predictions):
            if id_to_key[pred] == "CA03":
                print(noticies[i])
                print()
                f.write(str(noticies[i]))
                f.write("\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("model_path")
    parser.add_argument("samples_path")
    parser.add_argument("--text-key", default="summary")
    parser.add_argument("--label-key", default="source")
    parser.add_argument("--with-excluded", action="store_true")
    parser.add_argument("--output", default="./resultats_classificacio/ca-summary.out")

    args = parser.parse_args()

    main(args)

