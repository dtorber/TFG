import json
import gzip
import argparse

from typing import List, Tuple

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

    with open(args.samples_path, 'rb') as ifile:
        for line in ifile:
            sample = json.loads(line.decode("utf-8").strip())
            
            for diccionari in sample.values():
                texts.append(diccionari[args.text_key])
                source = diccionari[args.label_key]

                if not isinstance(source, int):
                    references.append(key_to_id[source])

                else:
                    references.append(source)

    predictions, _ = predict(args.model_path, texts, NUM_SOURCES - len(EXCLUDED))

    f = open(args.output, "w")
    f.write("\nResults\n")
    f.write("Confusion Matrix:\n")
    
    header = "     | " + "".join(f"{source:>5}" for source in key_to_id)
    f.write(header + "\n")
    f.write("-"*len(header) + "\n")

    for source, line in zip(key_to_id, confusion_matrix(references, predictions)):
        f.write(f"{source:<4} | " + "".join(f"{val:>5}" for val in line))
        f.write("\n")

    f.write(f"\nAccuracy:   {round((np.array(references)==np.array(predictions)).mean()*100, 2):>6}\n")
    f.write(f"Precision:  {round(precision_score(references, predictions, average='macro', zero_division=0)*100, 2):>6}\n")
    f.write(f"Recall:     {round(recall_score(references, predictions, average='macro', zero_division=0)*100, 2):>6}\n")
    f.write(f"F1-Score:   {round(f1_score(references, predictions, average='macro', zero_division=0)*100, 2):>6}\n")


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

