import json
import gzip
import argparse

from typing import Iterator

from metrics import compute_aggregated_metrics, Metrics


def read_json_ds(filename: str) -> Iterator[dict]:
    if filename.endswith(".json.gz"):
        with gzip.open(filename, "rb") as ifile:
            for line in ifile:
                sample = json.loads(line.decode("utf-8").strip())
                yield sample

    elif filename.endswith(".json"):
        with open(filename, "r", encoding="utf-8") as ifile:
            for line in ifile:
                sample = json.loads(line.strip())
                yield sample


def main(args):
    if args.dataset_path is None:
        # The text and the summary are in different files
        assert args.texts_path is not None
        assert args.summaries_path is not None

        texts = [sample[args.text_key] for sample in read_json_ds(args.texts_path)]
        summaries = [sample[args.summary_key] for sample in read_json_ds(args.summaries_path)]

    else:
        texts = []
        summaries = []


        #com nosaltres tenim els resums concencrats en test_i.json i test_ni.json indexats per id
        #el que fem és recórrer cadascuna de les eixides i agafar el seu id i buscar l'article
        f = None
        if "test_i" in args.dataset_path:
            f = open("jsons_bilingue/test_i.json", "r")
        else:
            f = open("jsons_bilingue/test_ni.json", "r")
        
        original = json.load(f)

        for sample in read_json_ds(args.dataset_path):
            for id, exemple in sample.items():
                #com l'abstractivitat realment té sentit en una mateixa llengua, només avaluarem el resum en la llengua original
                summaries.append(exemple[f"summary_{original[id]['lang']}"])
                article = original[id]["article"]
                texts.append(article)

    result = compute_aggregated_metrics(
        summaries, texts=texts, metrics=[Metrics.EXTRACTIVE, Metrics.NNG],
        nng=[1,2,3,4], ps=[2,3,4]
    )

    if args.output_filename is None:
        print(json.dumps(result, indent=2))

    else:
        assert args.output_filename.endswith(".json")
        with open(args.output_filename, "w", encoding="utf-8") as ofile:
            json.dump(result, ofile, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--text-key", default="article")
    parser.add_argument("--summary-key", default="summary")
    parser.add_argument("--dataset-path", help="When samples contain text and summary")
    parser.add_argument("--texts-path", help="The samples that contain the texts")
    parser.add_argument("--summaries-path", help="The samples that contain the summaries")
    parser.add_argument("--output-filename", help="JSON file where the results will be stored")

    args = parser.parse_args()
    main(args)
