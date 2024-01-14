"""
    Aquest script pretèn rebre uns resums ja traduits del castellà al català i del català al castellà, i l'objectiu és desfer la traducció (tornant a traduir)
    per tant ja tindriem el resum en l'idioma original, i l'anem a comparar amb l'original mitjançant BLEU per veure si els models traductors que fiquem pel mig
    estigueren afectant molt a la qualitat del resum amb que hem entrenat
    ESTA ES LA PART DE L'SCRIPT QUE COMPARA AMBDÓS JSONS
"""

import gzip
import json
import evaluate
import sys

PATH_CORPUS = "../corpus_proves_traduccio/prova_traduccio_boso"
if len(sys.argv) > 1:
    PATH_CORPUS = sys.argv[1]

particions = ["train", "validation", "test-i", "test-ni"] 
metrica = dict()
resultats = dict()
for particio in particions:
    metrica[particio] = {
        "ca": evaluate.load("bleu"),
        "es": evaluate.load("bleu")
    }
    resultats[particio] = dict()
    with gzip.open(f"{PATH_CORPUS}/{particio}.json.gz", "r") as f:
        corpus = json.load(f)
        print(corpus)
        input()
        for i in range(len(corpus)):
            print(corpus[i])
            input()
            lang = corpus[i]["lang"]
            metrica[particio][lang].add_batch(predictions = corpus[i][f"summary_pred_{lang}"], references = corpus[i][f"summary_ref_{lang}"])

    for lang in ["ca", "es"]:
        resultats[particio][lang] = metrica[particio][lang].compute()

print(metrica)
print(resultats)
with open("resultats_bilingue/bleu/metrica_bleu.json", "w") as f:
    json.dump(metrica, f)

with open("resultats_bilingue/bleu/resultats_bleu.json", "w") as f:
    json.dump(resultats, f)