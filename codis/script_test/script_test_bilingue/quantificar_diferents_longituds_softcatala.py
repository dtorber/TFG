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
import os

PATH_CORPUS = "../corpus_proves_traduccio/prova_traduccio_softcatala"
if len(sys.argv) > 1:
    PATH_CORPUS = sys.argv[1]

particions = ["train", "validation", "test-i", "test-ni"] 
metrica = dict()
resultats = dict()
for particio in particions:
    if not os.path.exists(f"{PATH_CORPUS}/{particio}.json.gz"):
        continue
    print("Avaluant partició", particio)
    metrica[particio] = {
        "ca": list(),
        "es": list()
    }
    resultats[particio] = dict()
    with gzip.open(f"{PATH_CORPUS}/{particio}.json.gz", "r") as f:
        corpus = json.load(f)
        for i in range(len(corpus)):
            lang = corpus[i]["lang"]
            pred = corpus[i][f"summary_pred_{lang}"]
            ref = corpus[i][f"summary_ref_{lang}"]
            if len(pred) < 0.8 * len(ref):
                metrica[particio][lang].append({
                    "id": corpus[i]["id"],
                    "pred": pred,
                    "ref": ref
                })

print(metrica)

model_traduccio = "boso" if "boso" in PATH_CORPUS else "softcatala"


with open(f"resultats_bilingue/num_frases_incompletes_{model_traduccio}.json", "w") as f:
    json.dump(metrica, f)