"""
    Script per construir un fitxer JSON que sera el nou corpus bilingue ca-es
"""
import json
import datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
import evaluate
import numpy as np
import torch
import nltk
from tqdm import tqdm
import gzip
import requests

def translate_text (text, from_lang, to_lang):
    URL_SERVICE = "http://boso.dsic.upv.es:9009/translate"
    elements = [(i, text, from_lang, to_lang) for i, text in enumerate(text)]
    translated_elements = requests.post(
        URL_SERVICE, json=elements
    ).json()
    if not isinstance(translated_elements, list):
        raise Exception(
            f"Exception in the translation service: {translated_elements}"
        )
    res = []
    for (i, text, _, to_lang_, translation) in translated_elements:
        res.append(translation)
    return res

equivalencia = {
    "CA01": "ES01",
    "CA02": "ES02",
    "CA03": "ES03",
    "CA04": "ES04",
    "CA05": "ES05",
    "CA06": "ES06",
    "CA07": "ES11", #s'han d'agafar mostres per al test_ni que no s'hagen vist en el preentrenament
    "CA08": "ES15",
    "CA09": "ES16",
    "ES01": "CA01",
    "ES02": "CA02",
    "ES03": "CA03",
    "ES04": "CA04",
    "ES05": "CA05",
    "ES06": "CA06",
    "ES11": "CA07",
    "ES15": "CA08",
    "ES16": "CA09"
}

#carreguem els datasets en català
# validacio_ca = datasets.load_dataset("json", data_files="../corpus_anonimitzat/ca/validation.json.gz")
# entrenament_ca = datasets.load_dataset("json", data_files="../corpus_anonimitzat/ca/train.json.gz")
test_i_ca = datasets.load_dataset("json", data_files="../corpus_anonimitzat/ca/test-i.json.gz")
test_ni_ca = datasets.load_dataset("json", data_files="../corpus_anonimitzat/ca/test-ni.json.gz")

#com s'agafa tot el corpus de català directament ho col·loquem tot i anirem afegim els exemples de espanyol
corpus_ca = {
    # "train": entrenament_ca["train"],
    # "validation": validacio_ca["train"],
    "test-i": test_i_ca["train"],
    "test-ni": test_ni_ca["train"]
}

conteig_ca = {
    # "train": dict(),
    # "validation": dict(),
    "test-i": dict(),
    "test-ni": dict(),
}

corpus_ca_es = {
    # "train": [],
    # "validation": [],
    "test-i": [],
    "test-ni": []
}

#afegim el camp lang a tots els exemples de català i marquem el summary com a summary en català

for particio in corpus_ca.keys():
    resums_catala = []
    for i in tqdm(range(len(corpus_ca[particio])), desc=f"Preparant resums català {particio}"):
        entrada = dict()
        entrada["lang"] = "ca"
        entrada["summary_ref_ca"] = corpus_ca[particio][i]["summary"]
        entrada["id"] = corpus_ca[particio][i]["id"]
        corpus_ca_es[particio].append(entrada)

        conteig_ca[particio][corpus_ca[particio][i]["source"]] = conteig_ca[particio].get(corpus_ca[particio][i]["source"], 0) + 1
        resums_catala.append(entrada["summary_ref_ca"])
    print("Traduint resums català a espanyol...")
    traduccions_catala = translate_text(resums_catala, "ca", "es")
    print("Traduint de volta resums espanyol a català...")
    traduccions_predites = translate_text(traduccions_catala, "es", "ca")
    for i in range(len(corpus_ca_es[particio])):
        corpus_ca_es[particio][i]["summary_aux_es"] = traduccions_catala[i]
        corpus_ca_es[particio][i]["summary_pred_ca"] = traduccions_predites[i]

#carreguem els datasets en espanyol
# validacio_es = datasets.load_dataset("json", data_files="../corpus_anonimitzat/es/validation.json.gz")
# entrenament_es = datasets.load_dataset("json", data_files="../corpus_anonimitzat/es/train.json.gz")
test_i_es = datasets.load_dataset("json", data_files="../corpus_anonimitzat/es/test-i.json.gz")
test_ni_es = datasets.load_dataset("json", data_files="../corpus_anonimitzat/es/test-ni.json.gz")

corpus_es_aux = {
    # "train": entrenament_es["train"].shuffle(seed=42),
    # "validation": validacio_es["train"].shuffle(seed=42),
    "test-i": test_i_es["train"].shuffle(seed=42),
    "test-ni": test_ni_es["train"].shuffle(seed=42)
}

conteig_es = {
    # "train": dict(),
    # "validation": dict(),
    "test-i": dict(),
    "test-ni": dict(),
}
corpus_es = {
    # "train": [],
    # "validation": [],
    "test-i": [],
    "test-ni": [],
}

for particio in corpus_es_aux.keys():
    resums_espanyol = []
    for i in tqdm(range(len(corpus_es_aux[particio])), desc=f"Preparant resums espanyol {particio}"):
        source = corpus_es_aux[particio][i]["source"]
        #nomes l'afegim al corpus si es de la font que ens interessa i si hi ha encara no hem arribat al limit d'exemples que volem, que son tants com en castellà del seu equivalent
        if source in equivalencia and conteig_es[particio].get(source, 0) < conteig_ca[particio].get(equivalencia[source], 0):
            id = corpus_es_aux[particio][i]["id"]
            summary = corpus_es_aux[particio][i]["summary"]
            entrada = dict()
            entrada["summary_ref_es"] = summary
            entrada["id"] = id
            entrada["lang"] = "es"
            corpus_es[particio].append(entrada)

            conteig_es[particio][source] = conteig_es[particio].get(source, 0) + 1 
            resums_espanyol.append(summary)
    print("Traduint resums espanyol a català...")
    traduccions_espanyol = translate_text(resums_espanyol, "es", "ca")
    print("Traduint de volta resums català a espanyol...")
    traduccions_predites = translate_text(traduccions_espanyol, "ca", "es")
    for i in range(len(corpus_es[particio])):
        corpus_es[particio][i]["summary_aux_ca"] = traduccions_espanyol[i]
        corpus_es[particio][i]["summary_pred_es"] = traduccions_predites[i]

#ajuntem en el corpus final els exemples d'espanyol
for particio in corpus_es.keys():
    corpus_ca_es[particio] += corpus_es[particio]


for particio in corpus_ca_es.keys():
    with gzip.open(f"../corpus_proves_traduccio/prova_traduccio_boso/{particio}.json.gz", "wt") as f:
        json.dump(corpus_ca_es[particio], f)