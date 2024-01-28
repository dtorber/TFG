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
import ctranslate2
import pyonmttok

tokenizers={
    "es": pyonmttok.Tokenizer(mode="none", sp_model_path = "../models_softcatala/es-ca/tokenizer/sp_m.model"),
    "ca": pyonmttok.Tokenizer(mode="none", sp_model_path = "../models_softcatala/ca-es/tokenizer/sp_m.model"),
}

cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

def translate_text (text, from_lang, to_lang):
    text_tokenized = list()
    longitud_maxima = 0
    for t in text:
        tokenized = tokenizers[from_lang].tokenize(t)[0]
        longitud_maxima = max(longitud_maxima, len(tokenized))
        text_tokenized.append(tokenized)

    translator = ctranslate2.Translator(f"../models_softcatala/{from_lang}-{to_lang}/ctranslate2", max_queued_batches = -1)
    translated = translator.translate_batch(text_tokenized, max_input_length = longitud_maxima, max_decoding_length = int(1.1 * longitud_maxima), max_batch_size=32)
    translated = [tokenizers[from_lang].detokenize(t[0]["tokens"]) for t in translated]

    return translated

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
validacio_ca = datasets.load_dataset("json", data_files="../corpus_anonimitzat/ca/validation.json.gz")
entrenament_ca = datasets.load_dataset("json", data_files="../corpus_anonimitzat/ca/train.json.gz")
test_i_ca = datasets.load_dataset("json", data_files="../corpus_anonimitzat/ca/test-i.json.gz")
test_ni_ca = datasets.load_dataset("json", data_files="../corpus_anonimitzat/ca/test-ni.json.gz")

#com s'agafa tot el corpus de català directament ho col·loquem tot i anirem afegim els exemples de espanyol
corpus_ca = {
    "test-i": test_i_ca["train"],
    "test-ni": test_ni_ca["train"],
    "train": entrenament_ca["train"],
    "validation": validacio_ca["train"],
}

conteig_ca = {
    "test-i": dict(),
    "test-ni": dict(),
    "train": dict(),
    "validation": dict(),
}

corpus_ca_es = {
    "test-i": [],
    "test-ni": [],
    "train": [],
    "validation": [],
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
    print(f"Traduint resums català a espanyol ({particio})...")
    traduccions_catala = translate_text(resums_catala, "ca", "es")
    print(f"Traduint de volta resums espanyol a català ({particio})...")
    traduccions_predites = translate_text(traduccions_catala, "es", "ca")
    for i in range(len(corpus_ca_es[particio])):
        corpus_ca_es[particio][i]["summary_aux_es"] = traduccions_catala[i]
        corpus_ca_es[particio][i]["summary_pred_ca"] = traduccions_predites[i]

#carreguem els datasets en espanyol
test_i_es = datasets.load_dataset("json", data_files="../corpus_anonimitzat/es/test-i.json.gz")
test_ni_es = datasets.load_dataset("json", data_files="../corpus_anonimitzat/es/test-ni.json.gz")
validacio_es = datasets.load_dataset("json", data_files="../corpus_anonimitzat/es/validation.json.gz")
entrenament_es = datasets.load_dataset("json", data_files="../corpus_anonimitzat/es/train.json.gz")

corpus_es_aux = {
    "test-i": test_i_es["train"].shuffle(seed=42),
    "test-ni": test_ni_es["train"].shuffle(seed=42),
    "train": entrenament_es["train"].shuffle(seed=42),
    "validation": validacio_es["train"].shuffle(seed=42),
}

conteig_es = {
    "test-i": dict(),
    "test-ni": dict(),
    "train": dict(),
    "validation": dict(),
}
corpus_es = {
    "test-i": list(),
    "test-ni": list(),
    "train": list(),
    "validation": list(),
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
    print(f"Traduint resums espanyol a català ({particio})...")
    traduccions_espanyol = translate_text(resums_espanyol, "es", "ca")
    print(f"Traduint de volta resums català a espanyol ({particio})...")
    traduccions_predites = translate_text(traduccions_espanyol, "ca", "es")
    for i in range(len(corpus_es[particio])):
        corpus_es[particio][i]["summary_aux_ca"] = traduccions_espanyol[i]
        corpus_es[particio][i]["summary_pred_es"] = traduccions_predites[i]

#ajuntem en el corpus final els exemples d'espanyol
for particio in corpus_es.keys():
    corpus_ca_es[particio] += corpus_es[particio]


for particio in corpus_ca_es.keys():
    with gzip.open(f"../corpus_proves_traduccio/prova_traduccio_softcatala/{particio}.json.gz", "wt") as f:
        json.dump(corpus_ca_es[particio], f)