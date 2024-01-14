"""
    Script per construir un fitxer JSON que indexe per ID totes les notícies de test_i i de test_ni
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
from sys import stdout

nltk.download('punkt')
carregar_de_cache = False
#SI CANVIEM LA LLENGUA CANVIAR ELS TOKENS FONTS QUE FALTEN ELS DE CASTELLÀ
llengua = "ca"
tokens_fonts = list(range(60002, 60007 + 1)) if llengua == "ca" else list(range(60002, 60022 + 1))
carregar_de_hugginface = True

#carreguem el tokenitzador i el model
checkpoint = "NAS" + llengua + "-finetuned-diego-4-amb-metriques-anonim"
if carregar_de_hugginface:
    checkpoint = "dtorber/" + checkpoint
# Fonts que estàn presents en les particions d'entrenament de DACSA
SOURCES = {
    "ca": [
        f"CA{i:02d}" for i in range(1, 10)
    ],
    "es": [
        f"ES{i:02d}" for i in range(1, 22)
    ]
}

# Token de serparació entre les fonts i el text
SEP_TOKEN = "<text>"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only = not carregar_de_hugginface, use_fast=True)
tokenizer.add_tokens(["CA07", "CA08", "CA09"], special_tokens=True) #afegim els tokens de les fonts desconegudes

# Els nous tokens especials seran tant el SEP_TOKEN com les claus de les
# fonts per a l'idioma en concret, pero en principi ja estan en el tokenizer que hem descarregat
#tokenizer.add_tokens([SEP_TOKEN] + SOURCES["ca"], special_tokens=True)

#carreguem els datasets per a testejar:
test_i = datasets.load_dataset("json", data_files="../corpus_anonimitzat/ca-es2-jsonl/test-i.json.gz")
test_ni = datasets.load_dataset("json", data_files="../corpus_anonimitzat/ca-es2-jsonl/test-ni.json.gz")

#reduim els tamanys dels datasets per fer simplement proves:
test_i = test_i["train"]
test_ni = test_ni["train"]

#recorrem cadascuna de les notícies del json i les guardem indexades per id en un diccionari
json_i = {}
for noticia in test_i:
    json_i[noticia["id"]] = noticia
#write a json into disk
#com volem que el json quede guardat en disc i no només en memòria, escrivim un json 
with open("./jsons_bilingue/test_i.json", "w") as f:
    json.dump(json_i, f)
    
#fem el mateix amb el test_ni
json_ni = {}
for noticia in test_ni:
    json_ni[noticia["id"]] = noticia
#write a json into disk
with open("./jsons_bilingue/test_ni.json", "w") as f:
    json.dump(json_ni, f)
