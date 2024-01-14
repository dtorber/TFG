"""
    Este script s'encarregara de comparar amb ROUGE-Lsum els resums generats sense forçar la font amb els resums de referència
    Aprofitant que tenim resums generats en fitxers .json ja preparats
"""
import datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
import evaluate
import numpy as np
import torch
import nltk
from tqdm import tqdm
import os
from sys import stdout
import json

#en este codi el que anem a fer és provar si a l'hora de generar el model i dir-li tokens diferents realment genera un resum diferent o és idèntic
#i posarem els resultats en format de matriu "de confusió" i traurem també la matriun per veure com de diferents son (ROUGE) i la p-abstractivitat


def escriure_vector (vector, fitxer):
    if (vector != None):
        #escrivim en un fitxer
        fitxer.write("\t\t")
        for token, valor in vector.items():
            fitxer.write(f"{token}:\t{valor}\n")

def main(checkpoint= "NASca-finetuned-de-zero-amb-metriques-anonim", eixida = "resultats_script_test_fonts_ampliat.out", carregar_de_hugginface = False, carregar_de_cache = True, llengua = "ca"):
    nltk.download('punkt')
    carregar_de_cache = True
    #SI CANVIEM LA LLENGUA CANVIAR ELS TOKENS FONTS QUE FALTEN ELS DE CASTELLÀ
    # llengua = "ca"
    tokens_fonts = list(range(60002, 60007 + 1)) if llengua == "ca" else list(range(60002, 60022 + 1))
    tokens_fonts_ni = list(range(60008, 60010 + 1)) if llengua == "ca" else None
    #carregar_de_hugginface = False
    #carreguem el tokenitzador i el model
    #checkpoint = "NAS" + llengua + "-finetuned-diego-4-amb-metriques-anonim/checkpoint-650000"
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
    tokenizer.add_tokens(["CA07", "CA08","CA09"], special_tokens=True)

    #carreguem els jsons que tenen indexat per id tota la informació de cada notícia
    test_i = json.load(open("./jsons/test_i.json", "r"))
    test_ni = json.load(open("./jsons/test_ni.json", "r"))

    noticies_originals = {
        "test_i" : test_i,
        "test_ni" : test_ni
    }

    max_article_length = 512
    device = "cuda" if torch.cuda.is_available() else "cpu"

    label_relleno = -100
    multiple = 8
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, 
                                            label_pad_token_id=label_relleno,
                                            padding = "max_length",
                                            max_length = max_article_length,
                                            pad_to_multiple_of=multiple,
                                        )
    """
        Funcio per avaluar els resultats del model: ROUGE 
        @param subset: subconjunt de dades a avaluar
        @return: diccionari amb la matriu de ROUGE entre la font forçada i el resum de referència
    """
    def evaluate_model(subset = "test_i"):
        tokens = tokens_fonts if subset == "test_i" else tokens_fonts_ni
        #ara no volem una matriu el que volem és un diccionari amb la mitjana de rouge per a cada font
        rouge = dict()
        
        resums = json.load(open(f"./resums_generats/resums_{subset}_original.json", "r"))
        resums_per_font = dict()
        for tkn_font in tokens:
            resums_per_font[str(tkn_font)] = []
        
        for id, resum in resums.items():
            #cal que tokenitzem la font perquè està en format CA01 i volem 60002
            src = tokenizer.encode(resum["source"])[1] #quan fem l'encode me torna [1, font, 2] per tant me quede amb la posició 1 perquè lo altre són els tokens tipics del tokenizer
            #emmagatzeme per a cadascuna de les fonts originals una llista amb paerlls (resum generat, resum original)
            entrada = {
                        "generat": resum["summary"],
                        "referencia": noticies_originals[subset][id]["summary"],
                      } 
            resums_per_font[str(src)].append((entrada))

        for tkn_font in tqdm(tokens, desc="Files:", position=1): #este bucle governa les files (font original)
            if len(resums_per_font[str(tkn_font)]) == 0: continue
            metric = evaluate.load("rouge") #resetegem cada vegada la mètrica perquè no influisca en unes i altres
            for batch in tqdm(DataLoader(resums_per_font[str(tkn_font)], batch_size=8, shuffle=False), desc="Batch:", position = 2):
                predictions_original = batch["referencia"]
                predictions = batch["generat"]
                
                metric.add_batch(predictions=predictions, references=predictions_original)
            resultat = metric.compute()
            rouge[str(tkn_font)] = resultat["rougeLsum"]

        return rouge

    matriu_i = evaluate_model(subset="test_i")
    with open("./resultats_rouge/rouge_per_font_i.txt", "w") as f:
        escriure_vector(matriu_i, f)
        escriure_vector(matriu_i, stdout)

    matriu_ni = evaluate_model(subset="test_ni")
    with open("./resultats_rouge/rouge_per_font_ni.txt", "w") as f:
        escriure_vector(matriu_ni, f)
        escriure_vector(matriu_ni, stdout)

    return {
        "test_i": matriu_i,
        "test_ni": matriu_ni
    }

if __name__ == "__main__":
    main()