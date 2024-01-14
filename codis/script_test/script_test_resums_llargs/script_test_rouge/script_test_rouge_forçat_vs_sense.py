"""
    Este script va a fer un poc el mateix que script_test_fonts_ampliat però ara ho comprova sobre els resums ja generats
    i a banda d'això va a anar resum a resum, mirant els que siguen diferents quan es força la font i quan no
    i guardarem tots aquells casos en que el ROUGE siga != 1, aprofitarem i ho ordenarem de menor a major ROUGE
    de cada notícia diferent guardarem:
    -Identificador notícia
    -ROUGE L-sum
    -Font original
    -Resum original
    -Font forçada
    -Resum forçat
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

"""
    Funcio auxiliar per escriure la matriu de confusio en un fitxer
    @param matriu: matriu de confusio (com un diccionari)
    @param fitxer: fitxer on escriure la matriu
"""
def escriure_matriu (matriu, fitxer):
    if (matriu != None):
        #escrivim en un fitxer
        fitxer.write("\t\t")
        for token in matriu[list(matriu.keys())[0]].keys():
            fitxer.write("\t" + str(token) + "\t")
        fitxer.write("\n")
        for token1 in matriu.keys():
            fitxer.write(str(token1) + "\t")
            for token2 in matriu[token1].keys():
                fitxer.write("\t" + str(round(matriu[token1].get(token2, 0), 5)) + "\t")
            fitxer.write("\n")

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
        resums_originals = json.load(open(f"./resums_generats/resums_{subset}_original.json", "r"))
        resums_diferents = dict()
        jsons_forçats = dict()
        for tkn_font in tokens:
            resums_diferents[str(tkn_font)] = dict()
            for token_font in tokens_fonts:
                #organitzem una matriu on les files són els tokens de les fonts originals i les columnes les forçades
                #i ahi emmagatzarem el ROUGE i ID de cada notícia que se resumisca diferent
                resums_diferents[str(tkn_font)][str(token_font)] = list()
        for token_font in tqdm(tokens_fonts, desc="Columnes:", position=0): #este bucle governa les columnes (font forçada)
            #carreguem el json amb els resums generats per a la font forçada
            resums_forçats = json.load(open(f"./resums_generats/resums_{subset}_forçat_{token_font}.json", "r"))
            jsons_forçats[str(token_font)] = resums_forçats
            resums_forçats_per_font = dict()
            for tkn_font in tokens:
                resums_forçats_per_font[str(tkn_font)] = list()
            for id, resum in resums_forçats.items():
                #cal que tokenitzem la font perquè està en format CA01 i volem 60002
                src = tokenizer.encode(resum["source"])[1] #quan fem l'encode me torna [1, font, 2] per tant me quede amb la posició 1 perquè lo altre són els tokens tipics del tokenizer
                #emmagatzeme per a cadascuna de les fonts originals una llista amb paerlls (resum generat, resum original)
                entrada = {
                            "generat": resum["summary"],
                            "referencia": resums_originals[id]["summary"],
                            "id": id
                          } 
                resums_forçats_per_font[str(src)].append((entrada)) 

            for tkn_font in tqdm(tokens, desc="Files:", position=1): #este bucle governa les files (font original)
                if len(resums_forçats_per_font[str(tkn_font)]) == 0: continue
                for batch in tqdm(DataLoader(resums_forçats_per_font[str(tkn_font)], batch_size = 8), desc="Batch:", position = 2):
                    for generat, referencia, id in zip(batch["generat"], batch["referencia"], batch["id"]):
                        metric = evaluate.load("rouge") #resetegem cada vegada la mètrica perquè no influisca en unes i altres
                        #fa falta que ho fiquem com si fora una llista perquè és lo que està esperant
                        resultat = metric.compute(predictions=[generat], references=[referencia])
                        if resultat["rougeLsum"] != 1:
                            entrada = {
                                "id": id, #guardem l'identificador de la notícia i així podem accedir a tot després
                                "rougeLsum": resultat["rougeLsum"],
                                "font_forçada": token_font,
                            }
                            resums_diferents[str(tkn_font)][str(token_font)].append(entrada)
        #check if the directory resultats_rouge/per_font_original exists and create it if not
        if not os.path.exists(f"./resultats_rouge/per_font_original"):
            os.makedirs(f"./resultats_rouge/per_font_original")
        if not os.path.exists(f"./resultats_rouge/per_font_forçada"):
            os.makedirs(f"./resultats_rouge/per_font_forçada")
        
        for tkn_font in tokens:
            noticies_diferents = list()
            for token_font in tokens_fonts:
                noticies_diferents += (resums_diferents[str(tkn_font)][str(token_font)])
            noticies_diferents.sort(key=lambda x: x["rougeLsum"])
            with open(f"./resultats_rouge/per_font_original/resums_diferents_{subset}_original_{str(tkn_font)}.txt", "w") as fitxer:
                fitxer.write(f"Ací tenim tots els resums de les noticies de la font {tkn_font} que s'han resumit diferent en forçar la font i sense dir res. Ordenats de menor a major ROUGE (major a menor diferència)\n")
                if len(resums_forçats_per_font[str(tkn_font)]) * len(tokens_fonts) != 0:
                    fitxer.write(f"Total casos diferents: {len(noticies_diferents)} / {len(resums_forçats_per_font[str(tkn_font)]) * len(tokens_fonts)} ({round(len(noticies_diferents) / (len(resums_forçats_per_font[str(tkn_font)]) * len(tokens_fonts)), 4)}%)\n\n")
                for noticia in noticies_diferents:
                    fitxer.write(f"\t\tID: {noticia['id']}\n\t\tROUGE: {noticia['rougeLsum']}\n\t\t")
                    fitxer.write(f"Font original: {resums_originals[noticia['id']]['source']}\n\t\tFont forçada: {tokenizer.decode(noticia['font_forçada'])}\n\t\t")
                    fitxer.write(f"Resum original: {resums_originals[noticia['id']]['summary']}\n\t\tResum forçat: {jsons_forçats[str(noticia['font_forçada'])][noticia['id']]['summary']}\n\n\n")
        for token_font in tokens_fonts:
            noticies_diferents = list()
            for tkn_font in tokens:
                noticies_diferents += (resums_diferents[str(tkn_font)][str(token_font)])
            noticies_diferents.sort(key=lambda x: x["rougeLsum"])
            with open(f"./resultats_rouge/per_font_forçada/resums_diferents_{subset}_forçat_{str(token_font)}.txt", "w") as fitxer:
                if (len(resums_forçats_per_font[str(tkn_font)]) * len(tokens_fonts)) != 0:
                    fitxer.write(f"Total casos diferents: {len(noticies_diferents)} / {len(resums_forçats_per_font[str(tkn_font)]) * len(tokens_fonts)} ({round(len(noticies_diferents) / (len(resums_forçats_per_font[str(tkn_font)]) * len(tokens_fonts)), 4)}%)\n\n")
                fitxer.write(f"Ací tenim tots els resums que s'han resumit diferent en forçar la font {token_font} i sense dir res. Ordenats de menor a major ROUGE (major a menor diferència)\n\n")
                for noticia in noticies_diferents:
                    fitxer.write(f"\t\tID: {noticia['id']}\n\t\tROUGE: {noticia['rougeLsum']}\n\t\t")
                    fitxer.write(f"Font original: {resums_originals[noticia['id']]['source']}\n\t\tFont forçada: {tokenizer.decode(noticia['font_forçada'])}\n\t\t")
                    fitxer.write(f"Resum original: {resums_originals[noticia['id']]['summary']}\n\t\tResum forçat: {jsons_forçats[str(noticia['font_forçada'])][noticia['id']]['summary']}\n\n\n") 

    evaluate_model(subset="test_i")
    evaluate_model(subset="test_ni")

if __name__ == "__main__":
    main()