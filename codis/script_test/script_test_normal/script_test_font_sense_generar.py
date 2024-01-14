"""
    Script per a testar el model de resum de text mitjançant ROUGE, BERTscore i les mètriques de classificació
    Forçant per a cadascuna de les notícies un token cada vegada per veure si això millora o canvia d'alguna manera els resultats
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

def main(checkpoint="NASca-finetuned-de-zero-amb-metriques-anonim", carregar_de_hugginface=False, carregar_de_cache=True, llengua="ca"):
    nltk.download('punkt')
    carregar_de_cache = True
    #SI CANVIEM LA LLENGUA CANVIAR ELS TOKENS FONTS QUE FALTEN ELS DE CASTELLÀ
    # llengua = "ca"
    tokens_fonts = list(range(60002, 60010 + 1)) if llengua == "ca" else list(range(60002, 60022 + 1))
    #carregar_de_hugginface = True

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
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    # Els nous tokens especials seran tant el SEP_TOKEN com les claus de les
    # fonts per a l'idioma en concret, pero en principi ja estan en el tokenizer que hem descarregat
    #tokenizer.add_tokens([SEP_TOKEN] + SOURCES["ca"], special_tokens=True)

    #carreguem els datasets per a testejar:
    test_i = json.load(open("jsons/test_i.json","r"))
    test_ni = json.load(open("jsons/test_ni.json","r"))

    jsons_test = {
        "test_i": test_i,
        "test_ni": test_ni
    }

    max_article_length = 512
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    metric = evaluate.load("rouge")
    bert = evaluate.load("bertscore")

    label_relleno = -100
    multiple = 8
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, 
                                            label_pad_token_id=label_relleno,
                                            padding = "max_length",
                                            max_length = max_article_length,
                                            pad_to_multiple_of=multiple,
                                        )

    """
        Funcio per avaluar els resultats del model: ROUGE, BERTscore i la tasca de classificació si escau
        @param model: model a avaluar
        @param test_dataset: dataset amb les dades de test
        @param avaluar_classif: si és True, avaluem també la classificació
        @return: diccionari amb els distints resultats de l'avaluació i la matriu de confusió si avaluar_classif era True
    """
    def evaluate_model(subset = "test_i", tkn_font = None):
        model.eval()

        resums_json = json.load(open(f"./resums_generats/resums_{subset}_forçat_{tkn_font}.json", "r"))
        #afegim el id a cada resum
        for k, _ in resums_json.items():
            resums_json[k]["id"] = k

        for batch in tqdm(DataLoader(list(resums_json.values()), batch_size=8, shuffle=False), desc="Avaluant el model"):            
            predictions = batch["summary"]
            referencies = []
            for id in batch["id"]:
                referencies.append(jsons_test[subset][id]["summary"])

            metric.add_batch(predictions=predictions, references=referencies)

        resultat = metric.compute()
        #si volem fer servir este script per a NASes haurem de canviar el lang
        
        return resultat, None

    """
        funcio per escriure els resultats en un fitxer i per pantalla
        @param resultats: diccionari amb els resultats
        @param matriu: matriu de confusio (possible None)
        @param fitxer: fitxer on escriure els resultats
        @param nom_test: nom del test per a diferenciar els resultats
    """
    def escriure_resultats(resultats, matriu, fitxer, nom_test):
        #Escrivim en el fitxer rebut
        fitxer.write(nom_test + " resultats: \n")
        fitxer.write(str(resultats) + "\n") #Important fer el str()! perquè si no és un diccionari
        if matriu is not None: 
            fitxer.write("Matriu de confusio (" + nom_test + "): \n")
            fitxer.write("-" * 50 + "\n")
            escriure_matriu(matriu, fitxer)
            fitxer.write("-" * 50 + "\n")
        fitxer.write("\n" + "=" * 50 + "\n")

        #Escrivim per pantalla
        print(nom_test + " resultats: \n", end="")
        print(str(resultats) + "\n", end="")
        if matriu is not None:
            print("Matriu de confusio (" + nom_test + "): \n", end="")
            print("-" * 50 + "\n", end="")
            escriure_matriu(matriu, stdout)
            print("-" * 50 + "\n", end="")
        print("\n" + "=" * 50 + "\n", end="")

    """
        Funcio auxiliar per escriure la matriu de confusio en un fitxer
        @param matriu: matriu de confusio (com un diccionari)
        @param fitxer: fitxer on escriure la matriu
    """
    def escriure_matriu (matriu, fitxer):
        if (matriu != None):
            #escrivim en un fitxer
            fitxer.write("\t")
            for token in tokens_fonts:
                fitxer.write("\t" + str(token) + "\t")
            fitxer.write("\n")
            for token1 in tokens_fonts:
                fitxer.write(str(token1) + "\t")
                for token2 in tokens_fonts:
                    fitxer.write("\t" + str(matriu[token1].get(token2, 0)) + "\t")
                fitxer.write("\n")

    #per si de cas no existeix el directori on escriure els resultats el creem
    if not os.path.exists("resultats_fonts"):
        os.makedirs("resultats_fonts")
    
    tokens = list(range(60002, 60007 + 1)) 

    for tkn in tokens:
        f = open("../resultats_n_resums/resultats_fonts/resultats_script_test_ni_font_" + str(tkn) + ".out", "w")
        
        print("Test del model, forçant el símbol de font (totes les fonts)")
        f.write("Test del model, forçant el símbol de font (totes les fonts)\n")
        print("Font: " + str(tkn))
        f.write("Font: " + str(tkn) + "\n")

        #No té molt de sentit avaluar el NI en este cas, però ho avaluem de totes maneres per si de cas.
        res_ni, matriu_ni = evaluate_model(subset = "test_ni", tkn_font = tkn)
        escriure_resultats(res_ni, matriu_ni, f, "test_ni")
        f.close()
        
        f = open("./resultats_fonts/resultats_script_test_font_" + str(tkn) + ".out", "a")
        res_i, matriu_i = evaluate_model(subset = "test_ni", tkn_font = tkn)
        escriure_resultats(res_i, matriu_i, f, "test_i")
        f.close()

if __name__ == "__main__":
    main()