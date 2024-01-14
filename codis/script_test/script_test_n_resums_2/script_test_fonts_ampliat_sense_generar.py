"""
    Script per a testar el model de resum de text mitjançant ROUGE, BERTscore i les mètriques de classificació
    Forçant per a cadascuna de les notícies un token cada vegada per veure si això millora o canvia d'alguna manera els resultats 
    Però com se va observar que els resums eren idèntics (o variava una paraula pel puesto o cometes o coses així) se comprova si són diferents els resums
    sense forçar-li font i forçant-li-la.
    Una consideració és que s'ha hagut de comprovar si eren cometes, perquè segons el diari gastava un tipus de cometes o un altre
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
def main(checkpoint= "NASca-finetuned-de-zero-amb-metriques-anonim", eixida = "resultats_script_test_fonts_ampliat.out", carregar_de_hugginface = False, carregar_de_cache = True, llengua = "ca"):
    nltk.download('punkt')
    carregar_de_cache = True
    #SI CANVIEM LA LLENGUA CANVIAR ELS TOKENS FONTS QUE FALTEN ELS DE CASTELLÀ
    # llengua = "ca"
    tokens_fonts = list(range(60002, 60007 + 1)) if llengua == "ca" else list(range(60002, 60022 + 1))
    tokens_fonts_ni = list(range(60008, 60010 + 1)) if llengua == "ca" else None
    # carregar_de_hugginface = True

    #carreguem el tokenitzador i el model
    #checkpoint = "NAS" + llengua + "-finetuned-de-zero-amb-metriques-anonim"
    if carregar_de_hugginface:
        checkpoint = "dtorber/" + checkpoint
    else:
        checkpoint = "../" + checkpoint
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
    test_i = datasets.load_dataset("json", data_files="../../corpus_anonimitzat/" + llengua +"/test-i.json.gz")
    test_ni = datasets.load_dataset("json", data_files="../../corpus_anonimitzat/" + llengua +"/test-ni.json.gz")

    #reduim els tamanys dels datasets per fer simplement proves:
    test_i = test_i["train"]
    test_ni = test_ni["train"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    max_article_length = 512
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
    def evaluate_model(subset = "test_i"):
        resums_diferents = dict()
        conteig_resums_diferents = dict()
        noticies_amb_resum_diferent = 0
        for tkn_font in tokens_fonts:
            resums_diferents[tkn_font] = []
            conteig_resums_diferents[tkn_font] = 0
        resums_json = json.load(open(f"../resums_generats/n_resums/resums_{subset}_original.json", "r"))

        #afegim el id a cada resum
        for k, _ in resums_json.items():
            resums_json[k]["id"] = k

        #preparem un diccionari amb tots els jsons de les diferents fonts forçades
        jsons_font = dict()
        for token_font in tokens_fonts:
            jsons_font[token_font] = json.load(open(f"../resums_generats/n_resums/resums_{subset}_forçat_{token_font}.json", "r"))
       
        for batch in tqdm(DataLoader(list(resums_json.values()), batch_size=8, shuffle=False), desc="Avaluant el model"):
            predictions_original = batch["summary"]
            noticies_resum_diferent = set()
            for token_font in tokens_fonts:
                #i després el que farem és en cada moment consultar el json corresponent a la font forçada i traure el resum corresponent
                predictions = []
                for id in batch["id"]:
                    predictions.append(jsons_font[token_font][id]["summary"])
                for i, (pred, original) in enumerate(zip (predictions, predictions_original)):
                    for preds, original_car in zip(pred, original):
                        for pred_car in preds: #(ara tenim una llista amb vàrios resums, aleshores els recorrem tots)
                            #si es un dels tokens especials passem d'ell
                            if pred_car == original_car:
                                continue
                            elif (pred_car == '\"' or pred_car == '\'' or pred_car == '“' or pred_car == "”") and (original_car == '\"' or original_car == '\'' or original_car == '“' or original_car == "”"):
                                continue
                            else:
                                conteig_resums_diferents[token_font] += 1
                                resums_diferents[token_font].append((pred, original))
                                noticies_resum_diferent.add(i)
                                break
            break
            noticies_amb_resum_diferent += len(noticies_resum_diferent)
        return resums_diferents, conteig_resums_diferents, noticies_amb_resum_diferent


    f = open(eixida, "w")
    print("Resultats de comprovar si els resums generats són diferents si comencen amb un token de font diferent:")
    f.write("Resultats de comprovar si els resums generats són diferents si comencen amb un token de font diferent:\n")
    resums_diferents_i, contador_i, contador_noticies_i = evaluate_model("test_i")
    print("Resultats en test_i:")
    f.write("Resultats en test_i:\n")
    print(f"Noticies amb resums diferents: {contador_noticies_i} ({contador_noticies_i / len(test_i) * 100}%)")
    f.write(f"Noticies amb resums diferents: {contador_noticies_i} ({contador_noticies_i / len(test_i) * 100}%)\n")
    f.write("Raw matrix: \n")
    for font in tokens_fonts:
        print(f"Font {font}: {contador_i[font]} ({contador_i[font]}%)")
        f.write(f"Font {font}: {contador_i[font]} ({contador_i[font]}%)\n")
    for font in tokens_fonts:
        print(f"Font {font}: {contador_i[font]} ({contador_i[font]}%)")
        f.write(f"Font {font}: {contador_i[font]} ({contador_i[font]}%)\n")
        for resum_original, resums_referencia in resums_diferents_i[font]:
            print(f"\n\t{resum_original}\n\t{resums_referencia}\n")
            f.write(f"\n\t{resum_original}\n\t{resums_referencia}\n\n")
    
    f.write("Normalized matrix: \n")
    for font in tokens_fonts:
        print(f"Font {font}: {contador_i[font]} ({contador_i[font] / (len(test_i) * len(tokens_fonts) * 4) * 100}%)")
        f.write(f"Font {font}: {contador_i[font]} ({contador_i[font] / (len(test_i) * len(tokens_fonts) * 4) * 100}%)\n")
    for font in tokens_fonts:
        print(f"Font {font}: {contador_i[font]} ({contador_i[font] / (len(test_i) * len(tokens_fonts) * 4) * 100}%)")
        f.write(f"Font {font}: {contador_i[font]} ({contador_i[font] / (len(test_i) * len(tokens_fonts) * 4) * 100}%)\n")
        for resum_original, resums_referencia in resums_diferents_i[font]:
            print(f"\n\t{resum_original}\n\t{resums_referencia}\n")
            f.write(f"\n\t{resum_original}\n\t{resums_referencia}\n\n")
    f.close()
    f = open(eixida, "a")
    print("Resultats en test_ni:")
    f.write("Resultats en test_ni:\n")
    resums_diferents_ni, contador_ni, contador_noticies_ni = evaluate_model("test_ni")
    print(f"Noticies amb resums diferents: {contador_noticies_ni} ({contador_noticies_ni / (len(test_ni) * len(tokens_fonts_ni) * 4) * 100}%)")
    f.write(f"Noticies amb resums diferents: {contador_noticies_ni} ({contador_noticies_ni / len(test_ni) * 100}%)\n")
    f.write("Raw matrix: \n")
    for font in tokens_fonts:
        print(f"Font {font}: {contador_ni[font]} ({contador_ni[font]}%)")
        f.write(f"Font {font}: {contador_ni[font]} ({contador_ni[font]}%)\n")
    for font in tokens_fonts:
        print(f"Font {font}: {contador_ni[font]} ({contador_ni[font]}%)")
        f.write(f"Font {font}: {contador_ni[font]} ({contador_ni[font]}%)\n")
        for resum_original, resums_referencia in resums_diferents_ni[font]:
            print(f"\n\t{resum_original}\n\t{resums_referencia}\n")
            f.write(f"\n\t{resum_original}\n\t{resums_referencia}\n\n")
    
    f.write("Normalized matrix: \n")
    for font in tokens_fonts:
        print(f"Font {font}: {contador_ni[font]} ({contador_ni[font] / (len(test_ni) * len(tokens_fonts) * 4) * 100}%)")
        f.write(f"Font {font}: {contador_ni[font]} ({contador_ni[font] / (len(test_ni) * len(tokens_fonts) * 4) * 100}%)\n")
    for font in tokens_fonts:
        print(f"Font {font}: {contador_ni[font]} ({contador_ni[font] / (len(test_ni) * len(tokens_fonts) * 4) * 100}%)")
        f.write(f"Font {font}: {contador_ni[font]} ({contador_ni[font] / (len(test_ni) * len(tokens_fonts) * 4) * 100}%)\n")
        for resum_original, resums_referencia in resums_diferents_ni[font]:
            print(f"\n\t{resum_original}\n\t{resums_referencia}\n")
            f.write(f"\n\t{resum_original}\n\t{resums_referencia}\n\n")
    f.close()
   
if __name__ == "__main__":
    main()