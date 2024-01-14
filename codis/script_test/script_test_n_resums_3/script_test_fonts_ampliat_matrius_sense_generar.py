"""
    Script per a testar el model de resum de text mitjançant ROUGE, BERTscore i les mètriques de classificació
    Forçant per a cadascuna de les notícies un token cada vegada per veure si això millora o canvia d'alguna manera els resultats 
    Però com se va observar que els resums eren idèntics (o variava una paraula pel puesto o cometes o coses així) se comprova com de diferents són els resums
    sense forçar-li font i forçant-li-la, per això en este codi posarem els resultats en format de matriu "de confusió" i traurem també la matriun per veure com de 
    diferents son (ROUGE) i la p-abstractivitat
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

def main(checkpoint= "NASca-finetuned-de-zero-amb-metriques-anonim", eixida = "resultats_script_test_fonts_ampliat_matrius.out", carregar_de_hugginface = False, carregar_de_cache = True, llengua = "ca"):
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
    tokenizer.add_tokens(["CA07", "CA08","CA09"], special_tokens=True)
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

    
    max_article_length = 512

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)


    label_relleno = -100
    multiple = 8
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, 
                                            label_pad_token_id=label_relleno,
                                            padding = "max_length",
                                            max_length = max_article_length,
                                            pad_to_multiple_of=multiple,
                                        )
    

    resums = json.load(open(f"./resums_generats/n_resums/resums_test_i_original.json", "r"))
    #pero ara cal ordenar els jsons per font
    resums_per_font_i = dict()
    for tkn_font in tokens_fonts:
        resums_per_font_i[str(tkn_font)] = []
    
    for id, resum in resums.items():
        #cal que tokenitzem la font perquè està en format CA01 i volem 60002
        src = tokenizer.encode(resum["source"])[1] #quan fem l'encode me torna [1, font, 2] per tant me quede amb la posició 1 perquè lo altre són els tokens tipics del tokenizer
        #emmagatzeme per a cadascuna de les fonts originals una llista amb paerlls (resum generat, resum original)
        resums_per_font_i[str(src)].append(((id, resum["summary"])))
    
    resums = json.load(open(f"./resums_generats/n_resums/resums_test_ni_original.json", "r"))
    #pero ara cal ordenar els jsons per font
    resums_per_font_ni = dict()
    for tkn_font in tokens_fonts_ni:
        resums_per_font_ni[str(tkn_font)] = []
    
    for id, resum in resums.items():
        #cal que tokenitzem la font perquè està en format CA01 i volem 60002
        src = tokenizer.encode(resum["source"])[1] #quan fem l'encode me torna [1, font, 2] per tant me quede amb la posició 1 perquè lo altre són els tokens tipics del tokenizer
        #emmagatzeme per a cadascuna de les fonts originals una llista amb paerlls (resum generat, resum original)
        resums_per_font_ni[str(src)].append(((id, resum["summary"])))

    resums_per_font = {
        "test_i": resums_per_font_i,
        "test_ni": resums_per_font_ni,
    }

    """
        Funcio per avaluar els resultats del model: ROUGE, BERTscore i la tasca de classificació si escau
        @param model: model a avaluar
        @param test_dataset: dataset amb les dades de test
        @param avaluar_classif: si és True, avaluem també la classificació
        @return: diccionari amb els distints resultats de l'avaluació i la matriu de confusió si avaluar_classif era True
    """
    def evaluate_model(subset = "test_i"):
        #resums_diferents = dict()
        tokens = tokens_fonts if subset == "test_i" else tokens_fonts_ni
        conteig_resums_diferents = dict()
        noticies_amb_resum_diferent = set()
        matriu_confusio = dict()
        matriu_rouge = dict()     
        #com no dona temps d'executar-se tot cridem de columna en columna
        for tkn_font in tokens:
            #resums_diferents[str(tkn_font)] = []
            conteig_resums_diferents[str(tkn_font)] = 0
            matriu_confusio[str(tkn_font)] = dict()
            matriu_rouge[str(tkn_font)] = dict()
            for token_font in tokens_fonts:
                matriu_confusio[str(tkn_font)][str(token_font)] = 0
                matriu_rouge[str(tkn_font)][str(token_font)] = 0

        for token_font in tqdm(tokens_fonts, desc="Columnes:", position=0): #este bucle governa les columnes (font forçada)
            json_font = json.load(open(f"../resums_generats/n_resums/resums_{subset}_forçat_{token_font}.json", "r"))
            for tkn_font in tqdm(tokens, desc="Files:", position=1): #este bucle governa les files (font original)
                if len(resums_per_font[subset][str(tkn_font)]) == 0: continue
                metric = evaluate.load("rouge") #resetegem cada vegada la mètrica perquè no influisca en unes i altres
                n_batch = 0
                for batch in tqdm(DataLoader(resums_per_font[subset][str(tkn_font)], batch_size=8, shuffle=False), desc="Batch:", position = 2):
                    predictions_original = []
                    predictions = []
                    for i in range(len(batch[0])): #batch té dos components, que són tuples de ids i tuples de resums
                        id = batch[0][i]
                        #ara lo que passa que hi ha 4 components i en cadascuna hi ha una tupla amb un resum per cada id
                        for j in range(len(batch[1])): #per a cadascuna de les tuples m'agarre el resum que correspon a la id (i-essim)
                            predictions_original += [batch[1][j][i]]
                        predictions += json_font[id]["summary"] #i necessitem tindre 4 vegades el mateix resum per tal que se puga fer la metric.compute
                   
                    noticies_resum_diferent = set()
                    
                    for i, (pred, original) in enumerate(zip (predictions, predictions_original)):
                        for pred_car, original_car in zip(pred, original):
                            #si es un dels tokens especials passem d'ell
                            if pred_car == original_car:
                                continue
                            elif (pred_car == '\"' or pred_car == '\'' or pred_car == '“' or pred_car == "”") and (original_car == '\"' or original_car == '\'' or original_car == '“' or original_car == "”"):
                                continue
                            else:
                                conteig_resums_diferents[str(tkn_font)] += 1
                                #resums_diferents[str(token_font)].append((pred, original))
                                matriu_confusio[str(tkn_font)][str(token_font)] += 1
                                noticies_resum_diferent.add(i)
                                break
                    n_batch += len(batch)
                    metric.add_batch(predictions=predictions, references=predictions_original)
                    break
                resultat = metric.compute()
                matriu_rouge[str(tkn_font)][str(token_font)] = resultat["rougeLsum"]

                
            """ 
                Versió sense tindre en compte lo de les cometes (serà més ràpida perquè compara tokens, no caràcters)
                for i, (pred, original) in enumerate(zip (predictions, predictions_original)):
                        for pred_car, original_car in zip(pred, original):
                            #si es un dels tokens especials passem d'ell
                            if pred_car == original_car:
                                continue
                            elif (pred_car == '\"' or pred_car == '\'' or pred_car == '“' or pred_car == "”") and (original_car == '\"' or original_car == '\'' or original_car == '“' or original_car == "”"):
                                continue
                            else:
                                conteig_resums_diferents[str(token_font)] += 1
                                resums_diferents[str(token_font)].append((pred, original))
                                matriu_confusio[str(tkn_font)][str(token_font)] += 1
                                noticies_resum_diferent.add(i)
                                break """
        noticies_amb_resum_diferent = len(noticies_amb_resum_diferent)
        for tkn_font in tokens:
            for token_font in tokens_fonts:
                if len(resums_per_font[subset][str(tkn_font)]):
                    matriu_confusio[str(tkn_font)][str(token_font)] = matriu_confusio[str(tkn_font)][str(token_font)]  / len(resums_per_font[subset][str(tkn_font)])
        
        return conteig_resums_diferents, noticies_amb_resum_diferent, matriu_confusio, matriu_rouge

    f = open(eixida, "w")
    print("Resultats de comprovar si els resums generats són diferents si comencen amb un token de font diferent:")
    print("Amb les corresponents matrius \"de confusio\" i ROUGE:")
    f.write("Resultats de comprovar si els resums generats són diferents si comencen amb un token de font diferent:\n")
    f.write("Amb les corresponents matrius \"de confusio\" i ROUGE:\n")
    contador_i, contador_noticies_i, matriu_confusio_i, matriu_rouge_i = evaluate_model("test_i")
    #per si de cas peta
    f.write("Conteig: " + str(contador_i) + "\n")
    f.write("Matriu de confusio: " + str(matriu_confusio_i) + "\n")
    f.write("Matriu ROUGE: " + str(matriu_rouge_i) + "\n")
    f.write("Contador noticies: " + str(contador_noticies_i) + "\n")
    
    print("Resultats en test_i (les notícies no estan contades més d'una vegada):")
    f.write("Resultats en test_i (les notícies no estan contades més d'una vegada):\n")
    print(f"Noticies amb resums diferents: {contador_noticies_i} ({round(contador_noticies_i / len(test_i) * 100, 5)}%)")
    f.write(f"Noticies amb resums diferents: {contador_noticies_i} ({round(contador_noticies_i / len(test_i) * 100, 5)}%)\n")
    print("Fonts amb resums diferents (el conteig ací i en la matriu se fa sobre la font original): ")
    f.write("Fonts amb resums diferents (el conteig ací i en la matriu se fa sobre la font original): \n")
    for font in contador_i.keys():
        print(f"Font {font}: {contador_i[str(font)]} ({round(contador_i[str(font)] / (len(resums_per_font['test_i'][str(font)]) * len(tokens_fonts)) * 100, 5)}%)")
        f.write(f"Font {font}: {contador_i[str(font)]} ({round(contador_i[str(font)] / (len(resums_per_font['test_i'][str(font)]) * len(tokens_fonts)) * 100, 5)}%)\n")
    
    print("Matriu de confusio test_i: ")
    f.write("Matriu de confusio test_i: \n")
    escriure_matriu(matriu_confusio_i, f)
    escriure_matriu(matriu_confusio_i, stdout)

    print("Matriu ROUGE L-sum test_i: ")
    f.write("Matriu ROUGE L-sum test_i: \n")
    escriure_matriu(matriu_rouge_i, f)
    escriure_matriu(matriu_rouge_i, stdout)
    f.close()
    

    f = open(eixida, "a")
    print("Resultats en test_ni:")
    f.write("Resultats en test_ni:\n")
    contador_ni, contador_noticies_ni, matriu_confusio_ni, matriu_rouge_ni = evaluate_model("test_ni")
    print(f"Noticies amb resums diferents: {contador_noticies_ni} ({round(contador_noticies_ni / len(test_ni) * 100, 5)}%)")
    f.write(f"Noticies amb resums diferents: {contador_noticies_ni} ({round(contador_noticies_ni / len(test_ni) * 100, 5)}%)\n")
    for font in contador_ni.keys():
        print(f"Font {font}: {contador_ni[str(font)]} ({round (contador_ni[str(font)] / (len(resums_per_font['test_ni'][str(font)]) * len(tokens_fonts)) * 100, 5)}%)")
        f.write(f"Font {font}: {contador_ni[str(font)]} ({round(contador_ni[str(font)] / (len(resums_per_font['test_ni'][str(font)]) * len(tokens_fonts)) * 100, 5)}%)\n")
    
    print("Matriu de confusio test_ni: ")
    f.write("Matriu de confusio test_ni: \n")
    escriure_matriu(matriu_confusio_ni, f)
    escriure_matriu(matriu_confusio_ni, stdout)

    print("Matriu ROUGE L-sum test_ni: ")
    f.write("Matriu ROUGE L-sum test_ni: \n")
    escriure_matriu(matriu_rouge_ni, f)
    escriure_matriu(matriu_rouge_ni, stdout)

    """
    Com ocupa tant d'espai de moment no ho estem carregant en cap fitxer
    #deixem per al final del tot escriure 
    print("Exemples diferents en test_i:")
    f.write("Exemples diferents en test_i:\n")
    for font in contador_i.keys():
        print(f"Font {font}: {contador_i[str(font)]} ({contador_i[str(font)] / (len(resums_per_font["test_i"][font]) * len(tokens_fonts)) * 100}%)")
        f.write(f"Font {font}: {contador_i[str(font)]} ({contador_i[str(font)] / (len(resums_per_font["test_i"][font]) * len(tokens_fonts)) * 100}%)\n")
        for resum_original, resums_referencia in resums_diferents_i[str(font)]:
            print(f"\n\t{resum_original}\n\t{resums_referencia}\n")
            f.write(f"\n\t{resum_original}\n\t{resums_referencia}\n\n")
    print("-" * 50 + "\n" * 3)

    print("Exemples diferents en test_ni:")
    f.write("Exemples diferents en test_ni:\n")
    for font in contador_ni.keys():
        print(f"Font {font}: {contador_ni[str(font)]} ({contador_ni[str(font)] / (len(resums_per_font["test_ni"][str(font)]) * len(tokens_fonts)) * 100}%)")
        f.write(f"Font {font}: {contador_ni[str(font)]} ({contador_ni[str(font)] / (len(resums_per_font["test_ni"][str(font)]) * len(tokens_fonts)) * 100}%)\n")
        for resum_original, resums_referencia in resums_diferents_ni[str(font)]:
            print(f"\n\t{resum_original}\n\t{resums_referencia}\n")
            f.write(f"\n\t{resum_original}\n\t{resums_referencia}\n\n")"""
    f.close()
   
if __name__ == "__main__":
    main()