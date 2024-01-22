"""
    Script per a testar el model de resum de text mitjançant ROUGE, BERTscore i les mètriques de classificació.
    Però lo mal és que en esta versió no se poden fer les comprovacions de classificacio perquè ja s'ha passat tot a text, no hi ha un token de font
"""
import datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
import evaluate
import numpy as np
import torch
import nltk
from tqdm import tqdm
from sys import stdout
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, recall_score, confusion_matrix
import json
from diffusers import StableDiffusionPipeline

def main(checkpoint= "NAS-bilingue", carregar_de_hugginface = False, carregar_de_cache = True, llengua = "ca"):
    nltk.download('punkt')

    #carreguem el tokenitzador i el model
    #checkpoint = "NAS" + llengua + "-finetuned-diego-4-amb-metriques-anonim/checkpoint-650000"
    if carregar_de_hugginface:
        checkpoint = "dtorber/" + checkpoint
    # Fonts que estàn presents en les particions d'entrenament de DACSA

    # Token de serparació entre les fonts i el text
    SEP_TOKEN = "<text>"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only = not carregar_de_hugginface, use_fast=True, use_safetensors=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    # model = StableDiffusionPipeline.from_single_file(checkpoint, use_safetensors=True)


    # Els nous tokens especials seran tant el SEP_TOKEN com les claus de les
    # fonts per a l'idioma en concret, pero en principi ja estan en el tokenizer que hem descarregat
    #tokenizer.add_tokens([SEP_TOKEN] + SOURCES["ca"], special_tokens=True)

    #carreguem els datasets per a testejar:
    test_i = json.load(open("jsons_bilingue/test_i.json","r"))
    test_ni = json.load(open("jsons_bilingue/test_ni.json","r"))

    jsons_test = {
        "test_i": test_i,
        "test_ni": test_ni
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)


    #creem un diccionari que anira guardant els batch de cadascun dels 4 subconjunts que volem avaluar
    #tenen la forma llengua de NOTICIA-RESUM de manera que ca-es vol dir que és una notícia en català però
    #resum en castellà
    metriques_rouge = {
        "ca-ca": evaluate.load("rouge"),
        "ca-es": evaluate.load("rouge"),
        "es-ca": evaluate.load("rouge"),
        "es-es": evaluate.load("rouge")
    }

    metriques_bert = {
        "ca-ca": evaluate.load("bertscore"),
        "ca-es": evaluate.load("bertscore"),
        "es-ca": evaluate.load("bertscore"),
        "es-es": evaluate.load("bertscore")
    }

    label_relleno = -100
    multiple = 8
    max_article_length = 512
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
    def evaluate_model(subset="test_i"):
        model.eval()

        resums_json = {
            "ca": json.load(open(f"./resums_generats_bilingue/resums_normals/resums_{subset}_original_ca.json", "r")),
            "es": json.load(open(f"./resums_generats_bilingue/resums_normals/resums_{subset}_original_es.json", "r")),
        }
        #afegim el id a cada resum
        for llengua in resums_json.keys():
            for k, _ in resums_json[llengua].items():
                resums_json[llengua][k]["id"] = k
            for batch in tqdm(DataLoader(list(resums_json[llengua].values()), batch_size=8, shuffle=False), desc="Avaluant el model"):
                predictions_ca = batch["summary_<lang:ca>"]
                predictions_es = batch["summary_<lang:es>"]
                referencies_ca = list()
                referencies_es = list()
                for id in batch["id"]:
                    referencies_ca.append(jsons_test[subset][id]["summary_ca"])
                    referencies_es.append(jsons_test[subset][id]["summary_es"])


                #i ara l'única cosa que cal fer és afegir les prediccions i les referències a les mètriques
                #serà primer totes les notícies en castellà i després totes en català i s'agafen tant les prediccions com referències en una i altra llengua
                metriques_rouge[f"{llengua}-ca"].add_batch(predictions = predictions_ca, references = referencies_ca)
                metriques_rouge[f"{llengua}-es"].add_batch(predictions = predictions_es, references = referencies_es)

                # metriques_bert[f"{llengua}-ca"].add_batch(predictions = predictions_ca, references = referencies_ca)
                # metriques_bert[f"{llengua}-es"].add_batch(predictions = predictions_es, references = referencies_es)
       
        resultats_rouge = dict()
        for key, val in metriques_rouge.items():
            try:
                resultats_rouge[key] = val.compute()
            except:
                print(f"{key} has an empty value")
        #si volem fer servir este script per a NASes haurem de canviar el lang

        # resultats_bert = dict()
        # for key, val in metriques_bert.items():
        #     try:
        #         llengua = key.split("-")[1]
        #         resultats_bert[key] = np.mean(np.array(val.compute(lang = llengua)))
        #     except:
        #         print(f"{key} has an empty value")

        return {
            "rouge": resultats_rouge,
            # "bertscore": resultats_bert
        }

    with open("resultats_bilingue/rouge/resultats_rouge_test_i.json", "w") as f:
        res_i = evaluate_model(subset= "test_i")
        json.dump(res_i, f)

    with open("resultats_bilingue/rouge/resultats_rouge_test_ni.json", "w") as f:
        res_ni = evaluate_model(subset="test_ni")
        json.dump(res_ni, f)

if __name__ == "__main__":
    checkpoint = "checkpoint-119328"
    main(checkpoint = checkpoint)