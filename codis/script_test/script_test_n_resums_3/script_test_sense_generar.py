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

def main(checkpoint= "NASca-finetuned-de-zero-amb-metriques-anonim", eixida = "resultats_script_test.out", carregar_de_hugginface = False, carregar_de_cache = True, llengua = "ca"):
    nltk.download('punkt')
    #carregar_de_cache = True
    #SI CANVIEM LA LLENGUA CANVIAR ELS TOKENS FONTS QUE FALTEN ELS DE CASTELLÀ
    # llengua = "ca"
    tokens_fonts = list(range(60002, 60007 + 1)) if llengua == "ca" else list(range(60002, 60022 + 1))
    tokens_fonts_ni = list(range(60008, 60010 + 1)) if llengua == "ca" else None
    #carregar_de_hugginface = True

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
    tokenizer.add_tokens(["CA07", "CA08", "CA09"], special_tokens=True) #afegim els tokens de les fonts desconegudes
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    # Els nous tokens especials seran tant el SEP_TOKEN com les claus de les
    # fonts per a l'idioma en concret, pero en principi ja estan en el tokenizer que hem descarregat
    #tokenizer.add_tokens([SEP_TOKEN] + SOURCES["ca"], special_tokens=True)

    #carreguem els datasets per a testejar:
    test_i = json.load(open("../jsons/test_i.json","r"))
    test_ni = json.load(open("../jsons/test_ni.json","r"))

    jsons_test = {
        "test_i": test_i,
        "test_ni": test_ni
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    metric = evaluate.load("rouge")
    bert = evaluate.load("bertscore")

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

        resums_json = json.load(open(f"../resums_generats/n_resums/resums_{subset}_original.json", "r"))
        #creem un diccionari que per cada id guardarà una llista amb els resultats generats
        resultats = dict()
        #afegim el id a cada resum
        for k, _ in resums_json.items():
            resums_json[k]["id"] = k
        for batch in tqdm(DataLoader(list(resums_json.values()), batch_size=8, shuffle=False), desc="Avaluant el model"):
            for id, preds in zip(batch["id"], batch["summary"]):
                resultats[id] = []
                for pred in preds:
                    metric = evaluate.load("rouge")
                    resultats[id].append(metric.compute(predictions = [pred], references = [jsons_test[subset][id]["summary"]]))
            break
        #calcularem les mitjanes de rouge1, rouge2, rougeL i rougeLsum
        rouge1 = []
        rouge2 = []
        rougeL = []
        rougeLsum = []
        
        #calculem les mitjanes per al primer resum que és el que hem estat generant fins ara
        rouge1_primera = []
        rouge2_primera = []
        rougeL_primera = []
        rougeLsum_primera = []
        
        #calculem les mitjanes per als resums que no són el primer per veure si hi ha hagut alguna millora en generar més resums
        rouge1_restants = []
        rouge2_restants = []
        rougeL_restants = []
        rougeLsum_restants = []
        for id in resultats.keys():
            rouge1 += [r["rouge1"] for r in resultats[id]]
            rouge2 += [r["rouge2"] for r in resultats[id]]
            rougeL += [r["rougeL"] for r in resultats[id]]
            rougeLsum += [r["rougeLsum"] for r in resultats[id]]

            rouge1_primera.append(resultats[id][0]["rouge1"])
            rouge2_primera.append(resultats[id][0]["rouge2"])
            rougeL_primera.append(resultats[id][0]["rougeL"])
            rougeLsum_primera.append(resultats[id][0]["rougeLsum"])

            rouge1_restants += [r["rouge1"] for r in resultats[id][1:]]
            rouge2_restants += [r["rouge2"] for r in resultats[id][1:]]
            rougeL_restants += [r["rougeL"] for r in resultats[id][1:]]
            rougeLsum_restants += [r["rougeLsum"] for r in resultats[id][1:]]

        rouge1 = np.mean(np.array(rouge1))
        rouge2 = np.mean(np.array(rouge2))
        rougeL = np.mean(np.array(rougeL))
        rougeLsum = np.mean(np.array(rougeLsum))

        rouge1_primera = np.mean(np.array(rouge1_primera))
        rouge2_primera = np.mean(np.array(rouge2_primera))
        rougeL_primera = np.mean(np.array(rougeL_primera))
        rougeLsum_primera = np.mean(np.array(rougeLsum_primera))

        rouge1_restants = np.mean(np.array(rouge1_restants))
        rouge2_restants = np.mean(np.array(rouge2_restants))
        rougeL_restants = np.mean(np.array(rougeL_restants))
        rougeLsum_restants = np.mean(np.array(rougeLsum_restants))

        resultats["rouge1"] = rouge1
        resultats["rouge2"] = rouge2
        resultats["rougeL"] = rougeL
        resultats["rougeLsum"] = rougeLsum

        resultats["rouge1_primera"] = rouge1_primera
        resultats["rouge2_primera"] = rouge2_primera
        resultats["rougeL_primera"] = rougeL_primera
        resultats["rougeLsum_primera"] = rougeLsum_primera

        resultats["rouge1_restants"] = rouge1_restants
        resultats["rouge2_restants"] = rouge2_restants
        resultats["rougeL_restants"] = rougeL_restants
        resultats["rougeLsum_restants"] = rougeLsum_restants

        return resultats, None

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

    f = open(eixida, "w")
    # f.write("Test del model, sense forçar el símbol de font\n")

    res_i, matriu_i = evaluate_model(subset= "test_i")


    if llengua == "ca":
        #afegim els tokens de les nvoes fonts perquè s'actualitze abans de testejar ni, perquè ho necessitem per la matriu de confusió
        tokens_fonts += [60008, 60009, 60010]
    else:
        #si es el cas de castellà faltarà afegir-li els tokens que corresponen a les fonts de _ni
        tokens_fonts += []
    res_ni, matriu_ni = evaluate_model(subset="test_ni")
    
    res = {
        "test_i": res_i,
        "test_ni": res_ni
    }

    json.dump(res, f)


    #el tanquem i tornem a obrir per si de cas peta enmig d'una execucio, que se puga aprofitar l'altra
    f.close()
if __name__ == "__main__":
    main()