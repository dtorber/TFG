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

def main(checkpoint= "NASca-finetuned-de-zero-amb-metriques-anonim", eixida = "resultats_script_test_fonts_ampliat_amtrius.out", carregar_de_hugginface = False, carregar_de_cache = True, llengua = "ca"):
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
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    # Els nous tokens especials seran tant el SEP_TOKEN com les claus de les
    # fonts per a l'idioma en concret, pero en principi ja estan en el tokenizer que hem descarregat
    #tokenizer.add_tokens([SEP_TOKEN] + SOURCES["ca"], special_tokens=True)

    #carreguem els datasets per a testejar:
    test_i = datasets.load_dataset("json", data_files="../corpus_anonimitzat/" + llengua +"/test-i.json.gz")
    test_ni = datasets.load_dataset("json", data_files="../corpus_anonimitzat/" + llengua +"/test-ni.json.gz")

    #reduim els tamanys dels datasets per fer simplement proves:
    test_i = test_i["train"]
    test_ni = test_ni["train"]

    dataset = dict()	

    #creem un dataset que estiga organitzat per font i aixi podem utilitzar el batch, seguim el mateix format que tenia originalment el test_i
    for token in tokens_fonts:
        features = {}
        for key in test_i.features.keys():
            features[key] = []
        dataset[str(token)] = features

    #ara anem mostra a mostra afegint-les al diccionari que toque
    for mostra in test_i:
        token_font = tokenizer.convert_tokens_to_ids(mostra["source"])
        dataset[str(token_font)]["article"].append(mostra["article"])
        dataset[str(token_font)]["summary"].append(mostra["summary"])
        dataset[str(token_font)]["source"].append(mostra["source"])
        dataset[str(token_font)]["id"].append(mostra["id"])

    #una vegada ho tenim preparat en format diccionari cal passar-ho a DatasetDict per poder aplicar totes les operacion
    ds = datasets.DatasetDict({})
    for token in tokens_fonts:
        ds[str(token)] = datasets.Dataset.from_dict(dataset[str(token)])

    dataset_ni = dict()	

    #creem un dataset que estiga organitzat per font i aixi podem utilitzar el batch, seguim el mateix format que tenia originalment el test_i
    for token in tokens_fonts_ni:
        features = {}
        for key in test_ni.features.keys():
            features[key] = []
        dataset_ni[str(token)] = features

    #ara anem mostra a mostra afegint-les al diccionari que toque
    for mostra in test_ni:
        token_font = tokenizer.convert_tokens_to_ids(mostra["source"])
        dataset_ni[str(token_font)]["article"].append(mostra["article"])
        dataset_ni[str(token_font)]["summary"].append(mostra["summary"])
        dataset_ni[str(token_font)]["source"].append(mostra["source"])
        dataset_ni[str(token_font)]["id"].append(mostra["id"])

    #una vegada ho tenim preparat en format diccionari cal passar-ho a DatasetDict per poder aplicar totes les operacion
    ds_ni = datasets.DatasetDict({})
    for token in tokens_fonts_ni:
        ds_ni[str(token)] = datasets.Dataset.from_dict(dataset_ni[str(token)])
    
    max_article_length = 512
    def preproces_dades_test (examples):
        decoder_summary = [tokenizer.bos_token + resum for resum in examples["summary"]]
        model_inputs = tokenizer(examples["article"],
                                truncation=True,
                                padding="max_length",
                                max_length=max_article_length,
                                add_special_tokens=False,
                                pad_to_multiple_of=8,)
        labels = tokenizer(text_target=examples["summary"],
                            truncation=True, 
                            max_length=max_article_length,
                            add_special_tokens=False, 
                            padding="max_length",
                            pad_to_multiple_of=8,)  # perquè ho tokenitze com a target hem d'indicar-ho
        model_inputs["labels"] = labels["input_ids"]
        #ara falta comprovar que si en el resum (label) o text original (input_ids) no és un pad o un eos, el canviem per un eos (</s>) perquè era més llarg el text del que cabia en 512

        for i in range(len(model_inputs["input_ids"])):
            if model_inputs["input_ids"][i][-1] != tokenizer.pad_token_id and model_inputs["input_ids"][i][-1] != tokenizer.eos_token_id:
                model_inputs["input_ids"][i][-1] = tokenizer.eos_token_id
        for i in range(len(model_inputs["labels"])):
            if model_inputs["labels"][i][-1] != tokenizer.pad_token_id and model_inputs["labels"][i][-1] != tokenizer.eos_token_id:
                model_inputs["labels"][i][-1] = tokenizer.eos_token_id
            for j in range(len(model_inputs["labels"][i])):
                if model_inputs["labels"][i][j] == tokenizer.pad_token_id:
                    model_inputs["labels"][i][j] = -100
        #i creem tambe el decoder_input_ids que serà el mateix que tinguera input_ids però afegitn <s> al principi
        #model_inputs["decoder_input_ids"] = [tokenizer.bos_token_id] + model_inputs["input_ids"][:]
        model_inputs["decoder_input_ids"] = tokenizer(decoder_summary, 
                                                    truncation=True, 
                                                    max_length=max_article_length,
                                                    add_special_tokens=False, 
                                                    padding="max_length", #posant-ho a longest permetem que se faça a la longitud de la cadena més llarga, en lloc de 512 
                                                    pad_to_multiple_of=8,)["input_ids"]
        return model_inputs


    #Açò ho deixem preparat per a quan avaluem el model final, però per al model original no tenim eixa informació
    test_i_combinat = ds.map(lambda x: {
        "article": f"{x['article']}{tokenizer.eos_token}", #article en principi el deixem igual que estava
        "summary": f"{x['source']}{SEP_TOKEN}{x['summary']}{tokenizer.eos_token}",
    }, load_from_cache_file=carregar_de_cache)


    test_ni_combinat = ds_ni.map(lambda x: {
        "article": f"{x['article']}{tokenizer.eos_token}", #article en principi el deixem igual que estava
        "summary": f"{x['source']}{SEP_TOKEN}{x['summary']}{tokenizer.eos_token}",
    }, load_from_cache_file=carregar_de_cache)

    #ara com hem afegit un nou nivell d'indexacio, els column_names son els de qualsevol de les fonts
    tokenized_test_i = test_i_combinat.map(preproces_dades_test, load_from_cache_file=carregar_de_cache, batched=True, remove_columns=test_i_combinat[str(tokens_fonts[0])].column_names)
    tokenized_test_ni = test_ni_combinat.map(preproces_dades_test, load_from_cache_file=carregar_de_cache, batched=True, remove_columns=test_ni_combinat[str(tokens_fonts_ni[0])].column_names)

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
    """
        Funcio per avaluar els resultats del model: ROUGE, BERTscore i la tasca de classificació si escau
        @param model: model a avaluar
        @param test_dataset: dataset amb les dades de test
        @param avaluar_classif: si és True, avaluem també la classificació
        @return: diccionari amb els distints resultats de l'avaluació i la matriu de confusió si avaluar_classif era True
    """
    def evaluate_model(model, test_dataset):
        #resums_diferents = dict()
        conteig_resums_diferents = dict()
        noticies_amb_resum_diferent = set()
        matriu_confusio = dict()
        matriu_rouge = dict()     
        #com no dona temps d'executar-se tot cridem de columna en columna
        for tkn_font in test_dataset.keys():
            #resums_diferents[str(tkn_font)] = []
            conteig_resums_diferents[str(tkn_font)] = 0
            matriu_confusio[str(tkn_font)] = dict()
            matriu_rouge[str(tkn_font)] = dict()
            for token_font in tokens_fonts:
                matriu_confusio[str(tkn_font)][str(token_font)] = 0
                matriu_rouge[str(tkn_font)][str(token_font)] = 0
        for token_font in tqdm(tokens_fonts, desc="Columnes:", position=0): #este bucle governa les columnes (font forçada)
            for tkn_font in tqdm(test_dataset.keys(), desc="Files:", position=1): #este bucle governa les files (font original)
                if len(test_dataset[str(tkn_font)]) == 0: continue
                metric = evaluate.load("rouge") #resetegem cada vegada la mètrica perquè no influisca en unes i altres
                n_batch = 0
                for batch in tqdm(DataLoader(test_dataset[str(tkn_font)], collate_fn = data_collator, batch_size=8, shuffle=False), desc="Batch:", position = 2):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    input_ids = batch["input_ids"]
                    
                    summary_ids_original = model.generate(
                        input_ids, num_beams=4, min_length=25, max_length=150,
                        early_stopping=True, no_repeat_ngram_size=3, repetition_penalty=2.0,
                        #decoder_start_token_id=1, #forcem a que comence amb el token 1 i així no dona problemes lo del 2,1,1,1, però en realitat sí que dona perquè alguns són 1,1,1
                        #forced_bos_token_id = tkn_font if tkn_font is not None else model.config.forced_bos_token_id, #forcem a que comence com si fora de la font CA06
                    )

                    predictions_original = ([
                        tokenizer.decode(
                            g.tolist(), skip_special_tokens=True,
                            clean_up_tokenization_spaces=False
                        ) for g in summary_ids_original
                    ])
                    
                    predictions_original = ["\n".join(nltk.sent_tokenize(pred.strip(), language="spanish")) for pred in predictions_original]

                    noticies_resum_diferent = set()

                    summary_ids = model.generate(
                        input_ids, num_beams=4, min_length=25, max_length=150,
                        early_stopping=True, no_repeat_ngram_size=3, repetition_penalty=2.0,
                        #decoder_start_token_id=1, #forcem a que comence amb el token 1 i així no dona problemes lo del 2,1,1,1, però en realitat sí que dona perquè alguns són 1,1,1
                        forced_bos_token_id = token_font, #forcem a que comence com si fora de la font CA06
                    )
                    
                    predictions = ([
                        tokenizer.decode(
                            g.tolist(), skip_special_tokens=True,
                            clean_up_tokenization_spaces=False
                        ) for g in summary_ids
                    ])
                    
                    predictions = ["\n".join(nltk.sent_tokenize(pred.strip(), language="spanish")) for pred in predictions]
                    
                    for i in range(len(summary_ids_original)):
                        #comencem a mirar a partir del 3r token perquè si no se complirà sempre perquè tindrà una font diferent de l'original 
                        min_long = min(len(summary_ids_original[i]), len(summary_ids[i]))
                        for j in range(3, min_long):
                            #si son diferents en algun moment això vol dir que els resums ja no son iguals, però si tenen el mateix token seguim mirant
                            #a menys que eixe token siga el de padding, en eixe cas ja ha acabat el resum, però cap la possibilitat que tinguen longituds diferents
                            #llavors deixem de mirar per evitar falsos positius.
                            if summary_ids_original[i][j] != summary_ids[i][j]:
                                #resums_diferents[str(tkn_font)].append((predictions_original[i], predictions[i]))
                                conteig_resums_diferents[str(tkn_font)] += 1
                                matriu_confusio[str(tkn_font)][str(token_font)] += 1
                                noticies_amb_resum_diferent.add(str(tkn_font) + "_" + str(n_batch + i))
                                break
                            elif summary_ids_original[i][j] == tokenizer.pad_token_id: #and summary_ids[i][j] == tokenizer.pad_token_id:
                                break
                    n_batch += len(summary_ids_original)
                    metric.add_batch(predictions=predictions, references=predictions_original)
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
        for tkn_font in test_dataset.keys():
            for token_font in tokens_fonts:
                if len(test_dataset[str(tkn_font)]):
                    matriu_confusio[str(tkn_font)][str(token_font)] = matriu_confusio[str(tkn_font)][str(token_font)]  / len(test_dataset[str(tkn_font)])
        
        return conteig_resums_diferents, noticies_amb_resum_diferent, matriu_confusio, matriu_rouge

    f = open(eixida, "w")
    print("Resultats de comprovar si els resums generats són diferents si comencen amb un token de font diferent:")
    print("Amb les corresponents matrius \"de confusio\" i ROUGE:")
    f.write("Resultats de comprovar si els resums generats són diferents si comencen amb un token de font diferent:\n")
    f.write("Amb les corresponents matrius \"de confusio\" i ROUGE:\n")
    contador_i, contador_noticies_i, matriu_confusio_i, matriu_rouge_i = evaluate_model(model, tokenized_test_i)
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
        print(f"Font {font}: {contador_i[str(font)]} ({round(contador_i[str(font)] / (len(tokenized_test_i[str(font)]) * len(tokens_fonts)) * 100, 5)}%)")
        f.write(f"Font {font}: {contador_i[str(font)]} ({round(contador_i[str(font)] / (len(tokenized_test_i[str(font)]) * len(tokens_fonts)) * 100, 5)}%)\n")
    
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
    contador_ni, contador_noticies_ni, matriu_confusio_ni, matriu_rouge_ni = evaluate_model(model, tokenized_test_ni)
    print(f"Noticies amb resums diferents: {contador_noticies_ni} ({round(contador_noticies_ni / len(test_ni) * 100, 5)}%)")
    f.write(f"Noticies amb resums diferents: {contador_noticies_ni} ({round(contador_noticies_ni / len(test_ni) * 100, 5)}%)\n")
    for font in contador_ni.keys():
        print(f"Font {font}: {contador_ni[str(font)]} ({round (contador_ni[str(font)] / (len(tokenized_test_ni[str(font)]) * len(tokens_fonts)) * 100, 5)}%)")
        f.write(f"Font {font}: {contador_ni[str(font)]} ({round(contador_ni[str(font)] / (len(tokenized_test_ni[str(font)]) * len(tokens_fonts)) * 100, 5)}%)\n")
    
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
        print(f"Font {font}: {contador_i[str(font)]} ({contador_i[str(font)] / (len(tokenized_test_i[font]) * len(tokens_fonts)) * 100}%)")
        f.write(f"Font {font}: {contador_i[str(font)]} ({contador_i[str(font)] / (len(tokenized_test_i[font]) * len(tokens_fonts)) * 100}%)\n")
        for resum_original, resums_referencia in resums_diferents_i[str(font)]:
            print(f"\n\t{resum_original}\n\t{resums_referencia}\n")
            f.write(f"\n\t{resum_original}\n\t{resums_referencia}\n\n")
    print("-" * 50 + "\n" * 3)

    print("Exemples diferents en test_ni:")
    f.write("Exemples diferents en test_ni:\n")
    for font in contador_ni.keys():
        print(f"Font {font}: {contador_ni[str(font)]} ({contador_ni[str(font)] / (len(tokenized_test_ni[str(font)]) * len(tokens_fonts)) * 100}%)")
        f.write(f"Font {font}: {contador_ni[str(font)]} ({contador_ni[str(font)] / (len(tokenized_test_ni[str(font)]) * len(tokens_fonts)) * 100}%)\n")
        for resum_original, resums_referencia in resums_diferents_ni[str(font)]:
            print(f"\n\t{resum_original}\n\t{resums_referencia}\n")
            f.write(f"\n\t{resum_original}\n\t{resums_referencia}\n\n")"""
    f.close()
   
if __name__ == "__main__":
    main()