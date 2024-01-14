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
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

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
        Funcio per avaluar els resultats del model: ROUGE 
        @param subset: subconjunt de dades a avaluar
        @return: diccionari amb la matriu de ROUGE entre la font forçada i el resum de referència
    """
    def evaluate_model(model, test_dataset):
        #ara no volem una matriu el que volem és un diccionari amb la mitjana de rouge per a cada font
        rouge = dict()

        print(test_dataset) #aprofitem per veure les longituds ben fetes per si coincideixen

        for tkn_font in tqdm(test_dataset.keys(), desc="Files:", position=1): #este bucle governa les files (font original)
            if len(test_dataset[str(tkn_font)]) == 0: continue
            metric = evaluate.load("rouge") #resetegem cada vegada la mètrica perquè no influisca en unes i altres
            for batch in tqdm(DataLoader(test_dataset[str(tkn_font)], collate_fn = data_collator, batch_size=8, shuffle=False), desc="Batch:", position = 2):
                batch = {k: v.to(device) for k, v in batch.items()}
                input_ids = batch["input_ids"]
                resum_referencia = batch["labels"]


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
                
                resum_referencia = resum_referencia.cpu().numpy()
                resum_referencia = np.where(resum_referencia != -100, resum_referencia, tokenizer.pad_token_id)
                #Ací no faria falta també fer un padding_across_processes per tal de que tots els processos tinguin el mateix tamany independentment de quin procés els haja tokenitzat?
                referencies = ([
                    tokenizer.decode(
                        g.tolist(), skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    ) for g in resum_referencia
                ])

                predictions_original = ["\n".join(nltk.sent_tokenize(pred.strip(), language="spanish")) for pred in predictions_original]
                referencies = ["\n".join(nltk.sent_tokenize(label.strip(), language="spanish")) for label in referencies]
                metric.add_batch(predictions=predictions_original, references=referencies)
            resultat = metric.compute()
            rouge[str(tkn_font)] = resultat["rougeLsum"]
        return rouge

    vector_i = evaluate_model(model, tokenized_test_i)
    print(vector_i)
    escriure_vector(vector_i, open("rouge_per_font_i_resumint.txt", "w"))
    escriure_vector(vector_i, stdout)
    vector_ni = evaluate_model(model, tokenized_test_ni)
    print(vector_ni)
    escriure_vector(vector_ni, open("rouge_per_font_ni_resumint.txt", "w"))
    escriure_vector(vector_ni, stdout)


if __name__ == "__main__":
    main()