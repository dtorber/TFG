"""
    creem un script que s'encarregara de generar tots els resums possibles i emmagatzemar-los en un fitxer JSON per tal de poder gastar-los quan siga menester
"""

import json
from datetime import datetime
import datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
import evaluate
import numpy as np
import torch
import nltk
from tqdm import tqdm
from sys import stdout
import os
from diffusers import StableDiffusionPipeline

tokens_llengua_a_id = {
    "<lang:ca>": 60002,
    "<lang:es>": 60003
}
id_a_tokens_llengua = {
    60002: "<lang:ca>",
    60003: "<lang:es>"
}

def main(checkpoint= "NAS-bilingue", carpeta = "./resums_generats_bilingue", carregar_de_hugginface = False, carregar_de_cache = True, llengua = "ca"):
    nltk.download('punkt')
    #carregar_de_cache = True
    #SI CANVIEM LA LLENGUA CANVIAR ELS TOKENS FONTS QUE FALTEN ELS DE CASTELLÀ
    # llengua = "ca"
    #carregar_de_hugginface = True

    #carreguem el tokenitzador i el model
    #checkpoint = "NAS" + llengua + "-finetuned-diego-4-amb-metriques-anonim/checkpoint-650000"
    if carregar_de_hugginface:
        checkpoint = "dtorber/" + checkpoint

    # Token de serparació entre les fonts i el text
    SEP_TOKEN = "<text>"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only = not carregar_de_hugginface, use_fast=True)
    # model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, use_safetensors=True)
    model = StableDiffusionPipeline.from_single_file(checkpoint, use_safetensors=True)

    # Els nous tokens especials seran tant el SEP_TOKEN com les claus de les
    # fonts per a l'idioma en concret, pero en principi ja estan en el tokenizer que hem descarregat
    #tokenizer.add_tokens([SEP_TOKEN] + SOURCES["ca"], special_tokens=True)

    #carreguem els datasets per a testejar:
    test_i = datasets.load_dataset("json", data_files="../corpus_anonimitzat/ca-es/test-i.json.gz")
    test_ni = datasets.load_dataset("json", data_files="../corpus_anonimitzat/ca-es/test-ni.json.gz")

    #reduim els tamanys dels datasets per fer simplement proves:
    test_i = test_i["train"]
    test_ni = test_ni["train"]
    
    nou_test_i = dict()
    nou_test_ni = dict()
    
    features = dict()
    for key in test_i.features.keys():
        features[key] = list()
        nou_test_i = features
    
    features = dict()
    for key in test_ni.features.keys():
        features[key] = list()
        nou_test_ni = features
    

    for i in range(len(test_i)):
        for k,v in test_i[i].items():
            if v == None:
                v = ""
            nou_test_i[k].append(v)

    nou_test_i = datasets.Dataset.from_dict(nou_test_i)
    test_i = nou_test_i

    for i in range(len(test_ni)):
        for k,v in test_ni[i].items():
            if v == None:
                v = ""
            nou_test_ni[k].append(v)

    nou_test_ni = datasets.Dataset.from_dict(nou_test_ni)
    test_ni = nou_test_ni

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
    test_i_combinat = test_i.map(lambda x: {
        "article": f"{x['article']}{tokenizer.eos_token}", #article en principi el deixem igual que estava
        "summary": f"{x['source']}{SEP_TOKEN}{x['summary_ca']}{tokenizer.eos_token}",
    }, load_from_cache_file=carregar_de_cache)


    test_ni_combinat = test_ni.map(lambda x: {
        "article": f"{x['article']}{tokenizer.eos_token}", #article en principi el deixem igual que estava
        "summary": f"{x['source']}{SEP_TOKEN}{x['summary_ca']}{tokenizer.eos_token}",
    }, load_from_cache_file=carregar_de_cache)

    tokenized_test_i = test_i_combinat.map(preproces_dades_test, load_from_cache_file=carregar_de_cache, batched=True, remove_columns=test_i_combinat.column_names)
    tokenized_test_ni = test_ni_combinat.map(preproces_dades_test, load_from_cache_file=carregar_de_cache, batched=True, remove_columns=test_ni_combinat.column_names)

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
        Funcio per guardar en un json indexat per id de notícia el resum generat forçant (o no) la font
        @param model: model a avaluar
        @param test_dataset: dataset amb les dades de test
        @param ruta: ruta on es guardaran els resultats
        @param forced_token: token que es forçarà a que aparegui al principi de cada resum generat o None si no es vol forçar cap token
        @param min_length: longitud mínima del resum generat
        @param max_length: longitud màxima del resum generat
        @return el json que hem creat per si se vol aprofitar per alguna cosa més que només escriure-ho en un fitxer
    """
    def generate_summaries(model, test_dataset, test_original, ruta = f"./resums_generats/{datetime.now()}.json", subset = "test_i", min_length = 25, max_length = 150):        
        i = 0
        json_data = {
            "ca": dict(),
            "es": dict(),
        }
        for token_lang in ["<lang:ca>", "<lang:es>"]:
            forced_token = tokens_llengua_a_id[token_lang]
            for batch, batch_original in zip(tqdm(DataLoader(test_dataset, collate_fn = data_collator, batch_size=8, shuffle=False), desc="Generant resums"), DataLoader(test_original, batch_size=8, shuffle=False)):
                batch = {k: v.to(device
                ) for k, v in batch.items()}
                input_ids = batch["input_ids"]

                summary_ids = model.generate(
                    input_ids, num_beams=4, min_length=min_length, max_length=max_length,
                    early_stopping=True, no_repeat_ngram_size=3, repetition_penalty=2.0,
                    #decoder_start_token_id=1, #forcem a que comence amb el token 1 i així no dona problemes lo del 2,1,1,1, però en realitat sí que dona perquè alguns són 1,1,1
                    forced_bos_token_id = forced_token, #forcem a que comence amb el token de la llengua en que ha de resumir
                )
                
                predictions = ([
                    tokenizer.decode(
                        g.tolist(), skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    ) for g in summary_ids
                ])
        
                predictions = ["\n".join(nltk.sent_tokenize(pred.strip(), language="spanish")) for pred in predictions]
                #for pred, id, source in zip(predictions, batch_original["id"], batch_original["source"]):
                for i, pred in enumerate(predictions):
                    #es la i-essima noticia que hem recorregut, agafem el seu id de la llista i el posem com a clau del diccionari i li donem el resum generat per a eixa notícia
                    #id, source = test_i[i]["id"], test_i[i]["source"]
                    id = batch_original["id"][i]
                    source = batch_original["source"][i]
                    lang = batch_original["lang"][i]
                    pred_lang = summary_ids[i].tolist()[3]
                    if lang not in json_data:
                        json_data[lang] = dict()
                    if id not in json_data[lang]:
                        json_data[lang][id] = {
                            "source": source,
                            "lang": lang.upper(),
                        }
                    json_data[lang][id][f"summary_{token_lang.lower()}"] = pred
                    json_data[lang][id][f"pred_idioma_article_{token_lang.lower()}"] = id_a_tokens_llengua[pred_lang]
        
            #escrivim el resultat en un fitxer
            for llengua in json_data.keys():
                with open(f"{ruta}_{llengua}.json" , "w") as f:
                    json.dump(json_data[llengua], f)
            
        return json_data
    
    #generem els resums per a les notícies amb informació de font RESUMS NORMALS
    generate_summaries(model, tokenized_test_i, test_i, ruta = f"{carpeta}/resums_normals/resums_test_i_original", subset = "test_i", min_length = 25, max_length = 150)
    generate_summaries(model, tokenized_test_ni, test_ni, ruta = f"{carpeta}/resums_normals/resums_test_ni_original", subset = "test_ni", min_length = 25, max_length = 150)

    # #generem els resums per a les notícies amb informació de font RESUMS LLARGS
    # generate_summaries(model, tokenized_test_i, test_i, ruta = f"{carpeta}/resums_llargs/resums_test_i_original", subset = "test_i", min_length = 70, max_length = 420)
    # generate_summaries(model, tokenized_test_ni, test_ni, ruta = f"{carpeta}/resums_llargs/resums_test_ni_original", subset = "test_ni", min_length = 70, max_length = 420)

if __name__ == "__main__":
    checkpoint = "checkpoint-119328/model.safetensors"
    nom_carpeta = "resums_generats_bilingue"
    if not os.path.exists(nom_carpeta):
        os.makedirs(nom_carpeta)

    main(checkpoint=checkpoint,carpeta = f"./{nom_carpeta}")