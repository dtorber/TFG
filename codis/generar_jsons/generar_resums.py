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

def main(checkpoint= "NASca-finetuned-de-zero-amb-metriques-anonim", carpeta = "./resums_generats", carregar_de_hugginface = False, carregar_de_cache = True, llengua = "ca"):
    nltk.download('punkt')
    #carregar_de_cache = True
    #SI CANVIEM LA LLENGUA CANVIAR ELS TOKENS FONTS QUE FALTEN ELS DE CASTELLÀ
    # llengua = "ca"
    tokens_fonts = list(range(60002, 60007 + 1)) if llengua == "ca" else list(range(60002, 60022 + 1))
    #carregar_de_hugginface = True

    #carreguem el tokenitzador i el model
    #checkpoint = "NAS" + llengua + "-finetuned-diego-4-amb-metriques-anonim/checkpoint-650000"
    if carregar_de_hugginface:
        checkpoint = "dtorber/" + checkpoint

    # Token de serparació entre les fonts i el text
    SEP_TOKEN = "<text>"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only = not carregar_de_hugginface, use_fast=True)
    tokenizer.add_tokens(["CA07", "CA08", "CA09"], special_tokens=True) #afegim els tokens de les fonts desconegudes
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
        "summary": f"{x['source']}{SEP_TOKEN}{x['summary']}{tokenizer.eos_token}",
    }, load_from_cache_file=carregar_de_cache)


    test_ni_combinat = test_ni.map(lambda x: {
        "article": f"{x['article']}{tokenizer.eos_token}", #article en principi el deixem igual que estava
        "summary": f"{x['source']}{SEP_TOKEN}{x['summary']}{tokenizer.eos_token}",
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
        @return el json que hem creat per si se vol aprofitar per alguna cosa més que només escriure-ho en un fitxer
    """
    def generate_summarys(model, test_dataset, test_original, ruta = f"./resums_generats/{datetime.now()}.json", forced_token = None, subset = "test_i"):        
        i = 0
        json_data = dict()
        for batch, batch_original in zip(tqdm(DataLoader(test_dataset, collate_fn = data_collator, batch_size=8, shuffle=False), desc="Generant resums"), DataLoader(test_original, batch_size=8, shuffle=False)):
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids = batch["input_ids"]

            summary_ids = model.generate(
                input_ids, num_beams=4, min_length=25, max_length=150,
                early_stopping=True, no_repeat_ngram_size=3, repetition_penalty=2.0,
                #decoder_start_token_id=1, #forcem a que comence amb el token 1 i així no dona problemes lo del 2,1,1,1, però en realitat sí que dona perquè alguns són 1,1,1
                forced_bos_token_id = forced_token if forced_token is not None else model.config.forced_bos_token_id, #forcem a que comence com si fora de la font CA06
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
                id, source = batch_original["id"][i], batch_original["source"][i]
                json_data[id] = {
                    "source": source,
                    "summary": pred
                }  
        #escribim el resultat en un fitxer
        with open(ruta, "w") as f:
            json.dump(json_data, f)
        

        return json_data

    """
        Funcio per guardar en un json indexat per id de notícia el resum generat forçant la font correcta
        @param model: model a avaluar
        @param test_dataset: dataset amb les dades de test
        @param ruta: ruta on es guardaran els resultats
        @return el json que hem creat per si se vol aprofitar per alguna cosa més que només escriure-ho en un fitxer
    """
    def generate_summary_font_correcta(model, dataset_original, ruta = f"./resums_generats/{datetime.now()}.json"):        
        i = 0
        json_data = {}

        #primer fa falta preparar les dades perquè estiguen agrupades per fonts
        dataset = dict()	
        #creem un dataset que estiga organitzat per font i aixi podem utilitzar el batch, seguim el mateix format que tenia originalment el test_i
        for token in tokens_fonts:
            features = {}
            for key in dataset_original.features.keys():
                features[key] = []
            dataset[str(token)] = features

        #ara anem mostra a mostra afegint-les al diccionari que toque
        for mostra in dataset_original:
            token_font = tokenizer(mostra["source"])["input_ids"][1]
            dataset[str(token_font)]["article"].append(mostra["article"])
            dataset[str(token_font)]["summary"].append(mostra["summary"])
            dataset[str(token_font)]["source"].append(mostra["source"])
            dataset[str(token_font)]["id"].append(mostra["id"])

        #una vegada ho tenim preparat en format diccionari cal passar-ho a DatasetDict per poder aplicar totes les operacion
        ds = datasets.DatasetDict({})
        for token in tokens_fonts:
            ds[str(token)] = datasets.Dataset.from_dict(dataset[str(token)])
        
        test_i_combinat = ds.map(lambda x: {
            "article": f"{x['article']}{tokenizer.eos_token}", #article en principi el deixem igual que estava
            "summary": f"{x['source']}{SEP_TOKEN}{x['summary']}{tokenizer.eos_token}",
        }, load_from_cache_file=carregar_de_cache)

        #per eliminar les columnes com ara tenim un primer nivell d'indexació que són les fonts, les columnes realment són les columnes de cada font aleshores agarrem les d'una qualsevol
        test_dataset = test_i_combinat.map(preproces_dades_test, load_from_cache_file=carregar_de_cache, batched=True, remove_columns=test_i_combinat[str(tokens_fonts[0])].column_names)

        for token_font in tokens_fonts:
            #per si de cas hi ha alguna partició que esta buida, eixe cas no l'avaluariem.
            if len(test_dataset[str(token_font)]) == 0:
                continue
            i = 0 #ho resetegem cada vegada perquè estarem dins del subset de cadascuna de les fonts
            for batch in tqdm(DataLoader(test_dataset[str(token_font)], collate_fn = data_collator, batch_size=8, shuffle=False), desc="Generant resums"):
                batch = {k: v.to(device) for k, v in batch.items()}
                input_ids = batch["input_ids"]
             
                summary_ids = model.generate(
                    input_ids, num_beams=4, min_length=25, max_length=150,
                    early_stopping=True, no_repeat_ngram_size=3, repetition_penalty=2.0,
                    #decoder_start_token_id=1, #forcem a que comence amb el token 1 i així no dona problemes lo del 2,1,1,1, però en realitat sí que dona perquè alguns són 1,1,1
                    forced_bos_token_id = token_font, #forcem a que comence amb la font que li correspon
                )
                
                predictions = ([
                    tokenizer.decode(
                        g.tolist(), skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    ) for g in summary_ids
                ])
        
                predictions = ["\n".join(nltk.sent_tokenize(pred.strip(), language="spanish")) for pred in predictions]
                for pred in predictions:
                    #es la i-essima noticia que hem recorregut, agafem el seu id de la llista i el posem com a clau del diccionari i li donem el resum generat per a eixa notícia
                    #ara es important que mirem ds[str(token_font)] perquè no estan ja en un ordre general
                    id, source = ds[str(token_font)][i]["id"], ds[str(token_font)][i]["source"]
                    json_data[id] = {
                        "source": source,
                        "summary": pred
                    }
                    i += 1
        #escrivim el resultat en un fitxer
        with open(ruta, "w") as f:
            json.dump(json_data, f)
        
        return json_data
    #generem els resums per a les notícies amb informació de font
    generate_summarys(model, tokenized_test_i, test_i, ruta = f"{carpeta}/resums_test_i_original.json", subset = "test_i", )
    generate_summarys(model, tokenized_test_ni, test_ni, ruta = f"{carpeta}/resums_test_ni_original.json", subset = "test_ni")
    for token in tokens_fonts:
        generate_summarys(model, tokenized_test_i, test_i, ruta = f"{carpeta}/resums_test_i_forçat_{token}.json", forced_token = token, subset = "test_i")
        generate_summarys(model, tokenized_test_ni, test_ni,  ruta = f"{carpeta}/resums_test_ni_forçat_{token}.json", forced_token = token, subset = "test_ni")

    #per a este cas cal passar-li el test_i original perquè s'ha de preprocessar per tal de generar els resums amb la font correcta
    generate_summary_font_correcta(model, test_i, ruta = f"{carpeta}/resums_test_i_font_correcta.json")


if __name__ == "__main__":
    main(carpeta = "./resums_generats_primer_model")