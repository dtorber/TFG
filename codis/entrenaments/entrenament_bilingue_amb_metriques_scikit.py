import datasets
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
)
from torch.utils.data import DataLoader
import evaluate
import torch
from tqdm.auto import tqdm
import nltk
import numpy as np
import sys
import wandb
import os
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import recall_score
import gzip
import json

nltk.download("punkt")

#Definim una variable global per tal de destriar si volem carregar els datasets des de la cau o volem recalcular-los (posar a False en el segon cas)
carregar_de_cache=False
carregar_fonts = True
carregar_de_hugging_face = True

#############################################################
############# Definicio de variables d'entorn  ##############
#############################################################
os.environ["WANDB_PROJECT"]="projecte-bilingue-diego"
os.environ["WANDB_LOG_MODEL"]="true"
os.environ["WANDB_WATCH"]="false"

#Primer carreguem les dades (se pot agafar un arxiu comprimit i ho descomprimeix)
entrenament = datasets.load_dataset("json", data_files="../corpus_anonimitzat/ca-es2-jsonl/train.json.gz")
validacio = datasets.load_dataset("json", data_files="../corpus_anonimitzat/ca-es2-jsonl/validation.json.gz")


#Despres ho clavem tot dins del mateix  DatasetDict
ds_original = datasets.DatasetDict({
    "train": entrenament["train"],
    "validation": validacio["train"],
})

"""
	I ara preparem les dades per combinar en la secció de resum, el que ja teníem i el token
	de la font de la que prové ("encriptada") i el de separació
"""


# Token de serparació entre les fonts i el text
SEP_TOKEN = "<text>"

# Carreguem el tokenitzador original de NASca
checkpoint = "./NAS-bilingue" #agafem com a model previ el que no ha aplicat nasca encara
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)

# Els nous tokens especials seran tant el SEP_TOKEN com les claus de les
# fonts per a l'idioma en concret
if carregar_fonts:
    tokenizer.add_tokens([SEP_TOKEN] + ["CA", "ES"] , special_tokens=True)

#comprovem que tots els tokens espcials s'hagen convertit com un únic token i no vàrios
#for token_especial in SOURCES["ca"]:
#    print(tokenizer(f"{token_especial}<text>Hola món!"))

#necessitem afegir-li el token de la font, però també li afegim un </s> al final perquè així ja no cal que l'afegim mitjançant el tokenizer
# #si no volem carregar les fonts, caldrà que li llevem el token de font, però hem de deixar encara així el eos final </s>

#########################################################################
############# AÇO HA DE CANVIAR PER FICAR LANG<SEP>LANG<SEP>  ###########
#########################################################################


ds_combinat = ds_original.map(lambda x: {
    "article": f"{x['article']}{tokenizer.eos_token}", #article en principi el deixem igual que estava
    "summary_es" : f"ES{SEP_TOKEN}{x['lang'].upper()}{SEP_TOKEN}{x['summary_es']}{tokenizer.eos_token}",
    "summary_ca" : f"CA{SEP_TOKEN}{x['lang'].upper()}{SEP_TOKEN}{x['summary_ca']}{tokenizer.eos_token}",
}, load_from_cache_file=carregar_de_cache)

nou_ds = {
    "train": dict(),
    "validation": dict(),
}

#ara falta generar per cadascuna de les mostres, en lloc d'una entrada per al resum en castellà i una per al català, dos mostres una per cada resum
categories = ["article", "summary", "id", "source", "lang_resum", "lang_article"]
for particio in ds_combinat:
    for categoria in categories:
        nou_ds[particio][categoria] = list() 
    for exemple in ds_combinat[particio]:
        nou_ds[particio]["article"].append(exemple["article"])
        nou_ds[particio]["summary"].append(exemple["summary_es"])
        nou_ds[particio]["id"].append(exemple["id"])
        nou_ds[particio]["source"].append(exemple["source"])
        nou_ds[particio]["lang_resum"].append("ES")
        nou_ds[particio]["lang_article"].append(exemple["lang"].upper())

        nou_ds[particio]["article"].append(exemple["article"])
        nou_ds[particio]["summary"].append(exemple["summary_ca"])
        nou_ds[particio]["id"].append(exemple["id"])
        nou_ds[particio]["source"].append(exemple["source"])
        nou_ds[particio]["lang_resum"].append("CA")
        nou_ds[particio]["lang_article"].append(exemple["lang"].upper())

        
ds_combinat = datasets.DatasetDict(
    {
        "train": datasets.Dataset.from_dict(nou_ds["train"]),
        "validation": datasets.Dataset.from_dict(nou_ds["validation"]),
    }
) 
#########################################################################
#########################################################################
#########################################################################


# Carreguem el model original de NASca, però li canviem el decoder_start_token perquè per defecte en BART comença per </s> 
# i per al nostre cas especial volem que comence per <s> perquè és el format que espera del que han entrenat ells.
config = AutoConfig.from_pretrained(checkpoint)
config.decoder_start_token_id = tokenizer.bos_token_id
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, config=config)
#Important que reajustem la mida del token_embeddings perque coincideixi amb el nou tokenitzador (amb els nous tokens especials)
if carregar_fonts:
    model.resize_token_embeddings(len(tokenizer))
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


#Ara cal que tokenitzem tant els articles com els resums de referència
#???? PREGUNTAR QUE FAIG AMB LA LONGITUD -> poden ser molt llargs, els trunquem els aprofitem amb lo de 
max_article_length = 512
def preprocessar_dades(examples):
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
                                                  padding="max_length", 
                                                  pad_to_multiple_of=8,)["input_ids"]
    return model_inputs

#Ara li apliquem la funció, amb batched a True per tal de poder paral·lelitzar i eliminem les columnes originals, per tal de poder passar-li-ho despres al DataCollator
ds_combinat_tokenized = ds_combinat.map(preprocessar_dades, batched=True, remove_columns=ds_combinat.column_names["train"], load_from_cache_file=carregar_de_cache)
#ds_original_tokenized = ds_original.map(preprocessar_dades, batched=True, remove_columns=ds_combinat.column_names["train"])

#Ara cal que gastem un DataCollator, especial per a seq2seq, per tal de poder fer el padding i a més s'encarregarà de fer lo de que quan se genere una paraula
#només veja el que ha generat fins aleshores i no el que hi ha a continuació

label_relleno = -100 #valor que no es considera en el càlcul de la pèrdua
multiple = 8
data_collator = DataCollatorForSeq2Seq(tokenizer, 
                                        #model=model,
                                        label_pad_token_id=label_relleno,
                                        padding = "max_length",
                                        max_length = max_article_length,
                                        pad_to_multiple_of=multiple)


#Ara avaluem el model amb el que ja està preentrenat i fine-tunejat, per veure si el resultat és millor
#Ho fem tant per al dataset conforme ve donat i per al que li afegim els tokens de les fonts
metric = evaluate.load("rouge")
batch_size = 4
epoch = 1
def calcular_metriques (eval_pred):
    global epoch
    resum_generat, resum_referencia = eval_pred
    #print(resum_generat, resum_referencia)
    if isinstance(resum_generat, tuple): resum_generat = resum_generat[0]
    #Decodificar el resum generat:
    #ho deixem a False perquè si se salta els special_tokens no veurem la font
    decoded_resum_generat = tokenizer.batch_decode(resum_generat, skip_special_tokens=True)
    #print("a")
    #Canviar el -100 per 0 que haviem posat per tal de que no es considere en el càlcul de la pèrdua
    resum_referencia = np.where(resum_referencia != -100, resum_referencia, tokenizer.pad_token_id)
    #print("b")
    #descodificar el resum de referència
    decoded_resum_referencia = tokenizer.batch_decode(resum_referencia, skip_special_tokens=True)
    #print("c")
    #rouge espera que els resums estiguen separats per un \n
    #esta part de "\n" no està feta en el paper 
    decoded_resum_generat = ["\n".join(nltk.sent_tokenize(pred.strip(), language="spanish")) for pred in decoded_resum_generat]
    #print("d")
    decoded_resum_referencia = ["\n".join(nltk.sent_tokenize(label.strip(), language="spanish")) for label in decoded_resum_referencia]
    #print("e")
    #Avaluem amb ROUGE, però NOMES rougeLsum
    resultat = metric.compute(predictions=decoded_resum_generat, references=decoded_resum_referencia, rouge_types = ["rougeLsum"])

    #check if resums_generats_entrenament directory exists
    if not os.path.exists("./resums_generats_entrenament"):
        os.mkdir("./resums_generats_entrenament")

    #guardem per a cadascun dels epochs tots els resultats que haja generat
    with open(f"./resums_generats_entrenament/decoded_resum_generat_bilingue_{epoch}.txt", "w") as fitxer:
        for resum in decoded_resum_generat:
            fitxer.write(str(resum) + "\n\n")
        epoch += 1

    return resultat

#Ara cal que creem el Trainer, que és el que s'encarregarà d'entrenar el model
num_train_epochs = 15
#aço indica cada quantes vegades s'actualitza la informació en wandb per això no interessa que siga cada molt temps
logging_steps = 1000
#preparem els arguments que li passaerem al nostre Trainer: hiperparametres i algunes altres caracteristiques 
args = Seq2SeqTrainingArguments(
    output_dir= "NAS-bilingue-final",
    #output_dir="output",
    report_to="wandb",
    evaluation_strategy="epoch",
    learning_rate=0.0000013739167643078955, #hem agafat per a este original el learning_rate que va dir l'optuna la primera vegada
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    eval_accumulation_steps=4,
    num_train_epochs=num_train_epochs,
    save_total_limit = 3,
    save_strategy = "epoch",
    load_best_model_at_end = True,
    metric_for_best_model = "rougeLsum", #en lloc de tindre en compte la loss per a triar el mateix model, se triara el que tinga millor ROUGE LSum
    greater_is_better = True, #perque el ROUGE LSum es mes gran si es millor
    #save_steps = 50000,
    weight_decay=0.00021965549621889524, #hem agafat per a este original el weight_decay que va dir l'optuna la primera vegada
    logging_steps=logging_steps,
    push_to_hub=True,
    predict_with_generate=True, #perquè se generen resums durant l'avaluació i que se puga anar calculant el ROUGE mentrestant
    fp16=torch.cuda.is_available(),
)


#I ara ja entrenem el model però amb els hiperparametres optimitzats amb optuna
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    data_collator=data_collator,
    train_dataset=ds_combinat_tokenized["train"],
    eval_dataset=ds_combinat_tokenized["validation"],
    tokenizer=tokenizer,
    compute_metrics=calcular_metriques,
)


#llancem l'entrenament automàtic -> mirar si fem aixo mitjançant el loop amb accelerate o amb alguna eina de deepspeed o algo
trainer.train()

#guardem la versió definitiva
trainer.save_model("NAS-bilingue-final")
#avaluem el resultat (en cada epoch hauriem de veure el training loss decerixer i el ROUGE pujar)

trainer.push_to_hub(commit_message="Training complete", tags="summarization")
wandb.finish()
