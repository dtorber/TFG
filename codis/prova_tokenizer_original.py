import datasets
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
)
from torch.utils.data import DataLoader
import evaluate
import torch
from tqdm.auto import tqdm
import nltk
import numpy as np
import sys
import os
import tensorflow as tf
nltk.download("punkt")

#Definim una variable global per tal de destriar si volem carregar els datasets des de la cau o volem recalcular-los (posar a False en el segon cas)
carregar_de_cache=True


#Primer carreguem les dades (se pot agafar un arxiu comprimit i ho descomprimeix)
validacio = datasets.load_dataset("json", data_files="../corpus_anonimitzat/ca/validation.json.gz")
entrenament = datasets.load_dataset("json", data_files="../corpus_anonimitzat/ca/train.json.gz")

#Despres ho clavem tot dins del mateix  DatasetDict
ds_original = datasets.DatasetDict({
    "train": entrenament["train"].shuffle(seed=42).select(range(10)),
    "validation": validacio["train"].shuffle(seed=42).select(range(2)),
})

"""
	I ara preparem les dades per combinar en la secció de resum, el que ja teníem i el token
	de la font de la que prové ("encriptada") i el de separació
"""

# Token de serparació entre les fonts i el text
SEP_TOKEN = "<text>"

# Carreguem el tokenitzador original de NASca
checkpoint = "ELiRF/NASCA"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast = True)

# Carreguem el model original de NASca
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
#Important que reajustem la mida del token_embeddings perque coincideixi amb el nou tokenitzador (amb els nous tokens especials)

#Ara cal que tokenitzem tant els articles com els resums de referència
#???? PREGUNTAR QUE FAIG AMB LA LONGITUD -> poden ser molt llargs, els trunquem els aprofitem amb lo de 
max_article_length = 512
def preprocessar_dades(examples):
    model_inputs = tokenizer(examples["article"],
                             truncation=True,
                             #padding="max_length",
                             max_length=max_article_length)
    labels = tokenizer(text_target = examples["summary"], truncation=True) #perquè ho tokenitze com a target hem d'indicar-ho
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

#Ara li apliquem la funció, amb batched a True per tal de poder paral·lelitzar i eliminem les columnes originals, per tal de poder passar-li-ho despres al DataCollator
ds_original_tokenized = ds_original.map(preprocessar_dades, batched=True, remove_columns=ds_original.column_names["train"], load_from_cache_file=carregar_de_cache)
print(ds_original["train"]["summary"])
print(ds_original["validation"]["summary"])
print("=" * 100)
for m in ds_original_tokenized["train"]:
    print("Input_ids:" + str(m["input_ids"][:5])+ "..." , end=" ")
    trobat = False
    for i in range(5, len(m["input_ids"]) - 1):
        if m["input_ids"][i + 1] == tokenizer.pad_token_id:
            print(m["input_ids"][i], end="\n")
            trobat = True
            break
    if not trobat:
        print(m["input_ids"][-1])
    print("Labels: " + str(m["labels"]))
    print("-" * 50)
print("=" * 100)
for m in ds_original_tokenized["validation"]:
    print("Input_ids:" + str(m["input_ids"][:5])+ "..." , end=" ")
    trobat = False
    for i in range(len(m["input_ids"]) - 1):
        if m["input_ids"][i + 1] == tokenizer.pad_token_id:
            print(m["input_ids"][i], end="\n")
            trobat = True
            break
    if not trobat:
        print(m["input_ids"][-1])
    print("Labels: " + str(m["labels"]))
    print("-" * 50)
print("=" * 100)

#Ara cal que gastem un DataCollator, especial per a seq2seq, per tal de poder fer el padding i a més s'encarregarà de fer lo de que quan se genere una paraula
#només veja el que ha generat fins aleshores i no el que hi ha a continuació

label_relleno = -100 #valor que no es considera en el càlcul de la pèrdua
multiple = 8
data_collator = DataCollatorForSeq2Seq(tokenizer, 
                                        model=model,
                                        label_pad_token_id=label_relleno,
                                        padding = "max_length",
                                        max_length = max_article_length,
                                        pad_to_multiple_of=multiple,)

print(ds_original_tokenized["train"])
input()
ds_amb_dec_ids = ds_original_tokenized["train"].to_tf_dataset(batch_size=10,collate_fn=data_collator)
print("DataCollator decoder_input_ids:")
llista = list(ds_amb_dec_ids)
print("Llista: " + str(llista))
input()
ds = tf.data.Dataset.from_tensor_slices(llista)
print("DS: " + str(ds))
print("-" * 100)
input()
print(ds["decoder_input_ids"])
input()

#Ara avaluem el model amb el que ja està preentrenat i fine-tunejat, per veure si el resultat és millor
#Ho fem tant per al dataset conforme ve donat i per al que li afegim els tokens de les fonts
metric = evaluate.load("rouge")
def calcular_metriques (eval_pred):
    resum_generat, resum_referencia = eval_pred
    #print(resum_generat, resum_referencia)
    if isinstance(resum_generat, tuple): resum_generat = resum_generat[0]
    #Decodificar el resum generat:
    decoded_resum_generat = tokenizer.batch_decode(resum_generat, skip_special_tokens=True)
    #print("a")
    #Canviar el -100 per 0 que haviem posat per tal de que no es considere en el càlcul de la pèrdua
    resum_referencia = np.where(resum_referencia != -100, resum_referencia, tokenizer.pad_token_id)
    #print("b")
    #descodificar el resum de referència
    decoded_resum_referencia = tokenizer.batch_decode(resum_referencia, skip_special_tokens=True)
    #print("c")
    #rouge espera que els resums estiguen separats per un \nç
    #esta part de "\n" no està feta en el paper 
    decoded_resum_generat = ["\n".join(nltk.sent_tokenize(pred.strip(), language="spanish")) for pred in decoded_resum_generat]
    #print("d")
    decoded_resum_referencia = ["\n".join(nltk.sent_tokenize(label.strip(), language="spanish")) for label in decoded_resum_referencia]
    #print("e")
    #Avaluem amb ROUGE
    resultat = metric.compute(predictions=decoded_resum_generat, references=decoded_resum_referencia)
    #print(resultat)
    #Però ens dona molts resultats, de totes les mètriques ens quedarem només amb els de la mitjana
    return resultat

#Ara cal que creem el Trainer, que és el que s'encarregarà d'entrenar el model
batch_size = 4
num_train_epochs = 1
#aço indica cada quantes vegades s'actualitza la informació en wandb per això no interessa que siga cada molt temps
logging_steps = 100
#preparem els arguments que li passaerem al nostre Trainer: hiperparametres i algunes altres caracteristiques 
args = Seq2SeqTrainingArguments(
    output_dir = "output",
    evaluation_strategy="epoch",
    learning_rate=0.0000013739167643078955, #hem agafat per a este original el learning_rate que va dir l'optuna la primera vegada
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    eval_accumulation_steps=4,
    num_train_epochs=num_train_epochs,
    save_total_limit = 3,
    save_steps = 10000,
    weight_decay=0.00021965549621889524, #hem agafat per a este original el weight_decay que va dir l'optuna la primera vegada
    logging_steps=logging_steps,
    predict_with_generate=True, #perquè se generen resums durant l'avaluació i que se puga anar calculant el ROUGE mentrestant
)


#############################################################

#I ara ja entrenem el model però amb els hiperparametres optimitzats amb optuna
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    data_collator=data_collator,
    train_dataset=ds_original_tokenized["train"],
    eval_dataset=ds_original_tokenized["validation"],
    tokenizer=tokenizer,
    compute_metrics=calcular_metriques,
)

#llancem l'entrenament automàtic -> mirar si fem aixo mitjançant el loop amb accelerate o amb alguna eina de deepspeed o algo
trainer.train()
#avaluem el resultat (en cada epoch hauriem de veure el training loss decerixer i el ROUGE pujar)