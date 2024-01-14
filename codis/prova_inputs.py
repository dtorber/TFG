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
carregar_de_cache=False


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

# Fonts que estàn presents en les particions d'entrenament de DACSA
SOURCES = {
    "ca": [
        f"CA{i:02d}" for i in range(1, 10) if i not in [7, 8, 9] 
    ],
    "es": [
        f"ES{i:02d}" for i in range(1, 22) if i not in [11] + list(range(15, 22))
    ]
}

# Token de serparació entre les fonts i el text
SEP_TOKEN = "<text>"

# Carreguem el tokenitzador original de NASca
checkpoint = "ELiRF/NASCA"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast = True)

# Els nous tokens especials seran tant el SEP_TOKEN com les claus de les
# fonts per a l'idioma en concret
tokenizer.add_tokens([SEP_TOKEN] + SOURCES["ca"], special_tokens=True)

#comprovem que tots els tokens espcials s'hagen convertit com un únic token i no vàrios
#for token_especial in SOURCES["ca"]:
#    print(tokenizer(f"{token_especial}<text>Hola món!"))


ds_combinat = ds_original.map(lambda x: {
    "article": f"{x['article']}{tokenizer.eos_token}", #article en principi el deixem igual que estava
    "summary": f"{x['source']}{SEP_TOKEN}{x['summary']}{tokenizer.eos_token}",
}, load_from_cache_file=carregar_de_cache)

# Carreguem el model original de NASca
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
#Important que reajustem la mida del token_embeddings perque coincideixi amb el nou tokenitzador (amb els nous tokens especials)
model.resize_token_embeddings(len(tokenizer))

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
print("Input_ids: " + str(ds_combinat_tokenized["train"]["input_ids"]))
print("Labels: " + str(ds_combinat_tokenized["train"]["labels"]))
print("Decoder_input_ids: " + str(ds_combinat_tokenized["train"]["decoder_input_ids"]))
input()

"""
print(ds_combinat["train"]["summary"])
print(ds_combinat["validation"]["summary"])
print("=" * 100)
for m in ds_combinat_tokenized["train"]:
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
    print("Decoder_input_ids: " + str(m["decoder_input_ids"]))
    print("-" * 50)
print("=" * 100)
for m in ds_combinat_tokenized["validation"]:
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
    print("Decoder_input_ids: " + str(m["decoder_input_ids"]))
    print("-" * 50)
print("=" * 100)
"""
#Ara cal que gastem un DataCollator, especial per a seq2seq, per tal de poder fer el padding i a més s'encarregarà de fer lo de que quan se genere una paraula
#només veja el que ha generat fins aleshores i no el que hi ha a continuació

label_relleno = -100 #valor que no es considera en el càlcul de la pèrdua
multiple = 8
data_collator = DataCollatorForSeq2Seq(tokenizer, 
                                        #model=model,
                                        label_pad_token_id=label_relleno,
                                        padding = "max_length",
                                        max_length = max_article_length,
                                        pad_to_multiple_of=multiple,)

#Prova del data_collator per veure com fa el decoder_input_ids
# print(ds_combinat_tokenized["train"])
for columna in ds_combinat_tokenized["train"].column_names:
    print(ds_combinat_tokenized["train"][columna])
input()
ds_amb_dec_ids = ds_combinat_tokenized["train"].to_tf_dataset(batch_size=10,collate_fn=data_collator)
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
    fp16=True,
)


#############################################################

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
#avaluem el resultat (en cada epoch hauriem de veure el training loss decerixer i el ROUGE pujar)