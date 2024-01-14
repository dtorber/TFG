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

nltk.download("punkt")

#Definim una variable global per tal de destriar si volem carregar els datasets des de la cau o volem recalcular-los (posar a False en el segon cas)
carregar_de_cache=True
llengua = "ca"
carregar_fonts = True
tokens_fonts = list(range(60002, 60010 + 1)) if llengua == "ca" else list(range(60002, 60022 + 1))
carregar_de_hugging_face = True

#############################################################
############# Definicio de variables d'entorn  ##############
#############################################################
os.environ["WANDB_PROJECT"]="projecte-nas" + llengua + "-diego"
os.environ["WANDB_LOG_MODEL"]="true"
os.environ["WANDB_WATCH"]="false"

#Primer carreguem les dades (se pot agafar un arxiu comprimit i ho descomprimeix)
validacio = datasets.load_dataset("json", data_files="../../corpus/" + llengua +"/validation.json.gz")
entrenament = datasets.load_dataset("json", data_files="../../corpus/" + llengua +"/train.json.gz")
#Comentem els test perquè en principi això va en un altre script
#test_i = datasets.load_dataset("json", data_files="../corpus_anonimitzat/" + llengua +"/test-i.json.gz")
#test_ni = datasets.load_dataset("json", data_files="../corpus_anonimitzat/" + llengua +"/test-ni.json.gz")

#Despres ho clavem tot dins del mateix  DatasetDict
ds_original = datasets.DatasetDict({
    "train": entrenament["train"],
    "validation": validacio["train"],
    #"test-i": test_i["train"],
    #"test-ni": test_ni["train"]
})

"""
	I ara preparem les dades per combinar en la secció de resum, el que ja teníem i el token
	de la font de la que prové ("encriptada") i el de separació
"""

# Fonts que estàn presents en les particions d'entrenament de DACSA
SOURCES = {
    "ca": [
        f"CA{i:02d}" for i in range(1, 10) #if i not in [7, 8, 9] 
    ],
    "es": [
        f"ES{i:02d}" for i in range(1, 22) #if i not in [11] + list(range(15, 22))
    ]
}

# Token de serparació entre les fonts i el text
SEP_TOKEN = "<text>"

# Carreguem el tokenitzador original de NASca
checkpoint = "NASca-finetuned-de-zero-amb-metriques-anonim" #agafem com a model previ el que no ha aplicat nasca encara
if carregar_de_hugging_face:
    checkpoint = "dtorber/" + checkpoint

tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)

# Els nous tokens especials seran tant el SEP_TOKEN com les claus de les
# fonts per a l'idioma en concret
if carregar_fonts:
    tokenizer.add_tokens([SEP_TOKEN] + SOURCES[llengua], special_tokens=True)

#comprovem que tots els tokens espcials s'hagen convertit com un únic token i no vàrios
#for token_especial in SOURCES["ca"]:
#    print(tokenizer(f"{token_especial}<text>Hola món!"))

#necessitem afegir-li el token de la font, però també li afegim un </s> al final perquè així ja no cal que l'afegim mitjançant el tokenizer
# #si no volem carregar les fonts, caldrà que li llevem el token de font, però hem de deixar encara així el eos final </s>
ds_combinat = ds_original.map(lambda x: {
    "article": f"{x['article']}{tokenizer.eos_token}", #article en principi el deixem igual que estava
    "summary": f"{x['source']}{SEP_TOKEN}{x['summary']}{tokenizer.eos_token}" if carregar_fonts else f"{x['summary']}{tokenizer.eos_token}",
}, load_from_cache_file=carregar_de_cache)


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
avaluar_classif = carregar_fonts #només avaluarem la classificació si és possible fer-ho, és a dir si hem carregat les fonts
avaluar_bert = False
metric = evaluate.load("rouge")
if avaluar_bert:
    bert = evaluate.load("bertscore")
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
    #Avaluem amb ROUGE
    resultat = metric.compute(predictions=decoded_resum_generat, references=decoded_resum_referencia)

    #check if resums_generats_entrenament directory exists
    if not os.path.exists("./resums_generats_entrenament"):
        os.mkdir("./resums_generats_entrenament")

    #guardem per a cadascun dels epochs tots els resultats que haja generat
    with open(f"./resums_generats_entrenament/decoded_resum_generat_{llengua}_{epoch}.txt", "w") as fitxer:
        for resum in decoded_resum_generat:
            fitxer.write(str(resum) + "\n\n")
        epoch += 1

    if avaluar_bert:
        resultat_bert = bert.compute(predictions=decoded_resum_generat, references=decoded_resum_referencia, lang=llengua,)
        for key, values in resultat_bert.items():
            if type(values) == list:
                resultat["bert_" + key] = np.mean(np.array(values))

    #no tinc clar com se faria esta avaluació, perquè crec que s'hauria de fer fora? Perquè no tinc
    #clar què és el que arriba aci, imprimir-ho primer
    #tot lo que s'ha deixat comentat és perquè realment no ens fa falta calcular-ho lo important es la f1 i tot aixo
    if avaluar_classif:
        #evaluate classification of the font with scikit library (it is the first token of resum_referencia and the second of resun_generat)
        fonts_generades = np.array([pred[1] for pred in resum_generat])
        fonts_referencia = np.array([ref[0] for ref in resum_referencia])
        #calculate precision, recall and f1 with scikit
        precision, recall, f1, _ = precision_recall_fscore_support(fonts_referencia, fonts_generades, average="macro", labels=tokens_fonts)
        resultat["classificacio_precision"] = np.mean(precision)
        resultat["classificacio_recall"] = np.mean(recall)
        resultat["classificacio_f1"] = np.mean(f1)
        resultat["classicacio_accuracy"] = accuracy_score(fonts_referencia, fonts_generades)

        #resultat["classificacio_recall_micro"] = recall_score(fonts_referencia, fonts_generades, average="micro", labels=tokens_fonts)
        #resultat["classificacio_recall_weighted"] = recall_score(fonts_referencia, fonts_generades, average="weighted", labels=tokens_fonts)

        total_mostres = len(resum_generat)
        total_format_correcte = 0
        for pred in resum_generat:
                if pred[0] == 1 and pred[1] >= tokens_fonts[0] and pred[1] <= tokens_fonts[-1] and pred[2] == 60001:
                    total_format_correcte += 1
        resultat["classificacio_format_correcte"] = total_format_correcte / total_mostres
    return resultat

#Ara cal que creem el Trainer, que és el que s'encarregarà d'entrenar el model
num_train_epochs = 6
#aço indica cada quantes vegades s'actualitza la informació en wandb per això no interessa que siga cada molt temps
logging_steps = 1000
#preparem els arguments que li passaerem al nostre Trainer: hiperparametres i algunes altres caracteristiques 
args = Seq2SeqTrainingArguments(
    output_dir="NAS" + llengua + "-finetuned-de-zero-amb-metriques-anonim",
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

#############################################################
### Ara definim la busqueda d'hiperparametres amb Optuna ###
#############################################################

#Primer que res definim l'espai de busqueda:
"""def optuna_hp_space(trial):
    return {
        "learning_rate" : trial.suggest_float("learning_rate", low=1e-6, high=1e-4, log = True),
        "weight_decay" : trial.suggest_float("weight_decay", low=1e-4, high=0.01, log = True),
    }

#definim una funció per a tornar la inicialització del model
def model_init(trial):
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    #Important que reajustem la mida del token_embeddings perque coincideixi amb el nou tokenitzador (amb els nous tokens especials)
    model.resize_token_embeddings(len(tokenizer))
    model.to (device)
    return model

#el definim un poc diferent per a poder fer servir optuna
trainer = Seq2SeqTrainer(
    model=None,
    args=args,
    data_collator=data_collator,
    train_dataset=ds_combinat_tokenized["train"],
    eval_dataset=ds_combinat_tokenized["validation"],
    tokenizer=tokenizer,
    compute_metrics=calcular_metriques,
    model_init = model_init,
)   

#I amb açò en principi ja s'entrena quan haja acabat de buscar els millors paràmetres
best_trial = trainer.hyperparameter_search(
    direction="maximize", #en principi volem maximitzar perquè valora el ROUGE i F1 i tal
    hp_space=optuna_hp_space,
    n_trials=20,
    backend="optuna",
) """


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

#guardem la versió definitiva
#trainer.save_model("NAS" + llengua + "-finetuned-de-zero-amb-metriques-anonim")
#avaluem el resultat (en cada epoch hauriem de veure el training loss decerixer i el ROUGE pujar)

trainer.push_to_hub(commit_message="Training complete", tags="summarization")
wandb.finish()
