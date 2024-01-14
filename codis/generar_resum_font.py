import datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
import evaluate
import numpy as np
import torch
import nltk

nltk.download('punkt')
carregar_de_cache = True

llengua = "ca"
tokens_fonts = list(range(60002, 60007 + 1)) if llengua == "ca" else None
carregar_de_hugginface = False

checkpoint = "NAS" + llengua + "-finetuned-diego-4-amb-metriques-anonim"
if carregar_de_hugginface:
    checkpoint = "dtorber/" + checkpoint

tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only = not carregar_de_hugginface, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
test_i = datasets.load_dataset("json", data_files="../corpus_anonimitzat/" + llengua +"/test-i.json.gz")["train"]

SEP_TOKEN = "<text>"

mostra = test_i.shuffle(seed=42).select(range(8))
mostra = mostra.map(lambda x: {
    "article": x['article'], #article en principi el deixem igual que estava
    "summary": f"{x['source']}{SEP_TOKEN}{x['summary']}",
}, load_from_cache_file=carregar_de_cache)

#veiem originalment com era
for noticia in mostra: 
    print("Article:")
    print(noticia["article"])
    print("Resum:")
    print(noticia["summary"])
    print("-"*100 + "\n\n")

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

tokenized_mostra = mostra.map(preproces_dades_test, load_from_cache_file=carregar_de_cache, batched=True, remove_columns=mostra.column_names)

label_relleno = -100
multiple = 8
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, 
                                        label_pad_token_id=label_relleno,
                                        padding = "max_length",
                                        max_length = max_article_length,
                                        pad_to_multiple_of=multiple,
                                    )

f = open("resums_generats.txt", "w")
#fem el batch_sizer per poder anar mostra per mostra analitzant-la
for batch in DataLoader(tokenized_mostra, batch_size=1, collate_fn=data_collator):
    input_ids = batch["input_ids"]
    input = []
    for token in input_ids.tolist()[0]:
        if token == -100 or token == tokenizer.pad_token_id:
            break
        else:
            input.append(token)
    
    print("Article:" + tokenizer.decode([input], skip_special_tokens=False, clean_up_tokenization_spaces=False))
    f.write("Article:" + tokenizer.decode([input], skip_special_tokens=False, clean_up_tokenization_spaces=False))
    f.write("\n")
    resum_referencia = batch["labels"]
    #només copiem allò que siga diferent de -100 (padding)
    ref = []
    for token in resum_referencia.tolist()[0]:
        if token == -100 or token == tokenizer.pad_token_id:
            break
        else:
            ref.append(token)
    print("Resum referencia:" + tokenizer.decode(ref, skip_special_tokens=False, clean_up_tokenization_spaces=False))
    f.write("Resum referencia:" + tokenizer.decode(ref, skip_special_tokens=False, clean_up_tokenization_spaces=False))
    f.write("\n")
    for token_font in [model.config.forced_bos_token_id] + tokens_fonts:
        print("Font: ", token_font if token_font != model.config.forced_bos_token_id else "sense dir-li res")
        f.write("Font: " + str(token_font if token_font != model.config.forced_bos_token_id else "sense dir-li res"))
        f.write("\n")

        summary_ids = model.generate(
            input_ids, num_beams=4, min_length=25, max_length=150,
            early_stopping=True, no_repeat_ngram_size=3, repetition_penalty=2.0,
            forced_bos_token_id = token_font, #forcem a que comence com si fora de la font CA06
        )
        print(summary_ids)
        f.write(str(summary_ids))
        f.write("\n")

        predictions = ([
                tokenizer.decode(
                    g.tolist(), skip_special_tokens=False,
                    clean_up_tokenization_spaces=False
                ) for g in summary_ids
        ])
        predictions = ["\n".join(nltk.sent_tokenize(pred.strip(), language="spanish")) for pred in predictions]

        print(predictions)
        f.write(str(predictions))
        f.write("\n")
        input()
f.close()