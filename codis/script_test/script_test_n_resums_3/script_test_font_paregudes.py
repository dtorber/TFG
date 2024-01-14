import datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
import evaluate
import numpy as np
import torch
import nltk
from tqdm import tqdm
import tensorflow as tf
from sys import stdout

def main(checkpoint= "NASca-finetuned-de-zero-amb-metriques-anonim", eixida = "resultats_script_test_fonts_paregudes.out", carregar_de_hugginface = False, carregar_de_cache = True, llengua = "ca"):
    nltk.download('punkt')
    #carregar_de_cache = True
    #SI CANVIEM LA LLENGUA CANVIAR ELS TOKENS FONTS QUE FALTEN ELS DE CASTELLÀ
    #llengua = "ca"
    tokens_fonts = list(range(60002, 60010 + 1)) if llengua == "ca" else None
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
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    # Els nous tokens especials seran tant el SEP_TOKEN com les claus de les
    # fonts per a l'idioma en concret, pero en principi ja estan en el tokenizer que hem descarregat
    #tokenizer.add_tokens([SEP_TOKEN] + SOURCES["ca"], special_tokens=True)

    #carreguem els datasets per a testejar:
    test_i = datasets.load_dataset("json", data_files="../corpus_anonimitzat/" + llengua +"/test-i.json.gz")

    #reduim els tamanys dels datasets per fer simplement proves:
    test_i = test_i["train"].shuffle(seed=42).select(range(10))

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

    #per eliminar les columnes com ara tenim un primer nivell d'indexació que són les fonts, les columnes realment són les columnes de cada font aleshores agarrem les d'una qualsevol
    tokenized_test_i = test_i_combinat.map(preproces_dades_test, load_from_cache_file=carregar_de_cache, batched=True, remove_columns=test_i_combinat[str(tokens_fonts[0])].column_names)

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
        Funcio per avaluar els resultats del model: ROUGE, BERTscore i la tasca de classificació si escau
        @param model: model a avaluar
        @param test_dataset: dataset amb les dades de test
        @param avaluar_classif: si és True, avaluem també la classificació
        @return: diccionari amb els distints resultats de l'avaluació i la matriu de confusió si avaluar_classif era True
    """
    def evaluate_model(model, test_dataset):
        model.eval()
        for token_font, token_forçat in [(60005, 60002), (60005, 60004), (60004, 60002)]:
            #per si de cas hi ha alguna partició que esta buida, eixe cas no l'avaluariem.
            if len(test_dataset[str(token_font)]) == 0:
                continue
            for batch in tqdm(DataLoader(test_dataset[str(token_font)], collate_fn = data_collator, batch_size=8, shuffle=True), desc="Avaluant el model"):
                batch = {k: v.to(device) for k, v in batch.items()}
                input_ids = batch["input_ids"]
                resum_referencia = batch["labels"]

                #A l'hora de generar el resum forcem a que comence amb el token de la font que hem de predir (1r token dels labels) 
                #necessitem descomposar-ho com un bucle perquè hem d'aplicar sobre cada input_id la seua font corresponent
                summary_ids_original = model.generate(
                    input_ids, num_beams=4, min_length=25, max_length=150,
                    early_stopping=True, no_repeat_ngram_size=3, repetition_penalty=2.0,
                    #decoder_start_token_id=1, #forcem a que comence amb el token 1 i així no dona problemes lo del 2,1,1,1, però en realitat sí que dona perquè alguns són 1,1,1
                    #forced_bos_token_id = token_forçat, #forcem a que comence com si fora de la font CA06
                )
                
                predictions_original = ([
                    tokenizer.decode(
                        g.tolist(), skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    ) for g in summary_ids_original
                ])
                
                summary_ids = model.generate(
                    input_ids, num_beams=4, min_length=25, max_length=150,
                    early_stopping=True, no_repeat_ngram_size=3, repetition_penalty=2.0,
                    #decoder_start_token_id=1, #forcem a que comence amb el token 1 i així no dona problemes lo del 2,1,1,1, però en realitat sí que dona perquè alguns són 1,1,1
                    forced_bos_token_id = token_forçat, #forcem a que comence com si fora de la font CA06
                )
                
                predictions = ([
                    tokenizer.decode(
                        g.tolist(), skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    ) for g in summary_ids
                ])

                """print("Resum referencia abans de llevar els -100")
                print(resum_referencia)
                input()"""

                resum_referencia = resum_referencia.cpu().numpy()
                resum_referencia = np.where(resum_referencia != -100, resum_referencia, tokenizer.pad_token_id)
                #Ací no faria falta també fer un padding_across_processes per tal de que tots els processos tinguin el mateix tamany independentment de quin procés els haja tokenitzat?
                referencies = ([
                    tokenizer.decode(
                        g.tolist(), skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    ) for g in resum_referencia
                ])
                
                for pred, pred_org, ref in zip(predictions, predictions_original, referencies):
                    print(f"Referencia: {ref}")
                    print(f"Prediccio forçada: {pred}")
                    print(f"Prediccio original: {pred_org}")
                    print("-" * 50)
                    input()
                print("=" * 50)

                predictions = ["\n".join(nltk.sent_tokenize(pred.strip(), language="spanish")) for pred in predictions]
                predictions_original = ["\n".join(nltk.sent_tokenize(pred.strip(), language="spanish")) for pred in predictions_original]

                #fora de que pugen ser diferents perquè evidentment les ha confós veiem com de diferents realment són
                metric.add_batch(predictions=predictions, references=predictions_original)
                bert.add_batch(predictions=predictions, references=predictions_original)
        resultat = metric.compute()
        #si volem fer servir este script per a NASes haurem de canviar el lang
        bert_score = bert.compute(lang=llengua, batch_size = 8, use_fast_tokenizer=True)
        for k, v in bert_score.items():
            if k in ["precision", "recall", "f1"]:
                resultat["bert_" + k] = np.mean(np.array(v))
    
        return resultat

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
    f.write("Test del model, forçant el símbol de font correcte\n")
    res_i = evaluate_model(model, tokenized_test_i, avaluar_classif = False)
    escriure_resultats(res_i, None, f, "test_i")
    f.close()

if __name__ == "__main__":
    main()