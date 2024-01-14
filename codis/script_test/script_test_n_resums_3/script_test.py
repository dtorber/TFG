"""
    Script per a testar el model de resum de text mitjançant ROUGE, BERTscore i les mètriques de classificació
"""
import datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
import evaluate
import numpy as np
import torch
import nltk
from tqdm import tqdm
from sys import stdout
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, recall_score, confusion_matrix

def main(checkpoint= "NASca-finetuned-de-zero-amb-metriques-anonim", eixida = "resultats_script_test.out", carregar_de_hugginface = False, carregar_de_cache = True, llengua = "ca"):
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
        Funcio per avaluar els resultats del model: ROUGE, BERTscore i la tasca de classificació si escau
        @param model: model a avaluar
        @param test_dataset: dataset amb les dades de test
        @param avaluar_classif: si és True, avaluem també la classificació
        @return: diccionari amb els distints resultats de l'avaluació i la matriu de confusió si avaluar_classif era True
    """
    def evaluate_model(model, test_dataset, avaluar_classif = False, nom_test="test"):
        model.eval()
        fonts_generades = []
        fonts_referencia = []
        matriu_confusio = None #deixem la matriu de confusio a None per si no volem avaluar la classificacio
        if avaluar_classif:
            total_mostres = len(test_dataset)
        #     total_encerts = 0
            total_format_correcte = 0
        #     tp = {}
        #     fp = {}
        #     fn = {}
            matriu_confusio = dict() #gatarem un diccionari per a la matriu de confusio per mantindre els tokens com a claus
            for font in tokens_fonts: #creem una entrada per cada font, però en format tokenitzat així només cal destokenitzar al final
        #         tp[font] = 0
        #         fp[font] = 0
        #         fn[font] = 0
                matriu_confusio[font] = dict()
        for batch in tqdm(DataLoader(test_dataset, collate_fn = data_collator, batch_size=8, shuffle=True), desc="Avaluant el model"):
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids = batch["input_ids"]
            resum_referencia = batch["labels"]
        
            summary_ids = model.generate(
                input_ids, num_beams=4, min_length=25, max_length=150,
                early_stopping=True, no_repeat_ngram_size=3, repetition_penalty=2.0,
                #decoder_start_token_id=1, #forcem a que comence amb el token 1 i així no dona problemes lo del 2,1,1,1, però en realitat sí que dona perquè alguns són 1,1,1
                #forced_bos_token_id = 20000, #forcem a que comence com si fora de la font CA06
            )
            
            predictions = ([
                tokenizer.decode(
                    g.tolist(), skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                ) for g in summary_ids
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

            predictions = ["\n".join(nltk.sent_tokenize(pred.strip(), language="spanish")) for pred in predictions]
            referencies = ["\n".join(nltk.sent_tokenize(label.strip(), language="spanish")) for label in referencies]

            metric.add_batch(predictions=predictions, references=referencies)
            bert.add_batch(predictions=predictions, references=referencies)
            #en esta part d'aci hauriem de computar el accuracy, recall, precision, f1, etc.
            #els labels són FONT <text>
            #els predictions són <s> FONT <text>
            if avaluar_classif:
                for pred, ref in zip(summary_ids.tolist(), resum_referencia):
                    if pred[0] == 1 and pred[1] >= tokens_fonts[0] and pred[1] <= tokens_fonts[-1] and pred[2] == 60001:
                        total_format_correcte += 1
                        fonts_generades.append(pred[1])
                        fonts_referencia.append(ref[0])
                        # if pred[1] == ref[0]:
                        #     total_encerts += 1
                        #     tp[pred[1]] += 1
                        # else:
                        #     fp[pred[1]] += 1
                        #     fn[ref[0]] += 1
                        # #el valor real tria la fila i la prediccio la columna i sumem 1
                        # matriu_confusio[ref[0]][pred[1]] = matriu_confusio[ref[0]].get(pred[1], 0) + 1
        resultat = metric.compute()
        #si volem fer servir este script per a NASes haurem de canviar el lang
        bert_score = bert.compute(lang=llengua, batch_size = 8, use_fast_tokenizer=True)
        for k, v in bert_score.items():
            if k in ["precision", "recall", "f1"]:
                resultat["bert_" + k] = np.mean(np.array(v))
        if avaluar_classif:
            #ara gastem scikit per tal de generar les mètriques
            precision, recall, f1, _ = precision_recall_fscore_support(fonts_referencia, fonts_generades, average="macro", labels=tokens_fonts)
            resultat["classificacio_precision"] = np.mean(precision)
            resultat["classificacio_recall"] = np.mean(recall)
            resultat["classificacio_f1"] = np.mean(f1)
            resultat["classicacio_accuracy"] = accuracy_score(fonts_referencia, fonts_generades)
            confus = confusion_matrix(fonts_referencia, fonts_generades, labels=tokens_fonts, normalize = "all")
            #com no volem només una matriu de resultats sino que volem un diccionari que tinga les informacions de les fonts
            #ho postprocessem per recorrer tota la informació
            for i, token1 in enumerate(tokens_fonts):
                for j, token2 in enumerate(tokens_fonts):
                    matriu_confusio[token1][token2] = round(confus[i][j], 4)
            #resultat["classificacio_recall_micro"] = recall_score(fonts_referencia, fonts_generades, average="micro", labels=tokens_fonts)
            #resultat["classificacio_recall_weighted"] = recall_score(fonts_referencia, fonts_generades, average="weighted", labels=tokens_fonts)


            resultat["classificacio_format_correcte"] = total_format_correcte / len(test_dataset)
            # resultat["classificacio_recall"] = []
            # resultat["classificacio_precision"] = []
            # resultat["classificacio_f1"] = []
            # resultat["classificacio_accuracy"] = total_encerts / total_mostres
            # for i in tokens_fonts:
            #     if tp[i] + fn[i] > 0:
            #         resultat["classificacio_recall"].append(tp[i] / (tp[i] + fn[i]))
            #     else:
            #         resultat["classificacio_recall"].append(0)
            #     if tp[i] + fp[i] > 0:
            #         resultat["classificacio_precision"].append(tp[i] / (tp[i] + fp[i]))
            #     else:
            #         resultat["classificacio_precision"].append(0)
            #     if 2 * tp[i] + fp[i] + fn[i] > 0:
            #         resultat["classificacio_f1"].append(2 * tp[i] / (2 * tp[i] + fp[i] + fn[i]))
            #     else:
            #         resultat["classificacio_f1"].append(0)
            #     #ara faltava que normalitzarem la matriu de confussio
            #     for j in tokens_fonts:
            #         matriu_confusio[i][j] = round(matriu_confusio[i].get(j, 0) / total_mostres, 4)
            
            # resultat["classificacio_recall"] = np.mean(np.array(resultat["classificacio_recall"]))
            # resultat["classificacio_precision"] = np.mean(np.array(resultat["classificacio_precision"]))
            # resultat["classificacio_f1"] = np.mean(np.array(resultat["classificacio_f1"]))
        return resultat, matriu_confusio

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
    f.write("Test del model, sense forçar el símbol de font\n")

    res_i, matriu_i = evaluate_model(model, tokenized_test_i, avaluar_classif = True)
    escriure_resultats(res_i, matriu_i, f, "test_i")
    f.close()

    f = open(eixida, "a")

    if llengua == "ca":
        #afegim els tokens de les nvoes fonts perquè s'actualitze abans de testejar ni, perquè ho necessitem per la matriu de confusió
        tokens_fonts += [60008, 60009, 60010]
    else:
        #si es el cas de castellà faltarà afegir-li els tokens que corresponen a les fonts de _ni
        tokens_fonts += []
    res_ni, matriu_ni = evaluate_model(model, tokenized_test_ni, avaluar_classif = True)
    escriure_resultats(res_ni, matriu_ni, f, "test_ni")

    #el tanquem i tornem a obrir per si de cas peta enmig d'una execucio, que se puga aprofitar l'altra
    f.close()
if __name__ == "__main__":
    main()