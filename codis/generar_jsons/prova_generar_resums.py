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

    test_i = datasets.load_dataset("json", data_files="../corpus_anonimitzat/" + llengua +"/test-i.json.gz")
    test_i = test_i["train"]
    
    nou_test_i = {
        "train": dict()
    }
    categories = ["article", "summary", "id", "source"]

    for categoria in categories:
        nou_test_i["train"][categoria] = list() 
    # aux = [
    #     {'article': 'El nobre de persones que moren soles a casa ha crescut durant el primer mes de confinament un 33% respecte al mateix període de l\'any passat. En concret, des del 13 de març fins al 14 d\'abril els Bombers han trobat 42 persones mortes al entrar als respectius domicilis, segons ha informat l\'ACN. En canvi, en les mateixes dates de l\'any passat el nombre de defuncions va ser de 28, tot i que el de serveis que va fer el cos va ser més alt. La majoria d\'aquests casos són de persones grans que vivien soles a casa. Cada setmana moren més de dues persones grans soles a casa sev. Des de la Creu Roja alerten que el confinament afecta especialment aquest col·lectiu, i per això demanen que l\'entorn dels avis no perdi el contacte amb ells, tot i les restriccions derivades de l\'estat d\'alarma. De fet, des dels ajuntaments s\'ha incrementat l\'atenció telefònica a persones grans que viuen soles. Durant el primer mes de confinament, els Bombers han fet un total de 522 entrades en habitatges, alertats per veïns o serveis socials o d\'emergència. Això són 27 menys que en el mateix període de l\'any passat. En canvi, el nombre de persones mortes que s\'han trobat a dins ha crescut. De les 28 del 2019 a les 42 d\'aquest any. En la majoria de casos que el cos ha hagut d\'anar a obrir un domicili s\'hi han trobat una persona gran que viu sola. El problema amb el confinament és que aquests avis han perdut la xarxa social que tenien abans, un fet que preocupa als ajuntaments. Per això, la majoria de consistoris han començat a prendre mesures en aquest aspecte. A tall d\'exemple, a Girona l\'Ajuntament ha fet una llsita de les persones més grans de 65 anys que figuren al padró i ha fet una ronda de trucades per saber les seves necessitats, ja que les que fan ús dels Serveis Socials són "una part limitada". La regidora del ram, Núria Pi, explica a l\'ACN que la feina es va fer amb voluntaris i que, gràcies a això, van poder "desgranar" les necessitats que calia cobrir. "N\'hi ha que tenen família i una xarxa social activa. Aquestes persones no han necessitat l\'assistència. En canvi, d\'altres que també tenen família els ha afectat el fet d\'estar moltes hores soles tancades a casa i els hem ofert suport psicològic. També n\'hi ha d\'altres que necessiten ajuda en necessitats bàsiques com baixar les escombraries o fer la compra", explica Pi. L\'inconvenient més important que s\'han trobat, però, ha sigut en les persones que estaven contagiades per la covid-19 i es poden estar a casa perquè no requereixen anar a l\'hospital. Pi assenyala que a l\'inici "va costar més" treballar amb aquests avis, ja que no hi havia equips per donar el servei. "Quan ens trobàvem amb persones contagiades es feia molt difícil oferir una atenció a domicili, perquè no hi havia el material adequat i costava molt trobar-ne. En canvi, en els avis que viuen sols però que no tenen coronavirus ha sigut molt més àgil", assenyala la regidora. Des de la Creu Roja expliquen que el confinament genera "un major aïllament" en persones que estaven acostumades a tenir una xarxa social, encara que visquin soles. En aquest sentit, la tècnica de l\'entitat del tercer sector, Verònica Hortas, explica a l\'ACN que caldrà fer un treball posterior perquè aquests avis tornin a tenir la iniciativa de sortir al carrer. "Conforme s\'ha anat allargant el confinament i han aparegut més dades de persones que han mort ens hem trobat que ha crescut l\'ansietat, avis que han perdut la gana i que se senten més vulnerables pel fet d\'estar sols", diu Hortas. La tècnica de la Creu Roja explica que moltes situacions quotidianes, com visites de familiars o trobades amb els veïns que ara no es produeixen, són un punt de suport i de cert control. Per això, la tècnica de la Creu Roja demana que "no es perdi el contacte" i la xarxa amb les persones grans que viuen soles durant el confinament. "Hem de tenir en compte que tornar a la normalitat per a aquestes persones, que seran les últimes en sortir, serà un procés molt influenciat pel que han viscut durant aquest període", conclou Horta.', 'summary': "Entre el 13 de març i el 14 d'abril s'han trobat 42 avis i àvies morts, un 33% més que l'any passat.", 'source': 'CA01', 'id': 'a9d2cd679ba4fa46d8cd59dd868446cdf3d0e7d346f48276a3c3b91e7ec791a6'}
    #     {'article': 'La crisi de la covid-19 va provocar la destrucció de 63.700 llocs de treball durant la segona quinzena de març a Catalunya, amb una pèrdua de 7.000 feines al dia entre l\'11 i el 30 de març, segons estimacions del Govern a partir de les dades d\'afiliació a la Seguretat Social. El secretari general de Treball, Josep Ginesta, ha assegurat que, en els primers tres mesos del 2020, s\'han perdut 118.500 feines. Unes dades que, segons diu, reflecteixen de manera "més real" l\'afectació econòmica de la crisi de la covid-19 que l\'Enquesta de Població Activa (EPA), un indicador "històric" que no té capacitat de recollir la "celeritat" i la severitat" de la paràlisi econòmica que ha provocat la pandèmia. Ginesta ha fet aquestes declaracions el mateix dia en què s\'han conegut les dades trimestrals de l\'EPA que mostren que la taxa d\'atur ha pujat fins al 10,66% entre gener i març. Segons aquesta enquesta elaborada per l\'Institut Nacional d\'Estadística, la xifra de desocupats ha augmentat en 5.800 persones, fins a les 411.600, en comparació als darrers tres primeres mesos del 2019. Aquesta enquesta s\'elabora seguint les recomanacions de l\'OIT i Eurostat i no s\'han modificat els criteris de valoració a les circumstàncies de la covid-19, ha dit. En concret, ha assegurat que l\'EPA no recull les afectacions que han patit treballadors inclosos en un Expedient de Regulació Temporal de l\'Ocupació o els fixos discontinus, per exemple. El secretari de Treball ha remarcat que els efectes de la pandèmia sobre el mercat de treball "no es poden comparar" amb altres crisis com la del 2008, ja que hi ha centenars de milers de treballadors afectats pel decret d\'estat d\'alarma i el confinament general de la població, però totes les mesures "han de ser temporals i transitòries". "La temporalitat de les mesures legislatives i econòmiques estan afavorint decisions conjunturals que no seran definitives. Tot això fa que hi hagi un mercat de treball desdibuixat. Estem en una bombolla", ha explicat. Ginesta no s\'ha atrevit a fer previsions concretes de com pot evolucionar l\'atur a Catalunya però ha assegurat que l\'afectació al mercat de treball serà "molt important" si es té en compte que les previsions "més optimistes" auguren una caiguda del Producte Interior Brut (PIB) català del 5% a finals d\'any. Ampliar mesures excepcionals Per a pal·liar la crisi econòmica un cop aixecat l\'estat d\'alarma, el secretari de Treball considera necessari ampliar les mesures excepcionals pels expedients temporals, com ara l\'exoneració de les cotitzacions a la Seguretat Social que l\'empresa ha de pagar per treballador. "Moltes empreses no en tindran prou amb l\'aixecament de l\'estat d\'alarma o amb l\'inici de l\'activitat, necessitaran mesures progressives. Ens interessa que el motor del mercat de treball, les empreses, superin la crisi", ha dit. Pel que fa a les dades de l\'enquesta, ha destacat que la població ocupada que s\'ha incrementat en 62.000 persones en un any, la xifra més alta des del primer trimestre del 2008, i que situa a Catalunya com la comunitat autònoma amb més ocupats. Pel que fa als aturats, també ha remarcat la comparació anual, tot valorant que la taxa d\'atur, del 10,6% és un punt inferior a la de l\'any passat i es manté per sota de l\'11%. Entre les dades més preocupants que reflecteix l\'estudi elaborat per l\'INE, ha destacat el creixement de llars amb tots els actius aturats, que ha sigut de 14.400 persones en relació amb el trimestre anterior. També ha lamentat el descens de contractació al llarg del mes de març, amb una davallada del 40% de la contractació temporal. Pel que fa als Expedients de Regulació Temporal de l\'Ocupació (ERTO) que tramita el Departament de Treball des de l\'inici del confinament, Ginesta ha assegurat que ha resolt el 98,2% dels procediments. En el 92,3% dels casos s\'ha constatat que existeix una causa de força major, mentre que un 2,7% d\'expedients que al·leguen força major s\'han rebutjat i un 4,9% s\'han aprovat per silenci administratiu, és a dir, han quedat aprovats després de cinc dies hàbils. Preguntat sobre les irregularitats detectades en els ERTO, ha assegurat que la revisió d\'aquests expedients anirà a càrrec de la Inspecció de Treball estatal.', 'summary': "Treball creu que l'EPA no reflecteix la 'celeritat' i 'severitat' de les conseqüències econòmiques de la covid-19.", 'source': 'CA03', 'id': 'c626e584bfb4dc8bf1c37825772dee2c17c0ca059ba9750a9e0890c7c018bb0c'}
    # ]

    ids = set(["c626e584bfb4dc8bf1c37825772dee2c17c0ca059ba9750a9e0890c7c018bb0c", "a9d2cd679ba4fa46d8cd59dd868446cdf3d0e7d346f48276a3c3b91e7ec791a6"])

    for exemple in test_i:
        if exemple["id"] in ids:
            print(exemple)
            nou_test_i["train"]["article"].append(exemple["article"])
            nou_test_i["train"]["summary"].append(exemple["summary"])
            nou_test_i["train"]["id"].append(exemple["id"])
            nou_test_i["train"]["source"].append(exemple["source"])

    test_i = datasets.Dataset.from_dict(nou_test_i["train"])

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


    tokenized_test_i = test_i_combinat.map(preproces_dades_test, load_from_cache_file=carregar_de_cache, batched=True, remove_columns=test_i_combinat.column_names)

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
    def generate_summarys(model, test_dataset, test_original, ruta = f"./resums_generats/{datetime.now()}.json", forced_token = None, subset = "test_i", min_long=25, max_long=150):        
        i = 0
        json_data = dict()
        for batch, batch_original in zip(tqdm(DataLoader(test_dataset, collate_fn = data_collator, batch_size=8, shuffle=False), desc="Generant resums"), DataLoader(test_original, batch_size=8, shuffle=False)):
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids = batch["input_ids"]

            summary_ids = model.generate(
                input_ids, num_beams=4, min_length=min_long, max_length=max_long,
                early_stopping=True, no_repeat_ngram_size=3, repetition_penalty=2.0,
                #decoder_start_token_id=1, #forcem a que comence amb el token 1 i així no dona problemes lo del 2,1,1,1, però en realitat sí que dona perquè alguns són 1,1,1
                forced_bos_token_id = forced_token if forced_token is not None else model.config.forced_bos_token_id, #forcem a que comence com si fora de la font CA06
            )
            
            predictions = ([
                tokenizer.decode(
                    g.tolist(), skip_special_tokens=False,
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

    #generem els resums per a les notícies amb informació de font
    print(generate_summarys(model, tokenized_test_i, test_i, ruta = f"{carpeta}/prova_resums_presentacio.json", subset = "test_i", ))
    print(generate_summarys(model, tokenized_test_i, test_i, ruta = f"{carpeta}/prova_resums_presentacio_llargs.json", subset = "test_i", min_long=70, max_long=420))


if __name__ == "__main__":
    main(carpeta = "./resums_generats_primer_model")