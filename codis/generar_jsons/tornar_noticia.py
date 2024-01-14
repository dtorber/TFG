"""
    script que rep per linia de comands si vol el fitxer test_i o test_ni (per defecte test_i) i l'id de la notícia i la torna, siga per imprimir-la o gastar-la
"""
import json
import sys
import random

def tornar_noticia (ruta = "./jsons/test_i.json", id= None, complet = False):
    resultat = None
    if id is not None:
        #obrim els arxius .json i si la clau existeix tornem la notícia associada
        with open(ruta, "r") as f:
            json_data = json.load(f)
            if id in json_data:
                resultat = json_data[id]
    else:
        #obrim els arxius .json i si la clau existeix tornem la notícia associada
        with open(ruta, "r") as f:
            json_data = json.load(f)
            resultat = json_data[json_data.keys()[0]]
    #si me diuen que volen el complet vol dir que volem també el resum forçat, sense forçar, etc.
    subset = "test_i" if ruta.__contains__("test_i") else "test_ni"
    tokens_fonts = list(range(60002, 60007 + 1))
    tokens_to_src = {
        60002: "CA01",
        60003: "CA02",
        60004: "CA03",
        60005: "CA04",
        60006: "CA05",
        60007: "CA06"
    }
    if complet:
        for token in tokens_fonts:
            with open(f"./resums_generats/resums_{subset}_forçat_{token}.json", "r") as f:
                json_data = json.load(f)
                resultat[tokens_to_src[token]] = json_data[id]["summary"]
        with open(f"./resums_generats/resums_{subset}_original.json", "r") as f:
            json_data = json.load(f)
            resultat["sense_forçar"] = json_data[id]["summary"]
        with open(f"./resums_generats/resums_{subset}_font_correcta.json", "r") as f:
            json_data = json.load(f)
            resultat["font_correcta"] = json_data[id]["summary"]
        
    return resultat

#si volem tornar les primeres n noticies
def tornar_n_noticies (ruta = "./jsons/test_i.json", n = 10):
    noticies = []
    with open(ruta, "r") as f:
        json_data = json.load(f)
        for _, noticia in json_data.items():
            noticies.append(noticia)
            if len(noticies) >= n:
                break
    return noticies

#tornar n noticies en un ordre aleatori
def tornar_n_noticies_aleatories (ruta = "./jsons/test_i.json", n = 10):
    noticies = []
    with open(ruta, "r") as f:
        json_data = json.load(f)
        #shuffle the keys of the json_data
        keys = list(json_data.keys())
        random.shuffle(keys)
        for id in keys:
            noticia = json_data[id]
            noticies.append(noticia)
            if len(noticies) >= n:
                break
    return noticies

#només si el programa és el principal s'ha llançat des de la terminal se lligen els paràmetres per teclat
if __name__ == "__main__":
    id = sys.argv[1].strip() if len(sys.argv) > 1 else None
    ruta = sys.argv[2].strip() if len(sys.argv) > 2 and sys.argv[2].strip() != "--verbose" else "./jsons/test_i.json"
    verbose = False
    #afegim una opció --verbose perquè traga tots els resums en les diferents combinacions per poder comparar-los
    if (len(sys.argv) > 2 and sys.argv[2] == "--verbose") or (len(sys.argv) > 3 and sys.argv[3] == "--verbose"):
        verbose = True
    res = tornar_noticia(ruta, id, complet = verbose)
    print(res)
    for key, val in res.items():
        print(f"{key}: {val}")