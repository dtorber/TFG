import gzip
import json
import datasets
from sys import stdout

equivalencia = {
    "CA01": "ES01",
    "CA02": "ES02",
    "CA03": "ES03",
    "CA04": "ES04",
    "CA05": "ES05",
    "CA06": "ES06",
    "CA07": "ES11",
    "CA08": "ES15",
    "CA09": "ES16",
}

with open("repartiment_noticies.txt", "w") as eixida:
    repartiment = dict()
    for particio in ["train", "validation", "test-i", "test-ni"]:
        repartiment[particio] = dict()
        for font_ca in equivalencia.keys():
            repartiment[particio][font_ca] = 0
            font_es = equivalencia[font_ca]
            repartiment[particio][font_es] = 0

        eixida.write(f"Partició: {particio.upper()}\n")
        stdout.write(f"Partició: {particio.upper()}\n")
        with gzip.open(f"../corpus_anonimitzat/ca-es/{particio}.json.gz", "rb") as f:
            for line in f:
                line = line.decode("utf-8")
                noticia = json.loads(line)
                repartiment[particio][noticia["source"]] += 1
    
    for font_ca in equivalencia.keys():
        font_es = equivalencia[font_ca]
        eixida.write(f"{font_ca} & {font_es} & ")
        stdout.write(f"{font_ca} & {font_es} & ")
        for particio in ["train", "validation", "test-i", "test-ni"]:
            if repartiment[particio][font_ca] == repartiment[particio][font_es]: 
                eixida.write(f"{repartiment[particio].get(font_ca, 0)} & ")
                stdout.write(f"{repartiment[particio].get(font_ca, 0)} & ")
            else: 
                eixida.write("ERROR & ")
                stdout.write("ERROR & ")
                
        eixida.write("\n")
        stdout.write("\n")
