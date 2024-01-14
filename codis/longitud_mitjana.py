import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

checkpoint = "dtorber/NASca-finetuned-de-zero-amb-metriques-anonim"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)

mostres = []
total = 0
longitud_mitjana = 0


with open("./jsons/test_i.json", "r") as f:
    test_i = json.load(f)
    for _, val in test_i.items():
        mostres.append(val["summary"])

tokenitzat = tokenizer(mostres)["input_ids"]

total = sum([len(x) for x in tokenitzat])

longitud_mitjana = total / len(mostres)

print("test_i")
print(longitud_mitjana)


mostres = []
longitud_mitjana = 0

with open("./jsons/test_ni.json", "r") as f:
    test_ni = json.load(f)
    for _, val in test_ni.items():
        mostres.append(val["summary"])

tokenitzat = tokenizer(mostres)["input_ids"]
aux = sum([len(x) for x in tokenitzat])
total += aux
longitud_mitjana = aux / len(mostres)

print("test_ni")
print(longitud_mitjana)

print("Mitjana global:")
total = total / (len(test_i) + len(test_ni))
print(total)