import json
import datasets

# fonts = dict()
# test_i = datasets.load_dataset("json", data_files="../corpus_anonimitzat/ca/test-i.json.gz")

# for noticia in test_i["train"]:
#     fonts[noticia["source"]] = fonts.get(noticia["source"], 0) + 1
# print(fonts)

with open("./jsons/test_i.json", "r") as f:
    test_i = json.load(f)

    with open("./resums_generats/prova_resums_test_i_original.json", "r") as f2:
        resums = json.load(f2)
        for id, noticia in test_i.items():
            if resums[id]["source"] != noticia["source"]:
                print("ERROR")
                print(noticia)
                print(resums[id])
                input()
