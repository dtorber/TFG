"""
    Aquest script val per a comprovar que la predicció que fa el model bilingüe de la llengua de l'article és correcta
"""
import json
llengues = ["ca", "es"]
resultats = dict()
for subset in ["test_i", "test_ni"]:
    resultats[subset] = dict()
    for llengua in llengues:
        with open(f"resums_generats_bilingue/resums_normals/resums_{subset}_original_{llengua}.json", "r") as f:
            item = json.load(f)
            if len(item) > 0:
                resultats[subset][llengua] = {
                    "accuracy_forcant_es": 0,
                    "accuracy_forcant_ca": 0,
                }
                for lang in llengues:
                    for k, v in item.items():
                        if v[f"pred_idioma_article_{lang}"].upper() == llengua.upper():
                            resultats[subset][llengua][f"accuracy_forcant_{lang}"] += 1
                    resultats[subset][llengua][f"accuracy_forcant_{lang}"] = resultats[subset][llengua][f"accuracy_forcant_{lang}"] * 100 / len(item)

with open("resultats_bilingue/resultats_accuracy.json", "w") as f:
    json.dump(resultats, f)
                