import json
import sys
import os

#obrir tots els fitxers json del directori actual .
#si no hi ha cap fitxer json, no es fa res
def obre_tots_jsons ():
    keys_originals = json.load(open("resums_test_i_original.json", "r")).keys()
    print(keys_originals)
    for file in os.listdir():
        print(file)
        if file.endswith(".json"):
            with open(file, "r") as f:
                json_data = json.load(f)
                print(json_data.keys() == keys_originals)
        
if __name__ == "__main__":
    obre_tots_jsons()
