import json
import gzip
from tqdm import tqdm


def convert_json_to_jsonl (subcarpeta : str, particio : str) -> None:
    print(f"Converting {particio} json to jsonl...")
    #unzip del corpus anonimitzat
    with gzip.open(f"../corpus_anonimitzat/{subcarpeta}/{particio}.json.gz", "rb") as f_in:
        bytes = f_in.read()
        json_str = bytes.decode('utf-8')
        json_dict = json.loads(json_str)
        with gzip.open(f"../corpus_anonimitzat/{subcarpeta}-jsonl/{particio}.json.gz", "wb") as f_out:
            #make a correct tqdm loop
            for i in tqdm(range(len(json_dict))):
                jout = json.dumps(json_dict[i]) + "\n"
                jout = jout.encode('utf-8')
                f_out.write(jout)

if __name__ == "__main__":
    carpeta_src = "ca-es2"
    convert_json_to_jsonl(carpeta_src, "train")
    convert_json_to_jsonl(carpeta_src, "validation")
    convert_json_to_jsonl(carpeta_src, "test-i")
    convert_json_to_jsonl(carpeta_src, "test-ni")