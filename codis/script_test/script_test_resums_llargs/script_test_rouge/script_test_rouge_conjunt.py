import script_test_rouge_per_font
import script_test_rouge_forçat
from sys import stdout

def normalitzar (matriu, fila):
    nova_matriu = dict()
    for font_original in matriu.keys(): #la fila és la font original, per tant en comú amb fila
        nova_matriu[font_original] = dict()
        for col in matriu[font_original].keys(): #les columnes són els tokens que s'han forçat
            nova_matriu[font_original][col] = matriu[font_original][col] / fila[font_original]
    return nova_matriu

if __name__ == "__main__":
    #llacem l'execució dels dos scripts
    #el primer s'encarregarà de tornar la matriu ROUGE L-sum agrupades per font i forçant cadascuna de les notícies
    res = script_test_rouge_forçat.main(carregar_de_hugginface=True)
    matriu_forçada_i = res["test_i"]
    matriu_forçada_ni = res["test_ni"]
    #el segon s'encarrega de tornar el ROUGE L-sum per cada font sense haver forçat res
    res = script_test_rouge_per_font.main(carregar_de_hugginface=True)
    rouge_per_font_i = res["test_i"]
    rouge_per_font_ni = res["test_ni"]
    #i ara ens falta dividir la primera matriu per la segona per així saber si ha millorat o no, si el valor és menor que 1, el rendiment decau i si és major millora
    normalitzada_i = normalitzar(matriu_forçada_i, rouge_per_font_i)
    normalitzada_ni = normalitzar(matriu_forçada_ni, rouge_per_font_ni)

    #ara només ens queda imprimir les matrius per veure si ha millorat o no
    with open("./resultats_rouge/normalitzada_i.txt", "w") as f:
        f.write("Matriu normalitzada forçant les notícies dividit per sense forçar-les. Si el valor és menor que 1, el rendiment decau i si és major millora\n\n")
        stdout.write("Matriu normalitzada forçant les notícies dividit per sense forçar-les. Si el valor és menor que 1, el rendiment decau i si és major millora\n\n")
        script_test_rouge_forçat.escriure_matriu(normalitzada_i, f)
        script_test_rouge_forçat.escriure_matriu(normalitzada_i, stdout)

    with open("./resultats_rouge/normalitzada_ni.txt", "w") as f:
        f.write("Matriu normalitzada forçant les notícies dividit per sense forçar-les. Si el valor és menor que 1, el rendiment decau i si és major millora\n\n")
        stdout.write("Matriu normalitzada forçant les notícies dividit per sense forçar-les. Si el valor és menor que 1, el rendiment decau i si és major millora\n\n")
        script_test_rouge_forçat.escriure_matriu(normalitzada_ni, f)
        script_test_rouge_forçat.escriure_matriu(normalitzada_ni, stdout)