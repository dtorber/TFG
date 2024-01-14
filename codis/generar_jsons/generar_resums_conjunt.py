import generar_resums
import generar_n_resums
import generar_resums_llargs

if __name__ == "__main__":
    # generar_resums.main(checkpoint= "NASca-finetuned-diego-4-amb-metriques-anonim", carregar_de_hugginface = True, carregar_de_cache = True, llengua = "ca", carpeta = "./resums_generats_primer_model/")
    # generar_n_resums.main(checkpoint= "NASca-finetuned-diego-4-amb-metriques-anonim", carregar_de_hugginface = True, carregar_de_cache = True, llengua = "ca", carpeta = "./resums_generats_primer_model/")
    generar_resums_llargs.main(checkpoint= "NASca-finetuned-diego-4-amb-metriques-anonim", carregar_de_hugginface = True, carregar_de_cache = True, llengua = "ca", carpeta = "./resums_generats_primer_model/")