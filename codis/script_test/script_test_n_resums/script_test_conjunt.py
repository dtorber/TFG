import script_test_font_correcta_sense_generar
import script_test_sense_generar
import script_test_font_sense_generar
import script_test_fonts_ampliat_sense_generar
import script_test_fonts_ampliat_matrius_sense_generar
import script_test_sense_generar_detallat

#per no complicar-nos la vida simplement cridem a tots els scripts i així no interferim entre els tokens_fonts
#i de més que necessita cadascun
if __name__ == "__main__":
    #fem el test prototípic
    script_test_sense_generar.main(checkpoint= "NASca-finetuned-de-zero-amb-metriques-anonim", eixida = "../resultats_n_resums/resultats_script_test_de_zero.json")
    #se comprova si canvia algo quan li donem la font correcta
    script_test_font_correcta_sense_generar.main(checkpoint= "NASca-finetuned-de-zero-amb-metriques-anonim", eixida = "../resultats_n_resums/resultats_script_test_font_correcta_de_zero.json")
    #avaluem les mètriques forçant cadascuna de les fonts
    script_test_font_sense_generar.main(checkpoint= "NASca-finetuned-de-zero-amb-metriques-anonim")
    #se comprova si en algun cas quan forcem el token canvia el resum o no
    script_test_fonts_ampliat_sense_generar.main(checkpoint= "NASca-finetuned-de-zero-amb-metriques-anonim", eixida = "../resultats_n_resums/resultats_script_test_fonts_ampliat_de_zero.out") 
    #se comprova si en algun cas quan forcem el token canvia el resum o no i quant canvia
    script_test_fonts_ampliat_matrius_sense_generar.main(checkpoint= "NASca-finetuned-de-zero-amb-metriques-anonim", eixida = "../resultats_n_resums/resultats_script_test_fonts_ampliat_matrius_de_zero.out")
    script_test_sense_generar_detallat.main(checkpoint= "NASca-finetuned-de-zero-amb-metriques-anonim", eixida = "../resultats_n_resums/resultats_script_test_de_zero_detallats.json")
