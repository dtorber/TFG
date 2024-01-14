import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

#els idx són perquè hi ha línies per dalt i per baix que no interessen
def preparar_dades(f, normalitzar = False, idx_inicial = 0, idx_final = None):
    # Leer el archivo y convertirlo en una lista
    data = f.read().splitlines()
    #eliminar dos primeras lineas
    data = data[idx_inicial:idx_final]
    if idx_final is not None:
        data = data[:idx_final]
    matrix = list()
    for line in data:
        if line.strip() == "": break #ha acabat la matriu i venen les estadistiques
        suma_fila = 0
        matrix.append(list())
        for value in line.split():
            #com pot ser que hi haja coses com CA01 que no siguen text fem un try-except
            try:
                matrix[-1].append(float(value))
                suma_fila += float(value)
            except ValueError:
                pass
        if normalitzar:
            #Normalitzar per fila
            for i in range(len(matrix[-1])):
                matrix[-1][i] = matrix[-1][i] / suma_fila * 100

    columnes = [f"CA0{i + 1}" for i in range(len(matrix[0]))]
    idxs = [f"CA0{i + 1}" for i in range(len(matrix))]
    # Convertir la matriz en un DataFrame de Pandas
    df = pd.DataFrame(
        matrix,
        columns=columnes,
        index=idxs,
    )
    return df

def generar_mapa_calor(data, file):
    # Crear el mapa de calor usando seaborn
    plt.figure(figsize=(8, 6))  # Tamaño de la figura
    sns.heatmap(
        data, annot=True, cmap="coolwarm", fmt=".2f"
    )  # Crear mapa de calor con anotaciones y formato decimal
    # plt.show()  # Mostrar el mapa de calor
    plt.savefig(f"grafiques/{file.split('.')[0]}sense_norm.png")  # Guardar el mapa de calor en un archivo (eliminant l'extensió)

def recorrer_directori(ruta):
    # abrir todos los archivos del directorio actual con extension .txt
    for file in os.listdir(ruta):
        if file.endswith(".txt") or (file.endswith(".out")):
            with open(os.path.join(ruta, file), "r") as f:
                df = preparar_dades(f, normalitzar=True, idx_inicial = 4, idx_final = -4)

                generar_mapa_calor(df, file)

if __name__ == "__main__":
    recorrer_directori(os.getcwd())
    recorrer_directori(os.getcwd() + "/dades/resums_llargs")