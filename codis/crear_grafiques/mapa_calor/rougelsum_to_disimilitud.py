import os
def invertir_valors(ruta):
    with open(ruta, "r") as f:
        data = f.read().splitlines()
        matrix = list()
        for line in data:
            matrix.append(list())
            for value in line.split():
                matrix[-1].append(100 - float(value))
    with open(ruta, "w") as f:
        for line in matrix:
            for value in line:
                f.write(f"{value:.2f} ")
            f.write("\n")

if __name__ == "__main__":
    invertir_valors(os.getcwd() + "/M2/dades/matriu_disimilitud_M2_resums_llargs.txt")
    invertir_valors(os.getcwd() + "/M2/dades/matriu_disimilitud_M2.txt")