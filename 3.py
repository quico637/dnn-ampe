import argparse
import os

def inferenciaMultiple(pesos_path):
    # Cargar el subconjunto de validación del dataset MNIST
    valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)

    # Seleccionar aleatoriamente 10 imágenes del subconjunto de validación
    imagenesTest = random.sample(range(len(valset)), 10)

    # Para cada imagen seleccionada, llamar a la función 'inferencia' y guardar el resultado
    for i, indice in enumerate(imagenesTest):
        imagen, etiqueta = valset[indice]
        nombre_archivo = f"resultado_inferencia_{i}.txt"
        os.system(f"python3 inferencia.py -i {i} -f {pesos_path}")

        # Guardar el resultado en un archivo de texto
        with open(nombre_archivo, 'w') as f:
            f.write(f"Etiqueta: {etiqueta}\n")
            f.write(f"Predicción: {digit.item()}\n")

def main():          
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="weights file", required=True)
    args = parser.parse_args()

    if args.file:
        WEIGHTS_PATH = args.file

    WEIGHTS_PATH = "./weights/" + WEIGHTS_PATH

    inferenciaMultiple(WEIGHTS_PATH)

    
if __name__ == "__main__":
    main()