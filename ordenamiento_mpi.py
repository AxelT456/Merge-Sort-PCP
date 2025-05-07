from mpi4py import MPI
import pandas as pd
import numpy as np
import time
import os

# Configuración de MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def merge(left, right):
    result = []
    while left and right:
        # Comparamos por el valor a ordenar (el segundo elemento del par)
        if left[0][1] <= right[0][1]:
            result.append(left.pop(0))
        else:
            result.append(right.pop(0))
    # Añadimos los elementos restantes
    result.extend(left)
    result.extend(right)
    return result

def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

# Maestro
if rank == 0:
    # Cargar el dataset
    dataset_path = "subset.csv"
    if not os.path.exists(dataset_path):
        print(f"El archivo '{dataset_path}' no existe.")
        exit()

    df = pd.read_csv(dataset_path)

    # Mostrar columnas disponibles para ordenamiento
    print("[Maestro] Dataset cargado con", len(df), "registros.")
    print("[Maestro] Columnas disponibles:", list(df.columns))

    # Sugerir cantidad de procesos
    registros = len(df)
    if registros < 1000:
        sugeridos = 2
    elif registros < 10000:
        sugeridos = 4
    elif registros < 100000:
        sugeridos = 8
    else:
        sugeridos = 16
    print(f"[SUGERENCIA] Para {registros} registros, se recomienda usar {sugeridos} procesos.")

    # Solicitar columna y orden de ordenamiento
    columna = input("Seleccione una columna para ordenar: ")
    orden = input("Seleccione el orden (ascendente/descendente): ").strip().lower()
    if orden not in ["ascendente", "descendente"]:
        print("Orden no válido, usando 'ascendente' por defecto.")
        orden = "ascendente"

    # Convertir a lista de tuplas (índice, valor) para ordenamiento
    datos = list(df[columna].items())

    # Dividir los datos en partes para cada proceso
    partes = np.array_split(datos, size)
    start_time = time.time()

else:
    partes = None
    columna = None
    orden = None

# Compartir columna y orden con todos los procesos
columna = comm.bcast(columna, root=0)
orden = comm.bcast(orden, root=0)

# Distribuir partes del dataset
parte = comm.scatter(partes, root=0)

# Ordenar localmente
inicio_local = time.time()
parte_ordenada = merge_sort(list(parte))
tiempo_local = time.time() - inicio_local
print(f"[Proceso {rank}] Tiempo de ordenamiento local: {tiempo_local:.4f} segundos.")

# Recolectar partes ordenadas
partes_ordenadas = comm.gather(parte_ordenada, root=0)

# Fusionar los resultados en el maestro
if rank == 0:
    # Fusionar todas las partes ordenadas
    resultado_final = []
    for parte in partes_ordenadas:
        resultado_final = merge(resultado_final, parte)

    # Invertir el resultado si el orden es descendente
    if orden == "descendente":
        resultado_final.reverse()

    # Extraer los valores para crear el dataframe final
    df_ordenado = pd.DataFrame({columna: [x[1] for x in resultado_final]})

    # Guardar en archivo CSV
    output_file = "dataset_ordenado.csv"
    df_ordenado.to_csv(output_file, index=False)
    print(f"[Maestro] Ordenamiento global completado en {time.time() - start_time:.4f} segundos.")
    print(f"[Maestro] Archivo ordenado guardado como '{output_file}'.")
