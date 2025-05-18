import pandas as pd
import os
import psutil
from mpi4py import MPI
import time


def merge_sort_dataframe(df, column_name, ascending=True):
    if df.empty:
        print("El dataframe está vacío. Devolviendo dataframe original")
        return df
    if column_name not in df.columns:
        print(f"Columna '{column_name}' no encontrada en el dataframe. Devolviendo dataframe original")
        return df

    data_to_sort = list(df.iterrows())

    def merge_sort(data):
        if len(data) <= 1:
            return data

        mid = len(data) // 2
        left = merge_sort(data[:mid])
        right = merge_sort(data[mid:])

        merged = []
        left_index, right_index = 0, 0

        while left_index < len(left) and right_index < len(right):
            # Ordenamiento según la dirección especificada
            if (ascending and left[left_index][1][column_name] <= right[right_index][1][column_name]) or \
               (not ascending and left[left_index][1][column_name] >= right[right_index][1][column_name]):
                merged.append(left[left_index])
                left_index += 1
            else:
                merged.append(right[right_index])
                right_index += 1

        merged.extend(left[left_index:])
        merged.extend(right[right_index:])
        return merged

    sorted_data = merge_sort(data_to_sort)
    sorted_df = pd.DataFrame([item[1] for item in sorted_data])
    return sorted_df


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        # Ruta al archivo de datos
        dataset_path = "subset.csv"
        df = pd.read_csv(dataset_path)
        num_rows = len(df)

        # Sugerencia de procesospi
        num_cores = psutil.cpu_count(logical=False)
        recommended_processes = min(num_cores, num_rows // 1000) 
        print(f"Sugerencia: Use alrededor de {recommended_processes} procesos para un rendimiento óptimo.")
        print(f"Número de filas en el dataset: {num_rows}")
    else:
        df = None
        num_rows = None

    num_rows = comm.bcast(num_rows, root=0)
    chunk_size = num_rows // size
    remainder = num_rows % size

    if rank == 0:
        chunks = [df.iloc[i:i + chunk_size + (1 if i < remainder else 0)] for i in range(0, num_rows, chunk_size)]
    else:
        chunks = None

    local_df = comm.scatter(chunks, root=0)

    # Selección de columna y dirección de ordenamiento
    if rank == 0:
        try:
            sort_column = input("Ingrese el nombre de la columna con la que desea ordenar el dataset: ")
            sort_direction = input("¿Desea ordenar en orden ascendente? (s/n): ").strip().lower()
            ascending = sort_direction == 's'
        except Exception as e:
            print(f"Ocurrió un error: {e}")
            comm.Abort(1)
    else:
        sort_column = None
        ascending = None

    sort_column = comm.bcast(sort_column, root=0)
    ascending = comm.bcast(ascending, root=0)

    inicio_local = time.time()
    local_sorted_df = merge_sort_dataframe(local_df.copy(), sort_column, ascending=ascending)
    tiempo_local = time.time() - inicio_local
    print(f"[Proceso {rank}] Tiempo de ordenamiento local: {tiempo_local:.4f} segundos.")

    sorted_chunks = comm.gather(local_sorted_df, root=0)

    if rank == 0:
        sorted_df = pd.concat(sorted_chunks)
        sorted_df = merge_sort_dataframe(sorted_df.copy(), sort_column, ascending=ascending)

        file_path = os.path.join(os.getcwd(), "dataset_ordenado.csv")
        sorted_df.to_csv(file_path, index=False)
        print(f"El dataset ordenado se guardó en: {file_path}")
    else:
        sorted_df = None
