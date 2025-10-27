import json
import pandas as pd
import openpyxl
from openpyxl.styles import Font
import os
import re
import glob
from pathlib import Path


def extract_model_name_from_filename(filename):
    """
    Extrae el nombre del modelo del nombre del archivo y lo mapea a nombres simples
    """
    # Mapeo de patrones de archivos a nombres simples en el orden deseado
    model_mapping = {
        "all-MiniLM-L6-v2": "all-mini-L6-v2",
        "all-MiniLM-L12-v2": "all-mini-L12-v2",
        "paraphrase-multilingual-MiniLM-L12-v2": "multilingual",
        "nomic-embed-text-v1.5": "nomic",
    }

    # Patrón para extraer el modelo entre 'sentence-transformers_' y '_deposito' o '_tranque'
    pattern = r"sentence-transformers[_/]([^_]+(?:-[^_]+)*?)_"
    match = re.search(pattern, filename)
    if match:
        model_key = match.group(1)
        return model_mapping.get(model_key, model_key)

    # Patrón alternativo para modelos como nomic-embed
    pattern_alt = r"eval_([^_]+(?:-[^_]+)*?)_(?:tranque|deposito)"
    match_alt = re.search(pattern_alt, filename)
    if match_alt:
        model_name = match_alt.group(1)
        # Limpiar prefijos comunes
        if model_name.startswith("sentence-transformers_"):
            model_name = model_name.replace("sentence-transformers_", "")
        return model_mapping.get(model_name, model_name)

    # Buscar directamente en el mapeo por patrones parciales
    for key, value in model_mapping.items():
        if key in filename:
            return value

    return None


def get_model_order():
    """
    Define el orden específico de los modelos
    """
    return ["all-mini-L6-v2", "all-mini-L12-v2", "multilingual", "nomic"]


def process_single_json_data(json_data, model_name=None):
    """
    Procesa un único archivo JSON y retorna los datos organizados
    """
    # Usar el nombre del modelo proporcionado o extraerlo del JSON
    if (
        not model_name
        and "model_info" in json_data
        and "display_name" in json_data["model_info"]
    ):
        display_name = json_data["model_info"]["display_name"]
        model_name = (
            display_name.split("/")[-1] if "/" in display_name else display_name
        )

    if not model_name:
        model_name = "Unknown-Model"

    # Crear diccionario para organizar los datos por matriz y parámetro
    matrix_data = {}

    # Procesar cada evaluación de parámetro
    for param_eval in json_data.get("parameter_evaluations", []):
        matriz = param_eval["matriz"]
        parametro = param_eval["parametro"]
        metrics = param_eval["evaluation"]["metrics"]

        if matriz not in matrix_data:
            matrix_data[matriz] = {}

        matrix_data[matriz][parametro] = {
            "precision": metrics.get("precision", 0),
            "recall": metrics.get("recall", 0),
            "f1_score": metrics.get("f1_score", 0),
            "average_precision": metrics.get("average_precision", 0),
        }

    return model_name, matrix_data


def create_table_data_from_matrix(matrix_data, model_name):
    """
    Crea los datos de la tabla para un modelo específico
    """
    # Ordenar matrices numéricamente
    sorted_matrices = sorted(matrix_data.keys(), key=lambda x: int(x.split("_")[1]))

    # Crear encabezados de columnas y fila de parámetros
    headers = [model_name]
    param_row = [""]  # Primera celda vacía para la columna de métricas

    for matriz in sorted_matrices:
        params = sorted(matrix_data[matriz].keys(), key=lambda x: int(x.split("_")[1]))
        for param in params:
            headers.append(f"{matriz.replace('_', ' ').title()}")
            param_row.append(f"{param.replace('_', ' ').title()}")

    # Crear filas de métricas
    precision_row = ["Precision"]
    recall_row = ["Recall"]
    f1_row = ["F1 Score"]
    ap_row = ["AP"]

    for matriz in sorted_matrices:
        params = sorted(matrix_data[matriz].keys(), key=lambda x: int(x.split("_")[1]))
        for param in params:
            param_data = matrix_data[matriz][param]
            precision_row.append(round(param_data["precision"], 4))
            recall_row.append(round(param_data["recall"], 4))
            f1_row.append(round(param_data["f1_score"], 4))
            ap_row.append(round(param_data["average_precision"], 4))

    return [headers, param_row, precision_row, recall_row, f1_row, ap_row]


def process_multiple_json_files(json_files_pattern, output_excel_path):
    """
    Procesa múltiples archivos JSON y crea un Excel con todas las tablas
    ordenadas según el orden específico de modelos

    Args:
        json_files_pattern: Patrón de archivos (ej: "*.json" o lista de rutas)
        output_excel_path: Ruta del archivo Excel de salida
    """
    # Obtener lista de archivos
    if isinstance(json_files_pattern, str):
        if "*" in json_files_pattern or "?" in json_files_pattern:
            json_files = glob.glob(json_files_pattern)
        else:
            # Es un directorio, buscar todos los JSON
            if os.path.isdir(json_files_pattern):
                json_files = glob.glob(os.path.join(json_files_pattern, "*.json"))
            else:
                # Es un archivo específico
                json_files = [json_files_pattern]
    elif isinstance(json_files_pattern, list):
        json_files = json_files_pattern
    else:
        raise ValueError(
            "json_files_pattern debe ser una cadena con patrón, directorio, archivo o lista de archivos"
        )

    if not json_files:
        raise ValueError("No se encontraron archivos JSON para procesar")

    print(f"Procesando {len(json_files)} archivos JSON...")

    # Procesar cada archivo JSON y almacenar temporalmente
    processed_data = {}  # {model_name: (table_data, json_file)}

    for json_file in sorted(json_files):
        print(f"Procesando: {json_file}")

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            # Extraer nombre del modelo del archivo
            filename = os.path.basename(json_file)
            model_name = extract_model_name_from_filename(filename)

            if not model_name:
                print(f"No se pudo extraer el nombre del modelo de: {filename}")
                continue

            # Procesar datos del JSON
            model_name, matrix_data = process_single_json_data(json_data, model_name)

            # Crear datos de la tabla
            table_data = create_table_data_from_matrix(matrix_data, model_name)

            processed_data[model_name] = (table_data, json_file)

        except Exception as e:
            print(f"Error procesando {json_file}: {str(e)}")
            continue

    if not processed_data:
        raise ValueError("No se pudieron procesar datos de ningún archivo JSON")

    # Ordenar según el orden específico de modelos
    model_order = get_model_order()
    ordered_data = []
    ordered_model_names = []

    # Primero agregar los modelos en el orden específico
    for model_name in model_order:
        if model_name in processed_data:
            table_data, json_file = processed_data[model_name]
            ordered_data.append(table_data)
            ordered_model_names.append(model_name)
            print(f"Modelo ordenado: {model_name} (de {json_file})")

    # Luego agregar cualquier modelo que no esté en el orden específico
    for model_name, (table_data, json_file) in processed_data.items():
        if model_name not in model_order:
            ordered_data.append(table_data)
            ordered_model_names.append(model_name)
            print(f"Modelo adicional: {model_name} (de {json_file})")

    # Crear el Excel con todas las tablas ordenadas
    create_combined_excel(ordered_data, ordered_model_names, output_excel_path)

    return output_excel_path


def create_combined_excel(all_table_data, model_names, output_path):
    """
    Crea un archivo Excel con múltiples tablas en una sola hoja
    """
    # Crear workbook y worksheet
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.title = "Model Evaluation Results"

    current_row = 1

    for i, (table_data, model_name) in enumerate(zip(all_table_data, model_names)):
        print(f"Agregando tabla para modelo: {model_name}")

        # Escribir datos de la tabla
        for row_data in table_data:
            for col_idx, value in enumerate(row_data, 1):
                cell = worksheet.cell(row=current_row, column=col_idx, value=value)

                # Formatear encabezados (primera fila de cada tabla)
                if row_data == table_data[0]:  # Primera fila (headers)
                    cell.font = Font(bold=True)
                # Formatear nombres de métricas (primera columna desde la segunda fila)
                elif (
                    col_idx == 1 and row_data != table_data[1]
                ):  # Primera columna, no fila de parámetros
                    cell.font = Font(bold=True)

            current_row += 1

        # Agregar espacio entre tablas (excepto después de la última)
        if i < len(all_table_data) - 1:
            current_row += 2  # 2 filas de espacio

    # Ajustar ancho de columnas
    for column in worksheet.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                if cell.value and len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = min(max_length + 2, 20)
        worksheet.column_dimensions[column_letter].width = adjusted_width

    # Guardar el archivo
    workbook.save(output_path)
    workbook.close()

    print(f"Archivo Excel combinado guardado exitosamente: {output_path}")
    print(f"Orden de modelos en el Excel: {model_names}")


def process_directory_json_files(directory_path, output_excel_path=None):
    """
    Función de conveniencia para procesar todos los JSON en un directorio
    """
    if output_excel_path is None:
        output_excel_path = os.path.join(
            directory_path, "combined_evaluation_results.xlsx"
        )

    json_pattern = os.path.join(directory_path, "*.json")
    return process_multiple_json_files(json_pattern, output_excel_path)


# # Ejemplo de uso
# if __name__ == "__main__":
#     # Opción 1: Procesar todos los JSON en un directorio
#     directory_path = "./results/01_tranque_retrieval_evaluation/"
#     output_excel = "01_tranque_relave_evaluation_retrieval.xlsx"

#     try:
#         result_file = process_directory_json_files(directory_path, output_excel)
#         print(f"¡Procesamiento completado! Resultado guardado en: {result_file}")
#     except Exception as e:
#         print(f"Error durante el procesamiento: {str(e)}")

#     # Opción 2: Procesar archivos específicos
#     """
#     specific_files = [
#         "./results/01_tranque_retrieval_evaluation/eval_sentence-transformers_all-MiniLM-L6-v2_tranque_de_relaves_k13_20250628_235924_complete.json",
#         "./results/01_tranque_retrieval_evaluation/eval_sentence-transformers_all-MiniLM-L12-v2_tranque_de_relaves_k13_20250629_161132_complete.json",
#         "./results/01_tranque_retrieval_evaluation/eval_sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2_tranque_de_relaves_k13_20250629_161526_complete.json",
#         "./results/01_tranque_retrieval_evaluation/eval_nomic-nomic-embed-text-v1.5_tranque_de_relaves_k13_20250629_161940_complete.json"
#     ]

#     try:
#         result_file = process_multiple_json_files(specific_files, "specific_models_evaluation.xlsx")
#         print(f"¡Procesamiento completado! Resultado guardado en: {result_file}")
#     except Exception as e:
#         print(f"Error durante el procesamiento: {str(e)}")
#     """
