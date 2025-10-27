import pandas as pd
from typing import Callable
from tqdm import tqdm

# from llms_modules import LLMEvaluator


def auto_evaluation_rag(
    path_ground_truth: str,
    name_excel: str = "Evaluación final.xlsx",
    pipeline_rag: Callable[[str], str] = None,
    llm_evaluador: Callable = None,
):
    """Esta función evalúa los sistemas RAG

    Args:
        path_ground_truth: Ruta del ground truth para la evaluación del RAG,
        name_excel: Nombre de la evaluación final
    """

    excel_file = pd.read_excel(path_ground_truth, sheet_name=None)

    final_evaluation = {}
    summary_data = []  # Para almacenar los resultados de accuracy de cada matriz
    total_correct = 0  # Total de evaluaciones correctas (1)
    total_valid = 0  # Total de evaluaciones válidas procesadas

    for matriz_id in tqdm(range(1, len(excel_file) + 1), desc="Evaluación Matriz"):
        matriz_name = f"Matriz {matriz_id}"
        print(matriz_name)
        df_matriz = excel_file[matriz_name]

        final_evaluation[matriz_name] = {}
        parametro_counter = 1

        # Procesar cada 3 filas empezando desde la fila 1
        for j in range(1, len(df_matriz), 3):
            # Extraer valores y convertir a string
            try:
                query = str(df_matriz.iloc[j, 0]).strip()
                reference = str(df_matriz.iloc[j, 1]).strip()
            except IndexError:
                reference = "N/A"

            # Saltar filas inválidas
            if query.lower() == "nan" or query == "":
                continue
            if reference.lower() == "nan" or reference == "":
                continue

            # Procesar datos
            try:
                # rag_response = pipeline_rag.query(query) # Esta línea funciona para AsynMultiQuery RAG
                rag_response, metadata = pipeline_rag.query(query)
                llm_eval = llm_evaluador.evaluation(query, rag_response, reference)
            except Exception as e:
                print(f"Error en {matriz_name} fila {j}: {e}")
                rag_response = "ERROR"
                llm_eval = "ERROR"

            # Almacenar en el diccionario
            parametro_name = f"Parámetro {parametro_counter}"
            final_evaluation[matriz_name][parametro_name] = {
                "Pregunta": query,
                "Respuesta RAG": rag_response,
                "Respuesta Referencia": reference,
                "LLM Evaluador": llm_eval,
            }
            parametro_counter += 1

    # Exportar a Excel con cálculo de accuracy
    with pd.ExcelWriter(name_excel) as writer:
        for matriz_name, parametros in final_evaluation.items():
            data = []
            for param in parametros.values():
                data.append(
                    [
                        param["Pregunta"],
                        param["Respuesta RAG"],
                        param["Respuesta Referencia"],
                        param["LLM Evaluador"],
                    ]
                )
            df = pd.DataFrame(
                data,
                columns=[
                    "Consulta",
                    "Respuesta RAG",
                    "Respuesta Referencia",
                    "Evaluación LLM",
                ],
            )

            # Calcular accuracy de la matriz
            valid_evals = pd.to_numeric(df["Evaluación LLM"], errors="coerce").dropna()
            matrix_correct = valid_evals.sum()
            matrix_total = len(valid_evals)
            accuracy = (
                (matrix_correct / matrix_total * 100).round(2)
                if matrix_total > 0
                else 0.0
            )

            # Acumular totales
            total_correct += matrix_correct
            total_valid += matrix_total

            # Agregar fila de accuracy al DataFrame de la matriz
            accuracy_row = pd.DataFrame(
                {
                    "Consulta": ["Accuracy"],
                    "Respuesta RAG": [""],
                    "Respuesta Referencia": [""],
                    "Evaluación LLM": [f"{accuracy}%"],
                }
            )
            df = pd.concat([df, accuracy_row], ignore_index=True)

            # Guardar hoja de la matriz
            df.to_excel(writer, sheet_name=matriz_name, index=False)

            # Almacenar datos para la hoja de resumen
            summary_data.append({"Matriz": matriz_name, "Accuracy": f"{accuracy}%"})

        # Calcular accuracy total
        total_accuracy = (
            (total_correct / total_valid * 100).round(2) if total_valid > 0 else 0.0
        )
        summary_data.append({"Matriz": "Total", "Accuracy": f"{total_accuracy}%"})

        # Crear hoja de resumen final
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name="resultado final", index=False)
