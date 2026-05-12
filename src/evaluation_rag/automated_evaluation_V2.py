"""Exluyo las matrices 3 y 5 de la evaluación"""

import pandas as pd
from typing import Callable
from tqdm import tqdm

# from llms_modules import LLMEvaluator

# Matrices excluidas del análisis por ausencia de datos de control
# operacional periódico (ver Sección 5.1 del paper).
EXCLUDED_MATRICES = {3, 5}


def auto_evaluation_rag_v2(
    path_ground_truth: str,
    name_excel: str = "Evaluación final.xlsx",
    pipeline_rag: Callable[[str], str] = None,
    llm_evaluador: Callable = None,
):
    """Evalúa los sistemas RAG sobre las matrices definidas en el GT-RAG,
    excluyendo las Matrices 3 y 5 (sin datos de control operacional periódico).

    Los totales globales (accuracy acumulado y hoja 'resultado final') reflejan
    únicamente las matrices evaluadas, evitando que las matrices excluidas
    distorsionen las métricas reportadas en el paper.

    Args:
        path_ground_truth: Ruta del archivo Excel con el ground truth (GT-RAG).
        name_excel:        Nombre/ruta del Excel de salida con la evaluación.
        pipeline_rag:      Pipeline RAG a evaluar. Debe exponer el método .query(str).
        llm_evaluador:     Evaluador LLM. Debe exponer el método .evaluation(query, response, reference).
    """

    excel_file = pd.read_excel(path_ground_truth, sheet_name=None)

    final_evaluation = {}
    summary_data = []  # Resultados de accuracy por matriz (solo evaluadas)
    total_correct = 0  # Aciertos acumulados entre matrices evaluadas
    total_valid = 0  # Preguntas válidas procesadas entre matrices evaluadas

    num_matrices = len(excel_file)

    for matriz_id in tqdm(range(1, num_matrices + 1), desc="Evaluación Matriz"):

        # ── Exclusión de matrices no evaluadas ──────────────────────────────
        if matriz_id in EXCLUDED_MATRICES:
            print(
                f"Matriz {matriz_id} — EXCLUIDA (sin datos de control operacional periódico)"
            )
            continue

        matriz_name = f"Matriz {matriz_id}"
        print(matriz_name)
        df_matriz = excel_file[matriz_name]

        final_evaluation[matriz_name] = {}
        parametro_counter = 1

        # Procesar cada 3 filas empezando desde la fila 1
        for j in range(1, len(df_matriz), 3):
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

            # ── Inferencia RAG + evaluación LLM ─────────────────────────────
            try:
                result = pipeline_rag.query(query)

                if isinstance(result, tuple):
                    # Caso: retorna (Response, metadata)
                    rag_response = result[0].response
                    metadata = result[1]
                else:
                    # Caso: retorna solo Response
                    rag_response = result.response
                    metadata = result.source_nodes

                llm_eval = llm_evaluador.evaluation(query, rag_response, reference)

            except Exception as e:
                print(f"Error en {matriz_name} fila {j}: {e}")
                rag_response = "ERROR"
                llm_eval = "ERROR"

            parametro_name = f"Parámetro {parametro_counter}"
            final_evaluation[matriz_name][parametro_name] = {
                "Pregunta": query,
                "Respuesta RAG": rag_response,
                "Respuesta Referencia": reference,
                "LLM Evaluador": llm_eval,
            }
            parametro_counter += 1

    # ── Exportar a Excel ─────────────────────────────────────────────────────
    with pd.ExcelWriter(name_excel) as writer:

        for matriz_name, parametros in final_evaluation.items():

            data = [
                [
                    p["Pregunta"],
                    p["Respuesta RAG"],
                    p["Respuesta Referencia"],
                    p["LLM Evaluador"],
                ]
                for p in parametros.values()
            ]

            df = pd.DataFrame(
                data,
                columns=[
                    "Consulta",
                    "Respuesta RAG",
                    "Respuesta Referencia",
                    "Evaluación LLM",
                ],
            )

            # Accuracy de la matriz
            valid_evals = pd.to_numeric(df["Evaluación LLM"], errors="coerce").dropna()
            matrix_correct = valid_evals.sum()
            matrix_total = len(valid_evals)
            accuracy = (
                round(matrix_correct / matrix_total * 100, 2)
                if matrix_total > 0
                else 0.0
            )

            # Acumular totales globales (solo matrices evaluadas)
            total_correct += matrix_correct
            total_valid += matrix_total

            # Agregar fila de accuracy al final de la hoja
            accuracy_row = pd.DataFrame(
                {
                    "Consulta": ["Accuracy"],
                    "Respuesta RAG": [""],
                    "Respuesta Referencia": [""],
                    "Evaluación LLM": [f"{accuracy}%"],
                }
            )
            df = pd.concat([df, accuracy_row], ignore_index=True)

            df.to_excel(writer, sheet_name=matriz_name, index=False)

            summary_data.append({"Matriz": matriz_name, "Accuracy": f"{accuracy}%"})

        # Accuracy total (matrices evaluadas solamente)
        total_accuracy = (
            round(total_correct / total_valid * 100, 2) if total_valid > 0 else 0.0
        )
        summary_data.append(
            {"Matriz": "Total (matrices evaluadas)", "Accuracy": f"{total_accuracy}%"}
        )

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name="resultado final", index=False)
