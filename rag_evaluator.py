"""
RAG Evaluator universal para cualquier pipeline (cadena, grafo, función, etc.)
Basado en el tutorial oficial de LangSmith:
https://docs.langchain.com/langsmith/evaluate-rag-tutorial

Uso:
1. Define tu función RAG: debe aceptar un dict de entrada y devolver un dict de salida.
2. Prepara tu dataset (lista de dicts con 'inputs' y 'outputs').
3. Llama a evaluate_rag(rag_fn, dataset, ...)

Ejemplo de rag_fn:
def rag_fn(inputs: dict) -> dict:
    # ... tu pipeline ...
    return {"answer": ..., "documents": ...}

"""

from typing import Callable, List, Dict
from langsmith import Client
from langsmith.utils import LangSmithConflictError
from langchain_openai import ChatOpenAI
from typing_extensions import Annotated, TypedDict

# --- Evaluadores estándar ---

class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]

class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "True if the answer addresses the question"]

class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[bool, ..., "True if the answer is grounded in the docs"]

class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "True if the docs are relevant to the question"]

# LLMs para los evaluadores
llm = ChatOpenAI(model="gpt-4.1", temperature=0)

grader_llm = llm.with_structured_output(CorrectnessGrade, method="json_schema", strict=True)
relevance_llm = llm.with_structured_output(RelevanceGrade, method="json_schema", strict=True)
grounded_llm = llm.with_structured_output(GroundedGrade, method="json_schema", strict=True)
retrieval_relevance_llm = llm.with_structured_output(RetrievalRelevanceGrade, method="json_schema", strict=True)

# --- Evaluadores universales ---
def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Evalúa exactitud de la respuesta vs ground truth."""
    student_answer = outputs.get('answer', outputs.get('output', str(outputs)))
    ref_answer = reference_outputs.get('answer', reference_outputs.get('output', str(reference_outputs)))

    answers = f"""
QUESTION: {inputs['question']}
GROUND TRUTH ANSWER: {ref_answer}
STUDENT ANSWER: {student_answer}"""
    grade = grader_llm.invoke([
        {"role": "system", "content": correctness_instructions()},
        {"role": "user", "content": answers}
    ])
    return {"key": "correctness", "score": 1 if grade["correct"] else 0, "comment": grade.get("explanation", "")}

def relevance(inputs: dict, outputs: dict) -> dict:
    """Evalúa relevancia de la respuesta vs pregunta."""
    student_answer = outputs.get('answer', outputs.get('output', str(outputs)))
    answer = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {student_answer}"
    grade = relevance_llm.invoke([
        {"role": "system", "content": relevance_instructions()},
        {"role": "user", "content": answer}
    ])
    return {"key": "relevance", "score": 1 if grade["relevant"] else 0, "comment": grade.get("explanation", "")}

def groundedness(inputs: dict, outputs: dict) -> dict:
    """Evalúa groundedness de la respuesta vs docs recuperados o contexto disponible."""
    student_answer = outputs.get('answer', outputs.get('output', str(outputs)))
    docs = outputs.get("documents", [])
    # Para GraphRAG: si no hay documents, usar la respuesta como contexto (viene de la BD)
    if docs:
        doc_string = "\n\n".join(doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in docs)
    else:
        doc_string = outputs.get("context", student_answer)
    answer = f"FACTS: {doc_string}\nSTUDENT ANSWER: {student_answer}"
    grade = grounded_llm.invoke([
        {"role": "system", "content": grounded_instructions()},
        {"role": "user", "content": answer}
    ])
    return {"key": "groundedness", "score": 1 if grade["grounded"] else 0, "comment": grade.get("explanation", "")}

def retrieval_relevance(inputs: dict, outputs: dict) -> dict:
    """Evalúa relevancia de los docs recuperados vs pregunta."""
    docs = outputs.get("documents", [])
    # Para GraphRAG: si no hay documents, usar la respuesta (que viene directamente de la BD)
    if docs:
        doc_string = "\n\n".join(doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in docs)
    else:
        doc_string = outputs.get("context", outputs.get("answer", ""))
    answer = f"FACTS: {doc_string}\nQUESTION: {inputs['question']}"
    grade = retrieval_relevance_llm.invoke([
        {"role": "system", "content": retrieval_relevance_instructions()},
        {"role": "user", "content": answer}
    ])
    return {"key": "retrieval_relevance", "score": 1 if grade["relevant"] else 0, "comment": grade.get("explanation", "")}

# --- Instrucciones para los evaluadores ---
def correctness_instructions():
    return """You are a teacher grading a quiz. You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. Here is the grade criteria to follow:\n(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. (2) Ensure that the student answer does not contain any conflicting statements.\n(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the  ground truth answer.\n\nCorrectness:\nA correctness value of True means that the student's answer meets all of the criteria.\nA correctness value of False means that the student's answer does not meet all of the criteria.\n\nExplain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""
def relevance_instructions():
    return """You are a teacher grading a quiz. You will be given a QUESTION and a STUDENT ANSWER. Here is the grade criteria to follow:\n(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION\n(2) Ensure the STUDENT ANSWER helps to answer the QUESTION\n\nRelevance:\nA relevance value of True means that the student's answer meets all of the criteria.\nA relevance value of False means that the student's answer does not meet all of the criteria.\n\nExplain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""
def grounded_instructions():
    return """You are a teacher grading a quiz. You will be given FACTS and a STUDENT ANSWER. Here is the grade criteria to follow:\n(1) Ensure the STUDENT ANSWER is grounded in the FACTS. (2) Ensure the STUDENT ANSWER does not contain \"hallucinated\" information outside the scope of the FACTS.\n\nGrounded:\nA grounded value of True means that the student's answer meets all of the criteria.\nA grounded value of False means that the student's answer does not meet all of the criteria.\n\nExplain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""
def retrieval_relevance_instructions():
    return """You are a teacher grading a quiz. You will be given a QUESTION and a set of FACTS provided by the student. Here is the grade criteria to follow:\n(1) You goal is to identify FACTS that are completely unrelated to the QUESTION\n(2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant\n(3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met\n\nRelevance:\nA relevance value of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.\nA relevance value of False means that the FACTS are completely unrelated to the QUESTION.\n\nExplain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

# --- Función principal universal ---
def evaluate_rag(
    rag_fn: Callable[[dict], dict],
    dataset: List[Dict],
    dataset_name: str = "rag-eval-dataset",
    project: str = "rag-eval-project",
    use_correctness: bool = True,
    use_relevance: bool = True,
    use_groundedness: bool = True,
    use_retrieval_relevance: bool = True,
    **kwargs
):
    """
    Evalúa cualquier función RAG sobre un dataset usando evaluadores estándar.
    """
    client = Client()
    # Intentar crear el dataset, si ya existe lo reutiliza
    try:
        ds = client.create_dataset(dataset_name=dataset_name)
        client.create_examples(dataset_id=ds.id, examples=dataset)
        print(f"✅ Dataset creado: '{dataset_name}' con {len(dataset)} ejemplos")
    except (LangSmithConflictError, Exception) as e:
        if "already exists" in str(e) or "Conflict" in str(e):
            # Buscar el dataset existente
            datasets = client.list_datasets()
            ds = next((d for d in datasets if d.name == dataset_name), None)
            if ds is None:
                raise RuntimeError(f"Dataset '{dataset_name}' exists but could not be found.")
            print(f"⚠️ Dataset '{dataset_name}' ya existe, se reutiliza.")
        else:
            raise

    # Seleccionar evaluadores
    evaluators = []
    if use_correctness:
        evaluators.append(correctness)
    if use_relevance:
        evaluators.append(relevance)
    if use_groundedness:
        evaluators.append(groundedness)
    if use_retrieval_relevance:
        evaluators.append(retrieval_relevance)

    # Ejecutar evaluación
    print(f"\n🚀 Ejecutando evaluación con evaluadores: {[e.__name__ for e in evaluators]}")
    results = client.evaluate(
        rag_fn,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=project,
        **kwargs
    )
    print(f"\n🌐 Ver resultados detallados en: https://smith.langchain.com")
    print(f"📁 Dataset: {dataset_name}")
    print(f"📁 Proyecto: {project}")
    return results

# --- Ejemplo de uso ---
if __name__ == "__main__":
    # Ejemplo de función RAG dummy
    def dummy_rag(inputs):
        return {"answer": "42", "documents": ["The answer to everything is 42."]}

    # Dataset de ejemplo
    dataset = [
        {"inputs": {"question": "What is the answer to life?"}, "outputs": {"answer": "42"}},
        {"inputs": {"question": "What is 6 x 7?"}, "outputs": {"answer": "42"}},
    ]

    evaluate_rag(dummy_rag, dataset)
