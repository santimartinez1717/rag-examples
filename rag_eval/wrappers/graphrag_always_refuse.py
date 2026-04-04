"""
US-G2: graphrag_always_refuse
Wrapper que siempre devuelve "I don't have information about that",
sin ejecutar ninguna consulta.

Propósito de validación:
- correctness ≈ 0.0 → nunca da una respuesta correcta
- negative_rejection ≈ 1.0 → siempre "rechaza" (incluso en preguntas con respuesta válida)
- faithfulness_nli: indeterminado (no hay afirmaciones que evaluar)

Criterio de aceptación (PLANNING.md US-G2):
  correctness ≈ 0.0, negative_rejection ≈ 1.0 en preguntas con respuesta válida
"""


def graphrag_always_refuse(inputs: dict) -> dict:
    """
    Wrapper que siempre rechaza responder, independientemente de la pregunta.

    Simula un sistema "sobre-conservador" que nunca alucina pero tampoco
    aporta ningún valor — el caso extremo de safe-but-useless.
    """
    return {
        "answer": "I don't have information about that.",
        "context": "",
        "cypher_query": "",
        "db_results": [],
        "architecture": "graphrag_always_refuse",
    }


if __name__ == "__main__":
    test_q = {"question": "How many employees are in the company?"}
    result = graphrag_always_refuse(test_q)
    print(f"Answer: {result['answer']}")
