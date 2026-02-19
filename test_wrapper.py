# Test para comprobar el wrapper neo4j_qa_chain_wrapper de rag_evaluator.py
from rag_evaluator import neo4j_qa_chain_wrapper

# Pregunta de prueba
question = "How many employees are in the company?"
inputs = {"question": question}

# Ejecutar el wrapper
result = neo4j_qa_chain_wrapper(inputs)
print("Input:", inputs)
print("Output:", result)
assert isinstance(result, dict), "El resultado debe ser un diccionario."
assert "answer" in result, "El resultado debe tener la clave 'answer'."
print("Test PASSED: El wrapper devuelve un dict con 'answer'.")
