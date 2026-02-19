"""
Script para ejecutar el notebook de evaluación GraphRAG
"""
import subprocess
import sys

def run_notebook():
    """Ejecuta el notebook 03_evaluacion_neo4j_rag.ipynb"""
    try:
        print("🚀 Ejecutando notebook de evaluación GraphRAG...")
        result = subprocess.run([
            "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            "03_evaluacion_neo4j_rag.ipynb"
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Notebook ejecutado exitosamente!")
            print(result.stdout)
        else:
            print("❌ Error ejecutando notebook:")
            print(result.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_notebook()
