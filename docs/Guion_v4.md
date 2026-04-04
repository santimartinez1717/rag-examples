# Guión — Presentación Anexo B TFG (v4)
**8 diapositivas · ~10 minutos**

---

## 🎤 SLIDE 1 — Portada (0:00 — 0:30)

> Buenos días / buenas tardes. Mi nombre es Santiago Martínez Díe, y hoy presento mi Trabajo Fin de Grado: **"Hacia el 95% de Confianza: Monitorización y Calibración de Agentes de IA basados en RAG"**. Dirigido por Francisco Ruiz Gonzalez y con la codirección de Julian Jack Monís Reeves, en el Grado en Ingeniería Matemática e Inteligencia Artificial de ICAI.

---

## 🎤 SLIDE 2 — Motivación (0:30 — 2:30)

> Para entender este proyecto hay que partir de un hecho: los sistemas RAG —Retrieval-Augmented Generation— se han convertido en la arquitectura dominante en aplicaciones empresariales de IA generativa. La idea es sencilla: en lugar de que el modelo responda solo desde su memoria interna, recupera documentos relevantes y los usa como contexto. Esto reduce alucinaciones y permite trabajar con información actualizada.
>
> Pero el ecosistema RAG ha evolucionado enormemente. Hoy existen muchas formas de hacer RAG: Reranking, Hybrid Search, Knowledge Graphs, Agentic RAG, Self-Reflective RAG... cada arquitectura con sus propias particularidades.
>
> Y aquí está el problema central: **todas estas arquitecturas son heterogéneas, y no existe hoy ningún framework que las evalúe a todas con las mismas métricas y produzca un indicador de confianza único, calibrado y comparable entre ellas**.
>
> La pregunta que guía el TFG es: ¿cuándo y cuánto podemos confiar en la respuesta de un agente RAG? ¿Y varía esa confianza según la arquitectura utilizada?

---

## 🎤 SLIDE 3 — Objetivos (2:30 — 4:00)

> El objetivo general es **construir un framework sobre LangChain y LangGraph** —los frameworks de referencia para construir agentes RAG— con **LangSmith como núcleo de observabilidad y evaluación**. LangChain proporciona las abstracciones para pipelines RAG; LangGraph permite flujos agénticos más complejos con decisiones condicionales; y LangSmith es donde viven las trazas, los datasets, los evaluadores y el análisis de resultados. Es el punto de apoyo central del proyecto, y la siguiente diapositiva entra en el detalle técnico de cómo se construye sobre él.
>
> Los cinco objetivos específicos siguen una lógica natural:
>
> **O1:** Tomar arquitecturas RAG reales desde repositorios públicos —Agentic, Graph, SQL, Hybrid— y evaluarlas sobre LangSmith. No construir agentes desde cero, sino evaluar los existentes.
> **O2:** Definir las métricas de evaluación: tanto universales —válidas para cualquier arquitectura— como específicas por tipología.
> **O3:** Construir una interfaz universal —mismo código para cualquier arquitectura RAG— con evaluadores automáticos de distintos tipos. El detalle viene en la siguiente slide.
> **O4:** Implementar la calibración que combina las métricas en un score único de 0 a 100% comparable entre sistemas.
> **O5:** Validar en casos reales y comparar arquitecturas heterogéneas objetivamente.

---

## 🎤 SLIDE 4 — Metodología: Framework y Métricas (4:00 — 5:30)

> Aquí ya entramos en el cómo. La arquitectura del framework tiene cuatro capas.
>
> En la base: los **agentes RAG existentes**, tomados de repositorios públicos. Sobre ellos: **LangSmith como núcleo** —captura trazas automáticamente, gestiona datasets y evaluadores, y registra todos los resultados. Y en la cima: el **score de confianza calibrado**.
>
> La capa clave es la de **evaluadores automáticos**, y aquí hay tres enfoques complementarios. El primero es **LLM-as-judge**: se usa un modelo de lenguaje como juez que, dado el contexto y la respuesta, razona y emite una puntuación —es flexible y entiende matices, pero tiene coste de API y cierta variabilidad. El segundo son **modelos NLI** —Natural Language Inference—: modelos ligeros entrenados específicamente para determinar si una hipótesis está implicada por una premisa. Aplicado a RAG, la pregunta es directa: ¿está la respuesta entailed por el contexto recuperado? Modelos como DeBERTa o RoBERTa-MNLI resuelven esto en milisegundos, sin API, y de forma determinista. Es especialmente útil para groundedness y retrieval relevance. Y como plus opcional, **frameworks externos** como Ragas o DeepEval para contrastar resultados y validar que las métricas propias son coherentes con lo que propone la literatura.
>
> Las métricas universales —válidas para cualquier arquitectura— son correctness, groundedness, answer relevance y retrieval relevance. Y el próximo paso son métricas específicas por tipología: Rerank Stability, validez del SQL generado, calidad del Cypher para Graph RAG...

---

## 🎤 SLIDE 5 — Metodología: Calibración (5:30 — 7:00)

> El flujo completo del framework: el agente ejecuta, LangSmith SDK captura las trazas, calculamos las métricas, las calibramos y combinamos, y obtenemos ese score único de 0 a 100%.
>
> Los tres productos concretos del framework:
>
> Primero, el **score único calibrado**: un único porcentaje que integra las métricas del framework de forma calibrada. Esto no existe en LangSmith —es la aportación central del TFG. LangSmith te da métricas separadas por experimento; el framework las convierte en un único número interpretable.
>
> Segundo, la **comparación entre arquitecturas heterogéneas**: como el score se calcula con el mismo proceso para cualquier sistema, permite afirmar con objetividad "Agentic RAG: 78%, Graph RAG: 85%, SQL RAG: 71%". Esto es posible porque el proceso de cálculo es idéntico para todas —algo que no existe hoy en ninguna herramienta.
>
> Tercero, el **análisis de sobreconfianza**: detectar cuándo el sistema reporta un score alto pero falla en la práctica —o viceversa—. Analizar si esos patrones son sistemáticos por arquitectura o tipo de consulta. Eso es lo que da rigor estadístico al framework.

---

## 🎤 SLIDE 6 — Plan de Trabajo (7:00 — 8:00)

> El plan de trabajo cubre desde diciembre de 2025 hasta finales de mayo de 2026, con seis fases bien delimitadas.
>
> La **Fase 0** —diciembre 2025 a enero 2026— ya está completada: investigación y familiarización con papers de métricas RAG, arquitecturas estado del arte, y toda la documentación de LangSmith SDK, LangChain y LangGraph.
>
> Ahora mismo inicio **F1+F2** —del 18 de febrero al 9 de marzo—: evaluar arquitecturas RAG reales desde repositorios con LangSmith y definir y testear los evaluadores: LLM-as-judge, modelos NLI y métricas lexicales. Estas dos fases van muy de la mano porque los evaluadores se diseñan probándolos sobre los agentes reales.
>
> **F3** —del 10 al 23 de marzo—: construir el esqueleto del framework, la interfaz universal. El "frontend" que unifica todo bajo la misma API independientemente de la arquitectura RAG subyacente.
>
> **F4** —del 24 de marzo al 13 de abril—: conectar lo hecho en F1+F2 con la interfaz de F3 e implementar la calibración y el score único.
>
> **F5** —del 14 de abril al 11 de mayo—: validación en un caso empresarial real, comparación objetiva entre arquitecturas y extensión con métricas específicas por tipología.
>
> **F6** —del 12 al 31 de mayo—: redacción completa y preparación de la defensa.

---

## 🎤 SLIDE 7 — Estado Actual (8:00 — 9:15)

> A día de hoy, 18 de febrero, tres tareas están completadas: revisión bibliográfica, entorno técnico configurado y arquitecturas RAG identificadas.
>
> La tarea activa en este momento es **probar agentes RAG reales con LangSmith**: estoy tomando implementaciones de repositorios públicos, evaluándolas, y buscando la forma más eficiente de analizar cada tipología de RAG. Esto alimentará directamente el diseño de los evaluadores y del framework.
>
> El siguiente paso inmediato es implementar los evaluadores como componentes reutilizables.
>
> *(Nota para el tribunal, si preguntan por ODS: ODS 9 porque el framework aporta infraestructuras de evaluación reproducibles para adoptar RAG de forma técnicamente informada. ODS 16 porque la trazabilidad y el score calibrado permiten auditar sistemas RAG en sectores sensibles, fomentando responsabilidad algorítmica alineada con marcos regulatorios.)*

---

## 🎤 SLIDE 8 — Cierre (9:15 — 10:00)

> Para cerrar, cuatro puntos:
>
> RAG se ha convertido en la arquitectura dominante de IA empresarial, pero evaluar arquitecturas heterogéneas de forma comparable es un problema abierto.
>
> LangChain/LangGraph + LangSmith son el núcleo. El framework añade lo que falta: interfaz universal, evaluadores automáticos de distintos tipos y score calibrado único.
>
> Ese score único permite comparar Agentic RAG, Graph RAG y SQL RAG con total objetividad. Eso no existe hoy en ninguna herramienta.
>
> Y el próximo paso concreto es construir el esqueleto de la interfaz universal —Fase 3, del 10 al 23 de marzo.
>
> Muchas gracias. Quedo a disposición del tribunal.

---

## ⏱ Resumen de tiempos

| Slide | Sección | Duración |
|-------|---------|----------|
| 1 | Portada | 0:30 |
| 2 | Motivación — RAG en auge + tipologías + problema | 2:00 |
| 3 | Objetivos (con contexto LangChain/LangSmith) | 1:30 |
| 4 | Metodología — framework y métricas | 1:30 |
| 5 | Metodología — calibración y score único | 1:30 |
| 6 | Plan de trabajo con fechas | 1:00 |
| 7 | Estado actual | 1:15 |
| 8 | Cierre | 0:45 |
| | **TOTAL** | **~10:00** |
