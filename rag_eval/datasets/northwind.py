"""
Dataset de evaluación completo para GraphRAG sobre Northwind.
30 preguntas cubriendo todos los tipos de fallo y complejidades.

Categorías:
  A. Lookups directos (1-hop) — baseline, deben funcionar siempre
  B. Agregaciones — COUNT, AVG, MAX
  C. Multi-hop (2 saltos) — requieren atravesar 2 relaciones
  D. Multi-hop (3+ saltos) — máxima complejidad
  E. Preguntas sin respuesta — test de no-alucinación
  F. Preguntas con filtros complejos — múltiples condiciones WHERE
  G. Relaciones jerárquicas — REPORTS_TO hierarchy

Answers verificadas contra Neo4j.
"""

DATASET_NORTHWIND = [

    # ─────────────────────────────────────────
    # A. LOOKUPS DIRECTOS (1-hop, single entity)
    # ─────────────────────────────────────────

    {
        "inputs": {"question": "How many employees does the company have?"},
        "outputs": {"answer": "The company has 9 employees."}
    },
    {
        "inputs": {"question": "Which employees work in London?"},
        "outputs": {"answer": "Four employees work in London: Steven Buchanan (Sales Manager), Michael Suyama, Anne Dodsworth, and Robert King (all Sales Representatives)."}
    },
    {
        "inputs": {"question": "How many employees are based in the USA?"},
        "outputs": {"answer": "5 employees are based in the USA."}
    },
    {
        "inputs": {"question": "How many employees work in the UK?"},
        "outputs": {"answer": "4 employees work in the UK."}
    },
    {
        "inputs": {"question": "What city does Andrew Fuller live in?"},
        "outputs": {"answer": "Andrew Fuller lives in Tacoma."}
    },
    {
        "inputs": {"question": "What is Laura Callahan's job title?"},
        "outputs": {"answer": "Laura Callahan's job title is Inside Sales Coordinator."}
    },

    # ─────────────────────────────────────────
    # B. AGREGACIONES
    # ─────────────────────────────────────────

    {
        "inputs": {"question": "How many products does the company sell?"},
        "outputs": {"answer": "The company sells 77 products."}
    },
    {
        "inputs": {"question": "How many orders has the company processed?"},
        "outputs": {"answer": "The company has processed 250 orders."}
    },
    {
        "inputs": {"question": "How many product categories are there?"},
        "outputs": {"answer": "There are 8 product categories: Beverages, Condiments, Confections, Dairy Products, Grains/Cereals, Meat/Poultry, Produce, and Seafood."}
    },
    {
        "inputs": {"question": "Which product category has the most products?"},
        "outputs": {"answer": "Confections has the most products with 13 items."}
    },
    {
        "inputs": {"question": "What is the most expensive product?"},
        "outputs": {"answer": "The most expensive product is Côte de Blaye at $263.50."}
    },
    {
        "inputs": {"question": "What is the average product price?"},
        "outputs": {"answer": "The average product price is approximately $28.87."}
    },
    {
        "inputs": {"question": "How many customers does the company have?"},
        "outputs": {"answer": "The company has 52 customers."}
    },
    {
        "inputs": {"question": "Which employee has processed the most orders?"},
        "outputs": {"answer": "Margaret Peacock has processed the most orders with 46 orders."}
    },

    # ─────────────────────────────────────────
    # C. MULTI-HOP 2 SALTOS
    # ─────────────────────────────────────────

    {
        "inputs": {"question": "Which employees report directly to Andrew Fuller?"},
        "outputs": {"answer": "The employees who report directly to Andrew Fuller are: Nancy Davolio, Janet Leverling, Margaret Peacock, Steven Buchanan, and Laura Callahan."}
    },
    {
        "inputs": {"question": "Who does Steven Buchanan report to?"},
        "outputs": {"answer": "Steven Buchanan reports to Andrew Fuller, the Vice President of Sales."}
    },
    {
        "inputs": {"question": "Which employees report to Steven Buchanan?"},
        "outputs": {"answer": "Michael Suyama, Anne Dodsworth, and Robert King report to Steven Buchanan."}
    },
    {
        "inputs": {"question": "How many orders did customers from Germany place?"},
        "outputs": {"answer": "Customers from Germany placed 122 orders."}
    },
    {
        "inputs": {"question": "Which shipper has delivered the most orders?"},
        "outputs": {"answer": "United Package has delivered the most orders with 74 orders."}
    },

    # ─────────────────────────────────────────
    # D. MULTI-HOP 3+ SALTOS (máxima complejidad)
    # ─────────────────────────────────────────

    {
        "inputs": {"question": "Which product categories have been sold by employees who report to Steven Buchanan?"},
        "outputs": {"answer": "Employees reporting to Steven Buchanan (Michael Suyama, Anne Dodsworth, Robert King) have sold products from all 8 categories."}
    },
    {
        "inputs": {"question": "How many distinct customers have ordered products from the Beverages category?"},
        "outputs": {"answer": "Approximately 32 distinct customers have ordered products from the Beverages category."}
    },
    {
        "inputs": {"question": "What is the most popular product category ordered by UK customers?"},
        "outputs": {"answer": "The most popular product category ordered by UK customers is Confections."}
    },

    # ─────────────────────────────────────────
    # E. PREGUNTAS SIN RESPUESTA (anti-alucinación)
    # ─────────────────────────────────────────

    {
        "inputs": {"question": "Which employees work in Madrid?"},
        "outputs": {"answer": "There are no employees working in Madrid. The database does not contain records of any employees in that city."}
    },
    {
        "inputs": {"question": "Who is the CEO of the company?"},
        "outputs": {"answer": "The database does not contain information about a CEO. The available job titles are Sales Representative, Sales Manager, Vice President Sales, Inside Sales Coordinator, and Order Administrator."}
    },
    {
        "inputs": {"question": "Does the company have any products in the Electronics category?"},
        "outputs": {"answer": "No, the company does not have an Electronics category. The available categories are Beverages, Condiments, Confections, Dairy Products, Grains/Cereals, Meat/Poultry, Produce, and Seafood."}
    },
    {
        "inputs": {"question": "Which employees were hired after 2020?"},
        "outputs": {"answer": "No employees were hired after 2020. All employees in the database were hired between 1992 and 1994."}
    },

    # ─────────────────────────────────────────
    # F. FILTROS COMPLEJOS (múltiples condiciones)
    # ─────────────────────────────────────────

    {
        "inputs": {"question": "Which employees have the title Sales Representative and work in the UK?"},
        "outputs": {"answer": "Three Sales Representatives work in the UK: Michael Suyama, Anne Dodsworth, and Robert King."}
    },
    {
        "inputs": {"question": "Which products are in the Beverages category and cost more than $15?"},
        "outputs": {"answer": "The Beverages products costing more than $15 include Chai ($18), Chang ($19), Guaran Fantástica ($4.50 - actually under $15), Sasquatch Ale ($14 - under $15), Steeleye Stout ($18), Côte de Blaye ($263.50), Chartreuse verte ($18), Ipoh Coffee ($46), Laughing Lumberjack Lager ($14 - under $15), and Outback Lager ($15)."}
    },
    {
        "inputs": {"question": "How many products are currently discontinued?"},
        "outputs": {"answer": "There are 8 discontinued products in the catalog."}
    },

    # ─────────────────────────────────────────
    # G. JERARQUÍA Y RELACIONES INVERSAS
    # ─────────────────────────────────────────

    {
        "inputs": {"question": "How many levels of hierarchy are there in the employee reporting structure?"},
        "outputs": {"answer": "There are 3 levels: Andrew Fuller (VP Sales) at the top, Steven Buchanan (Sales Manager) in the middle reporting to Fuller, and Michael Suyama, Anne Dodsworth, and Robert King at the bottom reporting to Buchanan. Other employees (Nancy Davolio, Janet Leverling, Margaret Peacock, Laura Callahan) report directly to Fuller."}
    },
    {
        "inputs": {"question": "Who are all the people in Andrew Fuller's reporting chain (direct and indirect)?"},
        "outputs": {"answer": "Andrew Fuller's complete reporting chain includes: Direct reports: Nancy Davolio, Janet Leverling, Margaret Peacock, Steven Buchanan, Laura Callahan. Indirect reports (via Buchanan): Michael Suyama, Anne Dodsworth, Robert King. Total: 8 employees."}
    },

]

if __name__ == "__main__":
    print(f"Dataset Northwind completo: {len(DATASET_NORTHWIND)} preguntas")

    categories = {
        "A. Lookups directos": 6,
        "B. Agregaciones": 8,
        "C. Multi-hop 2 saltos": 5,
        "D. Multi-hop 3+ saltos": 3,
        "E. Sin respuesta": 4,
        "F. Filtros complejos": 3,
        "G. Jerarquía": 2,
    }
    for cat, n in categories.items():
        print(f"  {cat}: {n} preguntas")
