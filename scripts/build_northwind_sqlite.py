"""
build_northwind_sqlite.py — Extrae datos Northwind de Neo4j y construye data/northwind.db (SQLite).

Prerequisito: Neo4j Desktop corriendo con base de datos 'neo4j' (Northwind cargado).
  bolt://localhost:7687, auth en .env (NEO4J_USERNAME / NEO4J_PASSWORD)

Tablas generadas:
  employees(9), categories(8), suppliers(29), customers(52),
  shippers(3), products(77), orders(94 únicos), order_details(213)

Uso:
  python scripts/build_northwind_sqlite.py

El script es idempotente — borra y recrea northwind.db en cada ejecución.
"""
import os
import sqlite3
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

DB_PATH = ROOT / "data" / "northwind.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "password")


def get_driver():
    from neo4j import GraphDatabase
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))


def build_sqlite(driver):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Drop and recreate all tables
    cur.executescript("""
    DROP TABLE IF EXISTS order_details;
    DROP TABLE IF EXISTS orders;
    DROP TABLE IF EXISTS products;
    DROP TABLE IF EXISTS customers;
    DROP TABLE IF EXISTS suppliers;
    DROP TABLE IF EXISTS shippers;
    DROP TABLE IF EXISTS categories;
    DROP TABLE IF EXISTS employees;

    CREATE TABLE employees (
        employee_id   INTEGER PRIMARY KEY,
        first_name    TEXT,
        last_name     TEXT,
        title         TEXT,
        city          TEXT,
        country       TEXT,
        hire_date     TEXT,
        reports_to    INTEGER
    );

    CREATE TABLE categories (
        category_id   INTEGER PRIMARY KEY,
        category_name TEXT
    );

    CREATE TABLE suppliers (
        supplier_id   INTEGER PRIMARY KEY,
        company_name  TEXT,
        country       TEXT,
        city          TEXT
    );

    CREATE TABLE customers (
        customer_id   TEXT PRIMARY KEY,
        company_name  TEXT,
        country       TEXT,
        city          TEXT
    );

    CREATE TABLE shippers (
        shipper_id    INTEGER PRIMARY KEY,
        company_name  TEXT,
        phone         TEXT
    );

    CREATE TABLE products (
        product_id    INTEGER PRIMARY KEY,
        product_name  TEXT,
        unit_price    REAL,
        units_in_stock INTEGER,
        discontinued  INTEGER,
        category_id   INTEGER,
        supplier_id   INTEGER
    );

    CREATE TABLE orders (
        order_id      INTEGER PRIMARY KEY,
        customer_id   TEXT,
        employee_id   INTEGER,
        order_date    TEXT,
        ship_country  TEXT,
        ship_city     TEXT,
        shipper_id    INTEGER
    );

    CREATE TABLE order_details (
        order_id      INTEGER,
        product_id    INTEGER,
        unit_price    REAL,
        quantity      INTEGER,
        discount      REAL,
        PRIMARY KEY (order_id, product_id)
    );
    """)

    with driver.session(database="neo4j") as session:

        # Employees
        result = session.run("""
            MATCH (e:Employee)
            OPTIONAL MATCH (e)-[:REPORTS_TO]->(mgr:Employee)
            RETURN e.employeeID as eid, e.firstName as fn, e.lastName as ln,
                   e.title as title, e.city as city, e.country as country,
                   e.hireDate as hd, mgr.employeeID as mgr_id
        """)
        seen_emp = set()
        for r in result:
            eid = int(r["eid"]) if r["eid"] else None
            if eid in seen_emp:
                continue
            seen_emp.add(eid)
            mgr_id = int(r["mgr_id"]) if r["mgr_id"] else None
            cur.execute("INSERT OR IGNORE INTO employees VALUES (?,?,?,?,?,?,?,?)",
                        (eid, r["fn"], r["ln"], r["title"], r["city"], r["country"],
                         str(r["hd"])[:10] if r["hd"] else None, mgr_id))

        # Categories
        result = session.run("MATCH (c:Category) RETURN c.categoryID as cid, c.categoryName as name")
        seen_cat = set()
        for r in result:
            cid = int(r["cid"]) if r["cid"] else None
            if cid in seen_cat:
                continue
            seen_cat.add(cid)
            cur.execute("INSERT OR IGNORE INTO categories VALUES (?,?)", (cid, r["name"]))

        # Suppliers
        result = session.run("""
            MATCH (s:Supplier)
            RETURN s.supplierID as sid, s.companyName as name, s.country as country, s.city as city
        """)
        seen_sup = set()
        for r in result:
            sid = int(r["sid"]) if r["sid"] else None
            if sid in seen_sup:
                continue
            seen_sup.add(sid)
            cur.execute("INSERT OR IGNORE INTO suppliers VALUES (?,?,?,?)",
                        (sid, r["name"], r["country"], r["city"]))

        # Customers
        result = session.run("""
            MATCH (c:Customer)
            RETURN c.customerID as cid, c.companyName as name, c.country as country, c.city as city
        """)
        seen_cust = set()
        for r in result:
            cid = r["cid"]
            if cid in seen_cust:
                continue
            seen_cust.add(cid)
            cur.execute("INSERT OR IGNORE INTO customers VALUES (?,?,?,?)",
                        (cid, r["name"], r["country"], r["city"]))

        # Shippers
        result = session.run("MATCH (s:Shipper) RETURN s.shipperID as sid, s.companyName as name, s.phone as phone")
        seen_ship = set()
        for r in result:
            sid = int(r["sid"]) if r["sid"] else None
            if sid in seen_ship:
                continue
            seen_ship.add(sid)
            cur.execute("INSERT OR IGNORE INTO shippers VALUES (?,?,?)", (sid, r["name"], r["phone"]))

        # Products
        result = session.run("""
            MATCH (p:Product)
            OPTIONAL MATCH (p)-[:PART_OF]->(c:Category)
            OPTIONAL MATCH (s:Supplier)-[:SUPPLIES]->(p)
            RETURN p.productID as pid, p.productName as name, p.unitPrice as price,
                   p.unitsInStock as stock, p.discontinued as disc,
                   c.categoryID as cid, s.supplierID as sid
        """)
        seen_prod = set()
        for r in result:
            pid = int(r["pid"]) if r["pid"] else None
            if pid in seen_prod:
                continue
            seen_prod.add(pid)
            cid = int(r["cid"]) if r["cid"] else None
            sid = int(r["sid"]) if r["sid"] else None
            disc = 1 if r["disc"] else 0
            price = float(r["price"]) if r["price"] else 0.0
            stock = int(r["stock"]) if r["stock"] else 0
            cur.execute("INSERT OR IGNORE INTO products VALUES (?,?,?,?,?,?,?)",
                        (pid, r["name"], price, stock, disc, cid, sid))

        # Orders (deduplicated by orderID)
        result = session.run("""
            MATCH (o:Order)
            OPTIONAL MATCH (o)-[:ORDERED_BY]->(c:Customer)
            OPTIONAL MATCH (o)-[:PROCESSED_BY]->(e:Employee)
            OPTIONAL MATCH (o)-[:SHIPPED_BY]->(s:Shipper)
            RETURN o.orderID as oid, c.customerID as cid, e.employeeID as eid,
                   o.orderDate as odate, o.shipCountry as country, o.shipCity as city,
                   s.shipperID as sid
        """)
        seen_ord = set()
        for r in result:
            oid = int(r["oid"]) if r["oid"] else None
            if oid is None or oid in seen_ord:
                continue
            seen_ord.add(oid)
            eid = int(r["eid"]) if r["eid"] else None
            sid = int(r["sid"]) if r["sid"] else None
            cur.execute("INSERT OR IGNORE INTO orders VALUES (?,?,?,?,?,?,?)",
                        (oid, r["cid"], eid,
                         str(r["odate"])[:10] if r["odate"] else None,
                         r["country"], r["city"], sid))

        # Order details
        result = session.run("""
            MATCH (o:Order)-[r:INCLUDES]->(p:Product)
            RETURN o.orderID as oid, p.productID as pid,
                   r.unitPrice as price, r.quantity as qty, r.discount as disc
        """)
        seen_det = set()
        for r in result:
            oid = int(r["oid"]) if r["oid"] else None
            pid = int(r["pid"]) if r["pid"] else None
            if (oid, pid) in seen_det or oid is None or pid is None:
                continue
            seen_det.add((oid, pid))
            price = float(r["price"]) if r["price"] else 0.0
            qty = int(r["qty"]) if r["qty"] else 0
            disc = float(r["disc"]) if r["disc"] else 0.0
            cur.execute("INSERT OR IGNORE INTO order_details VALUES (?,?,?,?,?)",
                        (oid, pid, price, qty, disc))

    conn.commit()
    conn.close()


def print_summary():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for table in ["employees", "categories", "suppliers", "customers", "shippers", "products", "orders", "order_details"]:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        print(f"  {table}: {cur.fetchone()[0]} rows")
    conn.close()


if __name__ == "__main__":
    print(f"Building {DB_PATH} ...")
    try:
        driver = get_driver()
        build_sqlite(driver)
        driver.close()
        print("Done. Table counts:")
        print_summary()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
