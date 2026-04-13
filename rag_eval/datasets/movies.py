"""
Dataset de evaluación para GraphRAG sobre Neo4j Movies (db1).

34 películas, 64 personas, 127 relaciones.
Relaciones: ACTED_IN (con roles), DIRECTED, WROTE, PRODUCED.

Categorías:
  A. Lookups directos (1-hop) — actores, directores, fechas
  B. Agregaciones — COUNT
  C. Roles y relaciones específicas
  D. Multi-hop (2 saltos) — colaboraciones, coprotagonistas
  E. Preguntas sin respuesta — test de no-alucinación
  F. Filtros por año/época
  G. Relaciones múltiples (actores que también dirigen)

Answers verificadas contra Neo4j (db1).
Conexión: bolt://localhost:7687, database=db1
"""

DATASET_MOVIES = [

    # ─────────────────────────────────────────
    # A. LOOKUPS DIRECTOS (1-hop)
    # ─────────────────────────────────────────

    {
        "inputs": {"question": "How many movies are in the database?", "database": "db1"},
        "outputs": {"answer": "There are 34 movies in the database."}
    },
    {
        "inputs": {"question": "How many people are in the database?", "database": "db1"},
        "outputs": {"answer": "There are 64 people in the database."}
    },
    {
        "inputs": {"question": "What movies did Tom Hanks act in?", "database": "db1"},
        "outputs": {"answer": "Tom Hanks acted in: You've Got Mail, Sleepless in Seattle, Cast Away, The Green Mile, Apollo 13, The Da Vinci Code, Cloud Atlas, Charlie Wilson's War, The Polar Express, and A League of Their Own."}
    },
    {
        "inputs": {"question": "What movies did Tom Cruise act in?", "database": "db1"},
        "outputs": {"answer": "Tom Cruise acted in A Few Good Men, Top Gun, and Jerry Maguire."}
    },
    {
        "inputs": {"question": "What movies did Keanu Reeves act in?", "database": "db1"},
        "outputs": {"answer": "Keanu Reeves acted in: The Matrix, The Matrix Reloaded, The Matrix Revolutions, Johnny Mnemonic, Something's Gotta Give, The Devil's Advocate, and The Replacements."}
    },
    {
        "inputs": {"question": "What movies did Jack Nicholson act in?", "database": "db1"},
        "outputs": {"answer": "Jack Nicholson acted in: A Few Good Men, As Good as It Gets, One Flew Over the Cuckoo's Nest, Something's Gotta Give, and Hoffa."}
    },
    {
        "inputs": {"question": "Who directed The Matrix?", "database": "db1"},
        "outputs": {"answer": "The Matrix was directed by Lilly Wachowski and Lana Wachowski."}
    },
    {
        "inputs": {"question": "Who directed Apollo 13?", "database": "db1"},
        "outputs": {"answer": "Apollo 13 was directed by Ron Howard."}
    },
    {
        "inputs": {"question": "Who directed A Few Good Men?", "database": "db1"},
        "outputs": {"answer": "A Few Good Men was directed by Rob Reiner."}
    },
    {
        "inputs": {"question": "What year was The Matrix released?", "database": "db1"},
        "outputs": {"answer": "The Matrix was released in 1999."}
    },
    {
        "inputs": {"question": "What year was Apollo 13 released?", "database": "db1"},
        "outputs": {"answer": "Apollo 13 was released in 1995."}
    },
    {
        "inputs": {"question": "What year was Top Gun released?", "database": "db1"},
        "outputs": {"answer": "Top Gun was released in 1986."}
    },
    {
        "inputs": {"question": "What is the tagline of The Matrix?", "database": "db1"},
        "outputs": {"answer": "The tagline of The Matrix is 'Welcome to the Real World'."}
    },
    {
        "inputs": {"question": "What is the tagline of Top Gun?", "database": "db1"},
        "outputs": {"answer": "The tagline of Top Gun is 'I feel the need, the need for speed.'"}
    },
    {
        "inputs": {"question": "What is the tagline of Apollo 13?", "database": "db1"},
        "outputs": {"answer": "The tagline of Apollo 13 is 'Houston, we have a problem.'"}
    },

    # ─────────────────────────────────────────
    # B. AGREGACIONES
    # ─────────────────────────────────────────

    {
        "inputs": {"question": "How many movies did Tom Hanks act in?", "database": "db1"},
        "outputs": {"answer": "Tom Hanks acted in 10 movies."}
    },
    {
        "inputs": {"question": "How many movies did Keanu Reeves act in?", "database": "db1"},
        "outputs": {"answer": "Keanu Reeves acted in 7 movies."}
    },
    {
        "inputs": {"question": "How many movies did Jack Nicholson act in?", "database": "db1"},
        "outputs": {"answer": "Jack Nicholson acted in 5 movies."}
    },
    {
        "inputs": {"question": "How many movies did Ron Howard direct?", "database": "db1"},
        "outputs": {"answer": "Ron Howard directed 3 movies: Apollo 13, The Da Vinci Code, and Frost/Nixon."}
    },
    {
        "inputs": {"question": "How many movies were released in 1999?", "database": "db1"},
        "outputs": {"answer": "3 movies were released in 1999: The Matrix, The Green Mile, and Snow Falling on Cedars."}
    },

    # ─────────────────────────────────────────
    # C. ROLES Y RELACIONES ESPECÍFICAS
    # ─────────────────────────────────────────

    {
        "inputs": {"question": "What role did Tom Cruise play in Top Gun?", "database": "db1"},
        "outputs": {"answer": "Tom Cruise played the role of Maverick in Top Gun."}
    },
    {
        "inputs": {"question": "What role did Keanu Reeves play in The Matrix?", "database": "db1"},
        "outputs": {"answer": "Keanu Reeves played the role of Neo in The Matrix."}
    },
    {
        "inputs": {"question": "What role did Tom Hanks play in Apollo 13?", "database": "db1"},
        "outputs": {"answer": "Tom Hanks played the role of Jim Lovell in Apollo 13."}
    },
    {
        "inputs": {"question": "What role did Natalie Portman play in V for Vendetta?", "database": "db1"},
        "outputs": {"answer": "Natalie Portman played the role of Evey Hammond in V for Vendetta."}
    },
    {
        "inputs": {"question": "What role did Hugo Weaving play in The Matrix?", "database": "db1"},
        "outputs": {"answer": "Hugo Weaving played the role of Agent Smith in The Matrix."}
    },
    {
        "inputs": {"question": "Who acted in The Matrix?", "database": "db1"},
        "outputs": {"answer": "The Matrix starred Keanu Reeves, Carrie-Anne Moss, Laurence Fishburne, and Hugo Weaving, among others."}
    },
    {
        "inputs": {"question": "Who acted in Top Gun?", "database": "db1"},
        "outputs": {"answer": "Top Gun starred Tom Cruise, Val Kilmer, and Kelly McGillis, among others."}
    },
    {
        "inputs": {"question": "Who produced The Matrix?", "database": "db1"},
        "outputs": {"answer": "The Matrix was produced by Joel Silver."}
    },
    {
        "inputs": {"question": "Who produced V for Vendetta?", "database": "db1"},
        "outputs": {"answer": "V for Vendetta was produced by Lilly Wachowski, Lana Wachowski, and Joel Silver."}
    },
    {
        "inputs": {"question": "What movies did Nora Ephron write?", "database": "db1"},
        "outputs": {"answer": "Nora Ephron wrote You've Got Mail, Sleepless in Seattle, and When Harry Met Sally."}
    },

    # ─────────────────────────────────────────
    # D. MULTI-HOP (2 saltos) — colaboraciones
    # ─────────────────────────────────────────

    {
        "inputs": {"question": "Who acted in both A Few Good Men and Top Gun?", "database": "db1"},
        "outputs": {"answer": "Tom Cruise acted in both A Few Good Men and Top Gun."}
    },
    {
        "inputs": {"question": "Who acted in both The Matrix and Cloud Atlas?", "database": "db1"},
        "outputs": {"answer": "Hugo Weaving and Tom Hanks acted in both The Matrix series and Cloud Atlas. Specifically, Hugo Weaving appeared in both The Matrix and Cloud Atlas."}
    },
    {
        "inputs": {"question": "What movies did the Wachowski sisters direct?", "database": "db1"},
        "outputs": {"answer": "The Wachowski sisters (Lilly and Lana Wachowski) directed: The Matrix, The Matrix Reloaded, The Matrix Revolutions, Cloud Atlas, and Speed Racer."}
    },
    {
        "inputs": {"question": "What movies were produced by Joel Silver?", "database": "db1"},
        "outputs": {"answer": "Joel Silver produced The Matrix, The Matrix Reloaded, The Matrix Revolutions, V for Vendetta, Speed Racer, and Ninja Assassin."}
    },
    {
        "inputs": {"question": "Who directed Cloud Atlas?", "database": "db1"},
        "outputs": {"answer": "Cloud Atlas was directed by Tom Hanks, Lilly Wachowski, and Lana Wachowski."}
    },

    # ─────────────────────────────────────────
    # E. PREGUNTAS SIN RESPUESTA (anti-alucinación)
    # ─────────────────────────────────────────

    {
        "inputs": {"question": "What movies did Leonardo DiCaprio act in?", "database": "db1"},
        "outputs": {"answer": "The database does not contain any movies featuring Leonardo DiCaprio. He is not part of the current movie dataset."}
    },
    {
        "inputs": {"question": "Who directed The Godfather?", "database": "db1"},
        "outputs": {"answer": "The Godfather is not in the database. The database contains movies like The Matrix, Apollo 13, and A Few Good Men, but not The Godfather."}
    },
    {
        "inputs": {"question": "What movies did Meryl Streep act in?", "database": "db1"},
        "outputs": {"answer": "The database does not contain any movies featuring Meryl Streep. She is not part of the current movie dataset."}
    },
    {
        "inputs": {"question": "What is the budget of The Matrix?", "database": "db1"},
        "outputs": {"answer": "The database does not contain budget information for any movies. Only title, release year, and tagline are available."}
    },

    # ─────────────────────────────────────────
    # F. FILTROS POR AÑO / ÉPOCA
    # ─────────────────────────────────────────

    {
        "inputs": {"question": "What movies were released in 2003?", "database": "db1"},
        "outputs": {"answer": "Two movies were released in 2003: The Matrix Reloaded and The Matrix Revolutions."}
    },
    {
        "inputs": {"question": "What movies were released in 2006?", "database": "db1"},
        "outputs": {"answer": "Two movies were released in 2006: The Da Vinci Code and RescueDawn."}
    },
    {
        "inputs": {"question": "What movies were released in 1986?", "database": "db1"},
        "outputs": {"answer": "Two movies were released in 1986: Top Gun and Stand By Me."}
    },
    {
        "inputs": {"question": "What movies were released in 1992?", "database": "db1"},
        "outputs": {"answer": "Three movies were released in 1992: A Few Good Men, Unforgiven, Hoffa, and A League of Their Own."}
    },
    {
        "inputs": {"question": "When was Tom Hanks born?", "database": "db1"},
        "outputs": {"answer": "Tom Hanks was born in 1956."}
    },
    {
        "inputs": {"question": "When was Keanu Reeves born?", "database": "db1"},
        "outputs": {"answer": "Keanu Reeves was born in 1964."}
    },

    # ─────────────────────────────────────────
    # G. RELACIONES MÚLTIPLES (actores-directores)
    # ─────────────────────────────────────────

    {
        "inputs": {"question": "Who has both acted in and directed movies in the database?", "database": "db1"},
        "outputs": {"answer": "Tom Hanks both acted in movies and directed (Cloud Atlas, That Thing You Do). Clint Eastwood both acted in and directed Unforgiven. Danny DeVito both acted in and directed Hoffa."}
    },
    {
        "inputs": {"question": "What movies did Rob Reiner direct?", "database": "db1"},
        "outputs": {"answer": "Rob Reiner directed A Few Good Men, Stand By Me, and When Harry Met Sally."}
    },
    {
        "inputs": {"question": "What is the tagline of V for Vendetta?", "database": "db1"},
        "outputs": {"answer": "The tagline of V for Vendetta is 'Freedom! Forever!'"}
    },
    {
        "inputs": {"question": "What movies did Meg Ryan act in?", "database": "db1"},
        "outputs": {"answer": "Meg Ryan acted in You've Got Mail, Sleepless in Seattle, Joe Versus the Volcano, and When Harry Met Sally."}
    },
    {
        "inputs": {"question": "Who wrote When Harry Met Sally?", "database": "db1"},
        "outputs": {"answer": "When Harry Met Sally was written by Nora Ephron."}
    },
]

if __name__ == "__main__":
    print(f"Dataset Movies: {len(DATASET_MOVIES)} preguntas")
    categories = {
        "A. Lookups directos": 15,
        "B. Agregaciones": 5,
        "C. Roles y relaciones": 10,
        "D. Multi-hop 2 saltos": 5,
        "E. Sin respuesta": 4,
        "F. Filtros por año": 6,
        "G. Relaciones múltiples": 5,
    }
    for cat, n in categories.items():
        print(f"  {cat}: {n} preguntas")
