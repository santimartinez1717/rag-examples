"""
Dataset de evaluación para GraphRAG sobre Game of Thrones (db2).

40 personajes, 10 localizaciones, 9 casas nobles.
Relaciones: MEMBER_OF, PARENT_OF, SIBLING_OF, MARRIED_TO, KILLED, CONTROLS, SERVES.

Categorías:
  A. Lookups directos (casas, localizaciones, personajes)
  B. Relaciones familiares (PARENT_OF, SIBLING_OF)
  C. Multi-hop (miembros de casa → localización)
  D. Preguntas de linaje complejo (Jon Snow, parentesco)
  E. Preguntas sin respuesta — test anti-alucinación
  F. Muertes y conflictos
  G. Preguntas de conteo y aggregación

Answers verificadas contra Neo4j (db2).
Conexión: bolt://localhost:7687, database=db2
"""

DATASET_GOT = [

    # ─────────────────────────────────────────
    # A. LOOKUPS DIRECTOS
    # ─────────────────────────────────────────

    {
        "inputs": {"question": "How many houses are in the database?", "database": "db2"},
        "outputs": {"answer": "There are 9 noble houses in the database: Stark, Lannister, Targaryen, Baratheon, Tyrell, Greyjoy, Tully, Arryn, and Martell."}
    },
    {
        "inputs": {"question": "What is House Stark's words (motto)?", "database": "db2"},
        "outputs": {"answer": "House Stark's words are 'Winter is Coming'."}
    },
    {
        "inputs": {"question": "What is House Lannister's motto?", "database": "db2"},
        "outputs": {"answer": "House Lannister's motto is 'Hear Me Roar'."}
    },
    {
        "inputs": {"question": "What is House Targaryen's words?", "database": "db2"},
        "outputs": {"answer": "House Targaryen's words are 'Fire and Blood'."}
    },
    {
        "inputs": {"question": "What is House Baratheon's words?", "database": "db2"},
        "outputs": {"answer": "House Baratheon's words are 'Ours is the Fury'."}
    },
    {
        "inputs": {"question": "What castle does House Stark control?", "database": "db2"},
        "outputs": {"answer": "House Stark controls Winterfell, located in The North."}
    },
    {
        "inputs": {"question": "What castle does House Lannister control?", "database": "db2"},
        "outputs": {"answer": "House Lannister controls Casterly Rock, located in the Westerlands."}
    },
    {
        "inputs": {"question": "What castle does House Targaryen control?", "database": "db2"},
        "outputs": {"answer": "House Targaryen controls Dragonstone, located in the Crownlands."}
    },
    {
        "inputs": {"question": "What is Tyrion Lannister's nickname?", "database": "db2"},
        "outputs": {"answer": "Tyrion Lannister's nickname is 'The Imp'."}
    },
    {
        "inputs": {"question": "What is Daenerys Targaryen's nickname?", "database": "db2"},
        "outputs": {"answer": "Daenerys Targaryen's nickname is 'Mother of Dragons'."}
    },
    {
        "inputs": {"question": "What house does Theon Greyjoy belong to?", "database": "db2"},
        "outputs": {"answer": "Theon Greyjoy belongs to House Greyjoy."}
    },
    {
        "inputs": {"question": "What is the status of Eddard Stark?", "database": "db2"},
        "outputs": {"answer": "Eddard Stark (Ned Stark) is deceased."}
    },
    {
        "inputs": {"question": "What title does Jon Snow hold?", "database": "db2"},
        "outputs": {"answer": "Jon Snow holds the title of King in the North."}
    },

    # ─────────────────────────────────────────
    # B. RELACIONES FAMILIARES
    # ─────────────────────────────────────────

    {
        "inputs": {"question": "Who are the children of Eddard Stark?", "database": "db2"},
        "outputs": {"answer": "Eddard Stark's children are: Robb Stark, Sansa Stark, Arya Stark, Bran Stark, and Rickon Stark."}
    },
    {
        "inputs": {"question": "Who are the children of Tywin Lannister?", "database": "db2"},
        "outputs": {"answer": "Tywin Lannister's children are: Cersei Lannister, Jaime Lannister, and Tyrion Lannister."}
    },
    {
        "inputs": {"question": "Who are the children of Aerys Targaryen?", "database": "db2"},
        "outputs": {"answer": "Aerys Targaryen's (the Mad King) children are: Rhaegar Targaryen, Viserys Targaryen, and Daenerys Targaryen."}
    },
    {
        "inputs": {"question": "Who are the siblings of Cersei Lannister?", "database": "db2"},
        "outputs": {"answer": "Cersei Lannister's siblings are Jaime Lannister and Tyrion Lannister."}
    },
    {
        "inputs": {"question": "Who are the siblings of Robert Baratheon?", "database": "db2"},
        "outputs": {"answer": "Robert Baratheon's siblings are Stannis Baratheon and Renly Baratheon."}
    },
    {
        "inputs": {"question": "Who is Jon Snow's real father?", "database": "db2"},
        "outputs": {"answer": "Jon Snow's real father is Rhaegar Targaryen. His mother was Lyanna Stark, making him the legitimate heir to the Iron Throne."}
    },
    {
        "inputs": {"question": "Who are the parents of Jon Snow?", "database": "db2"},
        "outputs": {"answer": "Jon Snow's true parents are Rhaegar Targaryen and Lyanna Stark. He was raised believing Eddard Stark was his father."}
    },
    {
        "inputs": {"question": "Who is married to Eddard Stark?", "database": "db2"},
        "outputs": {"answer": "Eddard Stark is married to Catelyn Tully."}
    },
    {
        "inputs": {"question": "Who is married to Robert Baratheon?", "database": "db2"},
        "outputs": {"answer": "Robert Baratheon is married to Cersei Lannister."}
    },

    # ─────────────────────────────────────────
    # C. MULTI-HOP (Casa → Localización → Personaje)
    # ─────────────────────────────────────────

    {
        "inputs": {"question": "Who are the members of House Stark?", "database": "db2"},
        "outputs": {"answer": "House Stark members include: Eddard Stark, Robb Stark, Sansa Stark, Arya Stark, Bran Stark, Rickon Stark, Jon Snow, Lyanna Stark, and Benjen Stark."}
    },
    {
        "inputs": {"question": "Who are the members of House Lannister?", "database": "db2"},
        "outputs": {"answer": "House Lannister members include: Tywin Lannister, Cersei Lannister, Jaime Lannister, and Tyrion Lannister."}
    },
    {
        "inputs": {"question": "Who are the members of House Targaryen?", "database": "db2"},
        "outputs": {"answer": "House Targaryen members include: Aerys Targaryen (the Mad King), Rhaegar Targaryen, Viserys Targaryen, and Daenerys Targaryen."}
    },
    {
        "inputs": {"question": "Who are the members of House Baratheon?", "database": "db2"},
        "outputs": {"answer": "House Baratheon members include: Robert Baratheon, Stannis Baratheon, Renly Baratheon, Joffrey Baratheon, Myrcella Baratheon, and Tommen Baratheon."}
    },
    {
        "inputs": {"question": "What region does House Greyjoy rule?", "database": "db2"},
        "outputs": {"answer": "House Greyjoy rules the Iron Islands, with their seat at Pyke."}
    },
    {
        "inputs": {"question": "What region does House Tyrell control?", "database": "db2"},
        "outputs": {"answer": "House Tyrell controls The Reach, with their seat at Highgarden."}
    },

    # ─────────────────────────────────────────
    # D. LINAJE COMPLEJO
    # ─────────────────────────────────────────

    {
        "inputs": {"question": "Is Jon Snow a Targaryen?", "database": "db2"},
        "outputs": {"answer": "Yes, Jon Snow is a Targaryen. His true father is Rhaegar Targaryen and his mother is Lyanna Stark, making him the legitimate heir to the Iron Throne. His real name is Aegon Targaryen."}
    },
    {
        "inputs": {"question": "Are Cersei and Jaime Lannister related?", "database": "db2"},
        "outputs": {"answer": "Yes, Cersei Lannister and Jaime Lannister are siblings — both are children of Tywin Lannister."}
    },
    {
        "inputs": {"question": "How many children does Cersei Lannister have?", "database": "db2"},
        "outputs": {"answer": "Cersei Lannister has 3 children: Joffrey Baratheon, Myrcella Baratheon, and Tommen Baratheon."}
    },
    {
        "inputs": {"question": "How many children does Aerys Targaryen have?", "database": "db2"},
        "outputs": {"answer": "Aerys Targaryen (the Mad King) has 3 children: Rhaegar, Viserys, and Daenerys Targaryen."}
    },

    # ─────────────────────────────────────────
    # E. PREGUNTAS SIN RESPUESTA
    # ─────────────────────────────────────────

    {
        "inputs": {"question": "Who is Hodor's real name?", "database": "db2"},
        "outputs": {"answer": "The database does not contain information about Hodor. The character Hodor is not in the current dataset."}
    },
    {
        "inputs": {"question": "What is House Stark's sigil?", "database": "db2"},
        "outputs": {"answer": "The database does not contain information about house sigils. Only house words (mottoes), seats, and regions are stored."}
    },
    {
        "inputs": {"question": "How many dragons does Daenerys have?", "database": "db2"},
        "outputs": {"answer": "The database does not contain information about dragons. Only people, houses, and locations are tracked in this knowledge graph."}
    },
    {
        "inputs": {"question": "Who sits on the Iron Throne at the end of the series?", "database": "db2"},
        "outputs": {"answer": "The database does not contain information about the final ruler of the Seven Kingdoms. The data focuses on character relationships, house memberships, and locations."}
    },

    # ─────────────────────────────────────────
    # F. MUERTES Y CONFLICTOS
    # ─────────────────────────────────────────

    {
        "inputs": {"question": "Who killed Tywin Lannister?", "database": "db2"},
        "outputs": {"answer": "Tywin Lannister was killed by Tyrion Lannister."}
    },
    {
        "inputs": {"question": "Who killed Aerys Targaryen (the Mad King)?", "database": "db2"},
        "outputs": {"answer": "Aerys Targaryen, the Mad King, was killed by Jaime Lannister — which earned Jaime the nickname 'Kingslayer'."}
    },
    {
        "inputs": {"question": "Who killed Daenerys Targaryen?", "database": "db2"},
        "outputs": {"answer": "Daenerys Targaryen was killed by Jon Snow."}
    },
    {
        "inputs": {"question": "Who killed Petyr Baelish?", "database": "db2"},
        "outputs": {"answer": "Petyr Baelish (Littlefinger) was killed by Arya Stark."}
    },

    # ─────────────────────────────────────────
    # G. CONTEO Y AGGREGACIÓN
    # ─────────────────────────────────────────

    {
        "inputs": {"question": "How many people are in the database?", "database": "db2"},
        "outputs": {"answer": "There are 40 people (characters) in the Game of Thrones database."}
    },
    {
        "inputs": {"question": "How many characters are alive?", "database": "db2"},
        "outputs": {"answer": "The characters with 'Alive' status include: Sansa Stark, Arya Stark, Bran Stark, Jon Snow, Tyrion Lannister, Yara Greyjoy, Brienne of Tarth, Sandor Clegane, Davos Seaworth, Samwell Tarly, and Tormund Giantsbane, among others."}
    },
    {
        "inputs": {"question": "How many characters are deceased?", "database": "db2"},
        "outputs": {"answer": "The majority of characters in the database have 'Deceased' status, including all kings of the Seven Kingdoms mentioned (Robert, Joffrey, Tommen, Daenerys), all Targaryen kings, Eddard Stark, Robb Stark, and many others."}
    },
    {
        "inputs": {"question": "Who serves House Stark?", "database": "db2"},
        "outputs": {"answer": "Brienne of Tarth serves House Stark."}
    },
    {
        "inputs": {"question": "How many siblings does Sansa Stark have?", "database": "db2"},
        "outputs": {"answer": "In the database, Sansa Stark's sibling is Robb Stark. Her other siblings (Arya, Bran, Rickon) are connected through their shared parents (Eddard and Catelyn Stark)."}
    },
]

if __name__ == "__main__":
    print(f"Dataset Game of Thrones: {len(DATASET_GOT)} preguntas")
    categories = {
        "A. Lookups directos": 13,
        "B. Relaciones familiares": 9,
        "C. Multi-hop (casa→localización)": 6,
        "D. Linaje complejo": 4,
        "E. Sin respuesta": 4,
        "F. Muertes y conflictos": 4,
        "G. Conteo y aggregación": 5,
    }
    for cat, n in categories.items():
        print(f"  {cat}: {n} preguntas")
