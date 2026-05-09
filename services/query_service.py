def detect_query_intent(query):
    query = query.lower()

    location_words = [
        "vart",
        "var i texten",
        "vart i texten",
        "på vilken sida",
        "hitta",
        "var står",
        "vart står"
    ]

    for word in location_words:
        if word in query:
            return "location"

    return "information"


def expand_query(query):
    query_lower = query.lower()
    expanded_terms = []

    if "risk" in query_lower or "risker" in query_lower or "gå fel" in query_lower:
        expanded_terms.extend([
            "risk",
            "risker",
            "problem",
            "utmaningar",
            "hinder",
            "säkerhetsrisk",
            "kan gå fel"
        ])

    if "slutsats" in query_lower:
        expanded_terms.extend([
            "slutsats",
            "rekommendation",
            "bör",
            "sammanfattningsvis"
        ])

    if "mål" in query_lower or "syfte" in query_lower:
        expanded_terms.extend([
            "mål",
            "syfte",
            "avsikt",
            "anledning"
        ])

    if not expanded_terms:
        return query

    return query + " " + " ".join(expanded_terms)