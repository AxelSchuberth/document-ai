def detect_query_intent(query):
    query = query.lower()

    location_words = [
        "vart",
        "var",
        "var i texten",
        "vart i texten",
        "på vilken sida",
        "hitta",
        "finns det",
        "var står",
        "vart står"
    ]

    for word in location_words:
        if word in query:
            return "location"

    return "information"