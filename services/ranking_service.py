IMPORTANT_KEYWORDS = [
    "viktigt", "viktig", "slutsats", "resultat", "problem",
    "risk", "rekommendation", "mål", "syfte", "krav",
    "beslut", "lärdomar", "nästa steg", "sammanfattning"
]


def score_chunk(chunk):
    text = chunk["text"].lower()
    score = 0

    # Keyword-score
    for keyword in IMPORTANT_KEYWORDS:
        if keyword in text:
            score += 3

    # Längd-score
    word_count = len(text.split())
    if word_count > 80:
        score += 1
    if word_count > 150:
        score += 1

    # Tidigt i dokumentet är ofta viktigt
    if chunk["id"] <= 3:
        score += 1

    return score


def get_important_chunks(chunks, limit=5):
    scored = []

    for chunk in chunks:
        scored.append({
            **chunk,
            "score": score_chunk(chunk)
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    return scored[:limit]