from sentence_transformers import SentenceTransformer
from services.text_utils import split_into_sentences
import numpy as np
import re

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

STOPWORDS = {
    "vad", "är", "i", "på", "om", "och", "som", "det", "den",
    "de", "finns", "vilka", "hur", "varför", "med", "för",
    "vart", "var", "texten", "sida", "vilken"
}


def create_embeddings(chunks):
    texts = [chunk["text"] for chunk in chunks]
    return model.encode(texts, normalize_embeddings=True)


def extract_keywords(text):
    words = re.findall(r"\w+", text.lower())
    return [
        word for word in words
        if word not in STOPWORDS and len(word) > 2
    ]


def get_location_keywords(query):
    keywords = extract_keywords(query)

    expansions = {
        "sammanfattning": ["sammanfattning", "sammanfattningsvis"],
        "slutsats": ["slutsats", "slutsatsen", "bör", "projektet bör"],
        "risk": ["risk", "risker", "säkerhetsrisk"],
        "risker": ["risk", "risker", "säkerhetsrisk"],
        "mål": ["mål", "syfte"],
        "syfte": ["syfte", "mål"],
        "ställningstagande": [
            "ställningstagande",
            "bör",
            "rekommenderas",
            "slutsats",
            "projektet bör"
        ]
    }

    expanded = []

    for keyword in keywords:
        if keyword in expansions:
            expanded.extend(expansions[keyword])
        else:
            expanded.append(keyword)

    return list(dict.fromkeys(expanded))


def keyword_score(query, chunk_text, query_intent="information"):
    if query_intent == "location":
        query_keywords = get_location_keywords(query)
    else:
        query_keywords = extract_keywords(query)

    chunk_text = chunk_text.lower()
    score = 0

    for keyword in query_keywords:
        if keyword in chunk_text:
            score += 0.25

    return score

def get_location_sentences(query, chunk_text, max_sentences=3):
    sentences = split_into_sentences(chunk_text)
    location_keywords = get_location_keywords(query)

    matches = []

    for sentence in sentences:
        sentence_lower = sentence.lower()

        for keyword in location_keywords:
            if keyword in sentence_lower:
                matches.append(sentence)
                break

    return matches[:max_sentences]


def get_best_sentences(query, chunk_text, max_sentences=3):
    sentences = split_into_sentences(chunk_text)

    if not sentences:
        return [chunk_text]

    sentence_embeddings = model.encode(
        sentences,
        normalize_embeddings=True
    )

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    semantic_scores = np.dot(sentence_embeddings, query_embedding)

    sentence_results = []

    for index, sentence in enumerate(sentences):
        s_keyword_score = keyword_score(query, sentence)

        combined_score = (
            float(semantic_scores[index]) * 0.7
            +
            s_keyword_score * 0.3
        )

        sentence_results.append({
            "index": index,
            "sentence": sentence,
            "score": combined_score
        })

    sentence_results.sort(key=lambda x: x["score"], reverse=True)

    selected_indexes = sorted([
        result["index"]
        for result in sentence_results[:max_sentences]
    ])

    grouped_sentences = []
    current_group = []
    previous_index = None

    for index in selected_indexes:
        sentence = sentences[index]

        if previous_index is not None and index == previous_index + 1:
            current_group.append(sentence)
        else:
            if current_group:
                grouped_sentences.append(" ".join(current_group))

            current_group = [sentence]

        previous_index = index

    if current_group:
        grouped_sentences.append(" ".join(current_group))

    return grouped_sentences


def calculate_confidence(semantic_score, keyword_score_value, best_sentences, query_intent):
    has_evidence = bool(best_sentences)

    if not has_evidence:
        return "Låg säkerhet", "low-confidence", "low"

    if query_intent == "location":
        if keyword_score_value >= 0.25:
            return "Hög säkerhet", "high-confidence", "high"

        if semantic_score >= 0.35:
            return "Medel säkerhet", "medium-confidence", "medium"

        return "Låg säkerhet", "low-confidence", "low"

    if semantic_score >= 0.45 and keyword_score_value >= 0.25:
        return "Hög säkerhet", "high-confidence", "high"

    if semantic_score >= 0.35 or keyword_score_value >= 0.25:
        return "Medel säkerhet", "medium-confidence", "medium"

    return "Låg säkerhet", "low-confidence", "low"


def add_sentences_to_chunks(chunks):
    updated_chunks = []

    for chunk in chunks:
        updated_chunks.append({
            **chunk,
            "sentences": split_into_sentences(chunk["text"])
        })

    return updated_chunks


def search_chunks(
    query,
    chunks,
    embeddings,
    limit=5,
    min_score=0.45,
    keyword_weight=0.3,
    semantic_weight=0.7,
    query_intent="information"
):
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    semantic_scores = np.dot(embeddings, query_embedding)

    results = []

    for index, chunk in enumerate(chunks):
        semantic_score = float(semantic_scores[index])

        k_score = keyword_score(
            query,
            chunk["text"],
            query_intent=query_intent
        )

        final_score = (
            semantic_score * semantic_weight
            +
            k_score * keyword_weight
        )

        if query_intent == "location":
            best_sentences = get_location_sentences(
                query,
                chunk["text"],
                max_sentences=3
            )
        else:
            best_sentences = get_best_sentences(
                query,
                chunk["text"],
                max_sentences=3
            )

        confidence_label, confidence_class, confidence_level = calculate_confidence(
            semantic_score,
            k_score,
            best_sentences,
            query_intent
        )

        results.append({
            **chunk,
            "semantic_score": round(semantic_score, 3),
            "keyword_score": round(k_score, 3),
            "final_score": round(final_score, 3),
            "best_sentences": best_sentences,
            "confidence_label": confidence_label,
            "confidence_class": confidence_class,
            "confidence_level": confidence_level
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)

    filtered_results = [
        result for result in results
        if result["final_score"] >= min_score
           and result["best_sentences"]
    ]

    return filtered_results[:limit]