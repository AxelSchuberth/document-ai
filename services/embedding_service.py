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


def get_expanded_keywords(query):
    keywords = extract_keywords(query)

    expansions = {
        "sammanfattning": ["sammanfattning", "sammanfattningsvis", "översikt"],
        "slutsats": ["slutsats", "slutsatsen", "slutligen", "sammanfattningsvis", "rekommendation"],
        "risk": ["risk", "risker", "problem", "utmaning", "utmaningar", "hinder", "hot", "säkerhetsrisk"],
        "risker": ["risk", "risker", "problem", "utmaningar", "hinder", "hot", "säkerhetsrisk"],
        "problem": ["problem", "risk", "risker", "utmaning", "hinder", "svårighet"],
        "mål": ["mål", "syfte", "ambition", "avsikt"],
        "syfte": ["syfte", "mål", "avsikt", "anledning"],
        "ställningstagande": ["ställningstagande", "bör", "rekommenderas", "slutsats", "projektet bör"],
        "rekommendation": ["rekommendation", "bör", "förslag", "råd"],
        "kostnad": ["kostnad", "kostnader", "utgift", "budget", "pris"],
        "säkerhet": ["säkerhet", "risk", "skydd", "hot", "behörighet"],
    }

    expanded = []

    for keyword in keywords:
        if keyword in expansions:
            expanded.extend(expansions[keyword])
        else:
            expanded.append(keyword)

    return list(dict.fromkeys(expanded))


def keyword_score(query, text):
    query_keywords = get_expanded_keywords(query)
    text = text.lower()

    score = 0

    for keyword in query_keywords:
        if keyword in text:
            score += 0.18

    return score


def group_adjacent_sentences(selected_sentence_items):
    selected_sentence_items = sorted(
        selected_sentence_items,
        key=lambda item: item["index"]
    )

    groups = []
    current_group = []
    previous_index = None

    for item in selected_sentence_items:
        if previous_index is not None and item["index"] == previous_index + 1:
            current_group.append(item["sentence"])
        else:
            if current_group:
                groups.append(" ".join(current_group))

            current_group = [item["sentence"]]

        previous_index = item["index"]

    if current_group:
        groups.append(" ".join(current_group))

    return groups


def select_relevant_sentences(query, chunk_text, query_intent="information", max_sentences=3):
    sentences = split_into_sentences(chunk_text)

    if not sentences:
        return {
            "highlight_sentences": [],
            "display_evidence": []
        }

    if query_intent == "location":
        keywords = get_expanded_keywords(query)
        matched_items = []

        for index, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()

            if any(keyword in sentence_lower for keyword in keywords):
                matched_items.append({
                    "index": index,
                    "sentence": sentence,
                    "score": keyword_score(query, sentence)
                })

        selected_items = matched_items[:max_sentences]

    else:
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
                float(semantic_scores[index]) * 0.65
                +
                s_keyword_score * 0.35
            )

            sentence_results.append({
                "index": index,
                "sentence": sentence,
                "score": combined_score
            })

        sentence_results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        selected_items = [
            item for item in sentence_results[:max_sentences]
            if item["score"] >= 0.18
        ]

    highlight_sentences = [
        item["sentence"]
        for item in selected_items
    ]

    display_evidence = group_adjacent_sentences(selected_items)

    return {
        "highlight_sentences": highlight_sentences,
        "display_evidence": display_evidence
    }


def calculate_confidence(semantic_score, keyword_score_value, highlight_sentences, query_intent):
    has_evidence = bool(highlight_sentences)

    if not has_evidence:
        return "Låg säkerhet", "low-confidence", "low"

    if query_intent == "location":
        if keyword_score_value >= 0.18:
            return "Hög säkerhet", "high-confidence", "high"

        if semantic_score >= 0.35:
            return "Medel säkerhet", "medium-confidence", "medium"

        return "Låg säkerhet", "low-confidence", "low"

    if semantic_score >= 0.45 and keyword_score_value >= 0.18:
        return "Hög säkerhet", "high-confidence", "high"

    if semantic_score >= 0.35 or keyword_score_value >= 0.18:
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
        k_score = keyword_score(query, chunk["text"])

        final_score = (
            semantic_score * semantic_weight
            +
            k_score * keyword_weight
        )

        evidence = select_relevant_sentences(
            query,
            chunk["text"],
            query_intent=query_intent,
            max_sentences=3
        )

        highlight_sentences = evidence["highlight_sentences"]
        display_evidence = evidence["display_evidence"]

        confidence_label, confidence_class, confidence_level = calculate_confidence(
            semantic_score,
            k_score,
            highlight_sentences,
            query_intent
        )

        results.append({
            **chunk,
            "semantic_score": round(semantic_score, 3),
            "keyword_score": round(k_score, 3),
            "final_score": round(final_score, 3),
            "highlight_sentences": highlight_sentences,
            "display_evidence": display_evidence,
            "best_sentences": display_evidence,
            "confidence_label": confidence_label,
            "confidence_class": confidence_class,
            "confidence_level": confidence_level
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)

    filtered_results = [
        result for result in results
        if result["final_score"] >= min_score
        and result["highlight_sentences"]
    ]

    return filtered_results[:limit]