from services.text_utils import split_into_sentences
import numpy as np


def build_sentence_index(chunks, model):
    sentence_items = []

    for chunk in chunks:
        sentences = split_into_sentences(chunk["text"])

        for sentence in sentences:
            sentence_items.append({
                "chunk_id": chunk["id"],
                "page": chunk["page"],
                "sentence": sentence
            })

    texts = [item["sentence"] for item in sentence_items]

    if not texts:
        return [], None

    embeddings = model.encode(texts, normalize_embeddings=True)

    return sentence_items, embeddings


def search_sentences(query, sentence_items, sentence_embeddings, model, limit=12):
    if not sentence_items or sentence_embeddings is None:
        return []

    query_embedding = model.encode([query], normalize_embeddings=True)[0]

    scores = np.dot(sentence_embeddings, query_embedding)

    ranked_indexes = np.argsort(scores)[::-1][:limit]

    results = []

    for index in ranked_indexes:
        item = sentence_items[index]

        results.append({
            **item,
            "sentence_score": round(float(scores[index]), 3)
        })

    return results


def improve_results_with_sentence_hits(semantic_results, sentence_results, max_sentences_per_chunk=3):
    """
    Tar chunk-resultat från retrieval och ersätter deras evidens
    med mer exakta meningar från sentence-level search.

    Detta gör:
    - highlight mer exakt
    - AI-svaret får mindre onödig text
    - felaktiga delar av chunken dras inte med lika lätt
    """

    if not semantic_results or not sentence_results:
        return semantic_results

    sentences_by_chunk = {}

    for sentence_result in sentence_results:
        chunk_id = sentence_result["chunk_id"]

        if chunk_id not in sentences_by_chunk:
            sentences_by_chunk[chunk_id] = []

        sentences_by_chunk[chunk_id].append(sentence_result)

    improved_results = []

    for result in semantic_results:
        chunk_id = result["id"]

        if chunk_id not in sentences_by_chunk:
            improved_results.append(result)
            continue

        best_sentence_hits = sentences_by_chunk[chunk_id][:max_sentences_per_chunk]

        highlight_sentences = [
            item["sentence"]
            for item in best_sentence_hits
        ]

        display_evidence = highlight_sentences.copy()

        avg_sentence_score = sum(
            item["sentence_score"]
            for item in best_sentence_hits
        ) / len(best_sentence_hits)

        improved_results.append({
            **result,
            "highlight_sentences": highlight_sentences,
            "display_evidence": display_evidence,
            "best_sentences": display_evidence,
            "sentence_score": round(avg_sentence_score, 3)
        })

    return improved_results