def rerank_results(query, results, limit=6):
    query_words = set(query.lower().split())

    reranked = []
    seen_texts = set()

    for result in results:
        evidence_text = " ".join(result.get("best_sentences", []))
        normalized = " ".join(evidence_text.lower().split())

        if not normalized or normalized in seen_texts:
            continue

        # Filtrera bort träffar utan tydligt textstöd
        if result.get("keyword_score", 0) < 0.18:
            continue

        seen_texts.add(normalized)

        evidence_words = set(normalized.split())
        overlap = len(query_words.intersection(evidence_words))

        adjusted_score = (
            result.get("final_score", 0)
            + overlap * 0.15
            + result.get("keyword_score", 0) * 0.4
        )

        reranked.append({
            **result,
            "rerank_score": adjusted_score
        })

    reranked.sort(
        key=lambda x: x.get("rerank_score", 0),
        reverse=True
    )

    return reranked[:limit]