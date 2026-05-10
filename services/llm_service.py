import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


def build_evidence_context(evidence_chunks):
    context_parts = []

    for index, chunk in enumerate(evidence_chunks, start=1):
        best_sentences = chunk.get("best_sentences", [])

        if not best_sentences:
            continue

        evidence_text = " ".join(best_sentences)

        context_parts.append(
            f"Stöd {index} (sida {chunk['page']}): {evidence_text}"
        )

    return "\n\n".join(context_parts)


def generate_answer(query, evidence_chunks):
    context = build_evidence_context(evidence_chunks)

    prompt = f"""
Du är en försiktig dokumentassistent.

Du får bara använda evidensen nedan.
Du får inte lägga till egen kunskap.
Om evidensen inte räcker ska du skriva:
"Det framgår inte tydligt av dokumentet."

Fråga:
{query}

Evidens:
{context}

Skriv ett kort, naturligt svar på svenska.

Regler:
- Sammanfatta evidensen istället för att kopiera den ord för ord.
- Om flera punkter nämns, skriv dem tydligt i en kort lista.
- Om frågan handlar om risker, mål, problem eller rekommendationer: strukturera svaret i punkter.
- Max 5 meningar eller 4 punkter.
- Nämn inte "Stöd 1" eller interna scores i svaret.
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )

    data = response.json()

    return data["response"].strip()